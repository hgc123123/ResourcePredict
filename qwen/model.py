from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


MODALITY_NAMES = ("code", "user", "metadata")


def _load_torch_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_modalities(modalities: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(item).strip().lower() for item in modalities))
    unknown = set(requested) - set(MODALITY_NAMES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one modality must be enabled")
    return tuple(name for name in MODALITY_NAMES if name in requested)


def masked_smooth_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    beta: float,
    resource_weights: Sequence[float] = (1.0, 1.0),
) -> torch.Tensor:
    if predictions.shape != targets.shape or predictions.shape != target_mask.shape:
        raise ValueError("predictions, targets, and target_mask must have identical shapes")
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("Expected [batch, 2] memory/VRAM tensors")
    if beta <= 0:
        raise ValueError("Smooth-L1 beta must be positive and explicit")
    if len(resource_weights) != 2 or any(float(weight) <= 0 for weight in resource_weights):
        raise ValueError("resource_weights must contain two positive values")

    elementwise = F.smooth_l1_loss(
        predictions,
        targets.to(predictions.dtype),
        reduction="none",
        beta=float(beta),
    )
    mask = target_mask.to(elementwise.dtype)
    counts = mask.sum(dim=0)
    per_resource = (elementwise * mask).sum(dim=0) / counts.clamp_min(1.0)
    weights = predictions.new_tensor(resource_weights)
    active = (counts > 0).to(weights.dtype)
    active_weight = weights * active
    return (per_resource * active_weight).sum() / active_weight.sum().clamp_min(1.0)


class GatedQwenRegressor(nn.Module):
    """Script-only Qwen tower plus independent user and metadata towers."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        num_users: int,
        metadata_dim: int,
        fusion_dim: int = 256,
        user_embedding_dim: int = 64,
        head_hidden_dim: int = 128,
        dropout: float = 0.3,
        enabled_modalities: Sequence[str] = MODALITY_NAMES,
        fusion_mode: str = "gated",
        torch_dtype: torch.dtype = torch.float32,
        attention_implementation: str | None = None,
        trust_remote_code: bool = True,
        gradient_checkpointing: bool = False,
        lora_config: Mapping[str, Any] | None = None,
        adapter_path: str | Path | None = None,
        adapter_trainable: bool = True,
    ) -> None:
        super().__init__()
        if num_users <= 0 or metadata_dim <= 0 or fusion_dim <= 0 or head_hidden_dim <= 0:
            raise ValueError(
                "num_users, metadata_dim, fusion_dim, and head_hidden_dim must be positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.enabled_modalities = normalize_modalities(enabled_modalities)
        if fusion_mode not in {"gated", "concat"}:
            raise ValueError("fusion_mode must be gated or concat")
        if fusion_mode == "concat" and self.enabled_modalities != MODALITY_NAMES:
            raise ValueError("concat fusion currently requires all three modalities")
        self.fusion_mode = fusion_mode
        self.model_name_or_path = str(model_name_or_path)
        self.num_users = int(num_users)
        self.metadata_dim = int(metadata_dim)
        self.fusion_dim = int(fusion_dim)
        self.user_embedding_dim = int(user_embedding_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.dropout_probability = float(dropout)
        self.attention_implementation = attention_implementation
        self.trust_remote_code = bool(trust_remote_code)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.lora_config = dict(lora_config or {})
        self.lora_enabled = bool(self.lora_config.get("enabled", False))
        self.code_encoder_loaded = "code" in self.enabled_modalities

        if self.code_encoder_loaded:
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": torch_dtype,
            }
            if attention_implementation:
                load_kwargs["attn_implementation"] = attention_implementation
            encoder = AutoModel.from_pretrained(self.model_name_or_path, **load_kwargs)
            encoder.config.use_cache = False

            if self.lora_enabled:
                try:
                    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
                except ImportError as exc:
                    raise RuntimeError("LoRA is enabled but peft is not installed") from exc

                if adapter_path is not None:
                    encoder = PeftModel.from_pretrained(
                        encoder,
                        str(adapter_path),
                        is_trainable=bool(adapter_trainable),
                    )
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        r=int(self.lora_config.get("r", 16)),
                        lora_alpha=int(self.lora_config.get("alpha", 32)),
                        lora_dropout=float(self.lora_config.get("dropout", 0.10)),
                        target_modules=list(
                            self.lora_config.get(
                                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                            )
                        ),
                        bias="none",
                    )
                    encoder = get_peft_model(encoder, peft_config)

            if self.gradient_checkpointing:
                encoder.gradient_checkpointing_enable()
                if hasattr(encoder, "enable_input_require_grads"):
                    encoder.enable_input_require_grads()

            self.code_encoder = encoder
            code_dim = int(self.code_encoder.config.hidden_size)
            self.code_projection = nn.Linear(code_dim, self.fusion_dim)
        else:
            # A strict no-code ablation must not allocate an unused code backbone.
            self.code_encoder = nn.Identity()
            self.code_projection = nn.Identity()

        self.user_embedding = nn.Embedding(self.num_users, self.user_embedding_dim)
        self.user_projection = nn.Linear(self.user_embedding_dim, self.fusion_dim)

        self.metadata_projection = nn.Linear(self.metadata_dim, self.fusion_dim)

        # A shared scorer matches e_jk = a^T h_jk. A shared bias would cancel in softmax.
        self.gate_scorer = nn.Linear(self.fusion_dim, 1, bias=False)
        self.concat_projection = nn.Linear(self.fusion_dim * 3, self.fusion_dim)
        if self.fusion_mode == "gated":
            self.concat_projection.requires_grad_(False)
        else:
            self.gate_scorer.requires_grad_(False)
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_probability)
        head_input_dim = self.fusion_dim + self.metadata_dim
        self.memory_head = self._prediction_head(head_input_dim)
        self.vram_head = self._prediction_head(head_input_dim)

        self.register_buffer(
            "enabled_mask",
            torch.tensor(
                [name in self.enabled_modalities for name in MODALITY_NAMES],
                dtype=torch.bool,
            ),
            persistent=True,
        )
        self._freeze_disabled_towers()

    def _prediction_head(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.head_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _freeze_disabled_towers(self) -> None:
        towers = {
            "code": (self.code_encoder, self.code_projection),
            "user": (self.user_embedding, self.user_projection),
            "metadata": (self.metadata_projection,),
        }
        for name, modules in towers.items():
            if name in self.enabled_modalities:
                continue
            for module in modules:
                module.requires_grad_(False)

    @staticmethod
    def _last_non_padding(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        last_indices = (positions * attention_mask.to(positions.dtype)).max(dim=1).values
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_indices, last_indices]

    def _zero_branch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.fusion_dim,
            device=device,
            dtype=self.gate_scorer.weight.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_ids: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        code_hidden = self.encode_script(input_ids, attention_mask)
        return self.predict_from_code_hidden(code_hidden, user_ids, metadata)

    def encode_script(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the immutable-checkpoint script branch for exact reuse.

        Cache entries contain only this script-dependent representation. User
        and request representations and prediction heads still run per job.
        This method is inference-equivalent to the script branch in ``forward``.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        if "code" in self.enabled_modalities:
            encoded = self.code_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            code_vector = self._last_non_padding(encoded.last_hidden_state, attention_mask)
            code_vector = code_vector.to(self.code_projection.weight.dtype)
            code_hidden = torch.tanh(self.code_projection(code_vector))
        else:
            code_hidden = self._zero_branch(batch_size, device)
        return code_hidden

    def predict_from_code_hidden(
        self,
        code_hidden: torch.Tensor,
        user_ids: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Recompute user/request fusion and both heads for a cached script."""
        batch_size = user_ids.shape[0]
        device = user_ids.device
        if code_hidden.shape != (batch_size, self.fusion_dim):
            raise ValueError("Cached script branch has an incompatible shape")
        code_hidden = code_hidden.to(device=device)

        if "user" in self.enabled_modalities:
            user_vector = self.user_embedding(user_ids)
            user_hidden = torch.tanh(self.user_projection(user_vector))
        else:
            user_hidden = self._zero_branch(batch_size, device)

        if "metadata" in self.enabled_modalities:
            metadata = metadata.to(self.metadata_projection.weight.dtype)
            metadata_hidden = torch.tanh(self.metadata_projection(metadata))
            metadata_condition = metadata
        else:
            metadata_hidden = self._zero_branch(batch_size, device)
            metadata_condition = torch.zeros(
                batch_size,
                self.metadata_dim,
                device=device,
                dtype=metadata_hidden.dtype,
            )

        stacked = torch.stack((code_hidden, user_hidden, metadata_hidden), dim=1)
        if self.fusion_mode == "concat":
            fused = self.concat_projection(stacked.reshape(batch_size, -1))
            gate_weights = torch.zeros(
                batch_size, 3, device=device, dtype=fused.dtype
            )
        else:
            gate_logits = self.gate_scorer(stacked).squeeze(-1)
            active = self.enabled_mask.to(device=gate_logits.device).unsqueeze(0)
            gate_logits = gate_logits.masked_fill(~active, torch.finfo(gate_logits.dtype).min)
            gate_weights = torch.softmax(gate_logits, dim=1)
            gate_weights = gate_weights * active.to(gate_weights.dtype)
            gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            fused = torch.sum(stacked * gate_weights.unsqueeze(-1), dim=1)
        fused = self.fusion_dropout(F.gelu(self.fusion_norm(fused)))
        final_features = torch.cat(
            (fused, metadata_condition.to(fused.dtype)),
            dim=1,
        )
        predictions = torch.cat(
            (self.memory_head(final_features), self.vram_head(final_features)),
            dim=1,
        )
        return {"predictions": predictions, "gate_weights": gate_weights}

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "num_users": self.num_users,
            "metadata_dim": self.metadata_dim,
            "fusion_dim": self.fusion_dim,
            "user_embedding_dim": self.user_embedding_dim,
            "head_hidden_dim": self.head_hidden_dim,
            "dropout": self.dropout_probability,
            "enabled_modalities": list(self.enabled_modalities),
            "fusion_mode": self.fusion_mode,
            "metadata_residual_conditioning": True,
            "attention_implementation": self.attention_implementation,
            "trust_remote_code": self.trust_remote_code,
            "gradient_checkpointing": self.gradient_checkpointing,
            "code_encoder_loaded": self.code_encoder_loaded,
            "lora": self.lora_config,
        }

    def save_checkpoint(self, directory: str | Path, tokenizer: Any | None = None) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        if self.code_encoder_loaded:
            encoder_subdir = (
                "code_encoder_adapter" if self.lora_enabled else "code_encoder_backbone"
            )
            self.code_encoder.save_pretrained(
                output / encoder_subdir, safe_serialization=True
            )

        non_encoder = {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("code_encoder.")
        }
        torch.save(non_encoder, output / "fusion_heads.pt")
        with (output / "model_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.checkpoint_metadata(), handle, indent=2, ensure_ascii=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output / "tokenizer")

    @classmethod
    def from_checkpoint(
        cls,
        directory: str | Path,
        *,
        torch_dtype: torch.dtype = torch.float32,
        model_name_override: str | None = None,
        attention_implementation: str | None = None,
    ) -> "GatedQwenRegressor":
        source = Path(directory)
        with (source / "model_config.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        lora_config = dict(metadata.get("lora") or {})
        lora_enabled = bool(lora_config.get("enabled", False))
        code_enabled = "code" in normalize_modalities(metadata["enabled_modalities"])
        adapter_path: Path | None
        if not code_enabled:
            model_path = model_name_override or metadata["model_name_or_path"]
            adapter_path = None
        elif lora_enabled:
            model_path = model_name_override or metadata["model_name_or_path"]
            adapter_path = source / "code_encoder_adapter"
        else:
            model_path = str(source / "code_encoder_backbone")
            adapter_path = None

        model = cls(
            model_name_or_path=model_path,
            num_users=int(metadata["num_users"]),
            metadata_dim=int(metadata["metadata_dim"]),
            fusion_dim=int(metadata["fusion_dim"]),
            user_embedding_dim=int(metadata["user_embedding_dim"]),
            head_hidden_dim=int(metadata["head_hidden_dim"]),
            dropout=float(metadata["dropout"]),
            enabled_modalities=metadata["enabled_modalities"],
            fusion_mode=str(metadata.get("fusion_mode", "gated")),
            torch_dtype=torch_dtype,
            attention_implementation=(
                attention_implementation
                if attention_implementation is not None
                else metadata.get("attention_implementation")
            ),
            trust_remote_code=bool(metadata.get("trust_remote_code", True)),
            gradient_checkpointing=False,
            lora_config=lora_config,
            adapter_path=adapter_path,
            adapter_trainable=False,
        )

        state = _load_torch_state(source / "fusion_heads.pt")
        if not code_enabled:
            # Old no-code checkpoints may contain an unused frozen projection.
            state = {
                name: value
                for name, value in state.items()
                if not name.startswith("code_projection.")
            }
        result = model.load_state_dict(state, strict=False)
        bad_missing = [key for key in result.missing_keys if not key.startswith("code_encoder.")]
        if bad_missing or result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible: "
                f"missing={bad_missing}, unexpected={result.unexpected_keys}"
            )
        return model
