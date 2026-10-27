from pathlib import Path

import torch

import qwen.model as model_module
from qwen.model import GatedQwenRegressor


def _no_code_model() -> GatedQwenRegressor:
    return GatedQwenRegressor(
        model_name_or_path="must-not-be-loaded",
        num_users=3,
        metadata_dim=4,
        fusion_dim=8,
        user_embedding_dim=4,
        head_hidden_dim=6,
        enabled_modalities=("user", "metadata"),
        torch_dtype=torch.float32,
        lora_config={"enabled": True},
    )


def test_no_code_ablation_never_loads_or_saves_encoder(tmp_path: Path, monkeypatch):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("no-code ablation attempted to load a code encoder")

    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", fail_if_loaded)
    model = _no_code_model()

    assert model.code_encoder_loaded is False
    assert isinstance(model.code_encoder, torch.nn.Identity)
    assert isinstance(model.code_projection, torch.nn.Identity)
    model.save_checkpoint(tmp_path)
    assert not (tmp_path / "code_encoder_adapter").exists()
    assert not (tmp_path / "code_encoder_backbone").exists()

    restored = GatedQwenRegressor.from_checkpoint(tmp_path)
    assert restored.code_encoder_loaded is False
    assert restored.enabled_modalities == ("user", "metadata")
