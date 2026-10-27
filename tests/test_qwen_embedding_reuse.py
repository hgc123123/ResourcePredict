"""Numerical equivalence of direct and cached-script inference, no model download."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from qwen.model import GatedQwenRegressor
except ImportError:
    torch = None
    GatedQwenRegressor = None


@unittest.skipIf(torch is None, "Requires the Qwen PyTorch/transformers environment")
class EmbeddingReuseTests(unittest.TestCase):
    def _model(self, mode="gated"):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = torch.nn.Embedding(20, 8)

            def forward(self, input_ids, attention_mask, return_dict):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        with patch("qwen.model.AutoModel.from_pretrained", return_value=TinyEncoder()):
            return GatedQwenRegressor(
                model_name_or_path="mock", num_users=4, metadata_dim=3,
                fusion_dim=6, user_embedding_dim=4, head_hidden_dim=5,
                dropout=0, fusion_mode=mode,
            ).eval()

    def test_gated_and_concat_reuse_recompute_user_and_metadata(self):
        torch.manual_seed(7)
        for mode in ("gated", "concat"):
            model = self._model(mode)
            ids = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 0]])
            mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
            users = torch.tensor([1, 2])
            metadata = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
            with torch.inference_mode():
                direct = model(ids, mask, users, metadata)
                branch = model.encode_script(ids[:1], mask[:1])
                cached = model.predict_from_code_hidden(branch.repeat(2, 1), users, metadata)
            torch.testing.assert_close(direct["predictions"], cached["predictions"], rtol=0, atol=0)
            torch.testing.assert_close(direct["gate_weights"], cached["gate_weights"], rtol=0, atol=0)
            self.assertFalse(torch.equal(cached["predictions"][0], cached["predictions"][1]))
            with self.assertRaises(ValueError):
                model.predict_from_code_hidden(branch, users, metadata)

    def test_no_code_preserves_forward_without_encoder(self):
        model = GatedQwenRegressor(model_name_or_path="not_loaded", num_users=3,
                                  metadata_dim=2, fusion_dim=4, user_embedding_dim=3,
                                  head_hidden_dim=4, dropout=0,
                                  enabled_modalities=("user", "metadata")).eval()
        ids = torch.zeros((1, 1), dtype=torch.long)
        mask = torch.ones_like(ids)
        users = torch.tensor([1])
        metadata = torch.tensor([[1., 2.]])
        with torch.inference_mode():
            direct = model(ids, mask, users, metadata)
            branch = model.encode_script(ids, mask)
            cached = model.predict_from_code_hidden(branch, users, metadata)
        torch.testing.assert_close(direct["predictions"], cached["predictions"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
