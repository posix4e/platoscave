"""Focused software tests; all tensors and state dictionaries are artificial."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

import vision_features as vision


class ArtificialEncoder(torch.nn.Module):
    def __init__(self, zero=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.feature_metadata = {"dimension": 2}
        self.zero = zero
        self.eval()

    def forward(self, x):
        output = torch.stack((x[:, 0].mean((1, 2)), x[:, 1].mean((1, 2))), dim=1)
        return output * (0 if self.zero else self.weight)


class VisionFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vision.configure_runtime()

    def setUp(self):
        self.model = ArtificialEncoder()
        self.inputs = torch.ones(3, 3, 224, 224, dtype=torch.float32)

    def test_extract_preserves_rows_and_normalizes(self):
        self.inputs[1, 0] = 3
        result = vision.extract(self.model, self.inputs, batch_size=2)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
        self.assertGreater(result[1, 0], result[0, 0])

    def test_zero_representation_rejected(self):
        with self.assertRaisesRegex(ValueError, "Zero or nonfinite"):
            vision.extract(ArtificialEncoder(zero=True), self.inputs)

    def test_nonfinite_input_rejected(self):
        self.inputs[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite input"):
            vision.extract(self.model, self.inputs)

    def test_shape_precision_and_batch_size_rejected(self):
        for value, batch in [(self.inputs.half(), 5), (self.inputs[:, :, :223], 5),
                             (self.inputs[:0], 5), (self.inputs, 0), (self.inputs, True)]:
            with self.subTest(shape=value.shape, batch=batch), self.assertRaises(ValueError):
                vision.extract(self.model, value, batch)

    def test_training_mode_rejected(self):
        self.model.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            vision.extract(self.model, self.inputs)

    def test_trainable_or_nonfp32_parameters_rejected(self):
        self.model.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen FP32"):
            vision.extract(self.model, self.inputs)
        self.model.requires_grad_(False).double()
        with self.assertRaisesRegex(ValueError, "frozen FP32"):
            vision.extract(self.model, self.inputs)

    def test_autocast_rejected(self):
        with torch.autocast("cpu"), self.assertRaisesRegex(ValueError, "Autocast"):
            vision.extract(self.model, self.inputs)

    def test_tf32_setting_change_rejected(self):
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with self.assertRaisesRegex(ValueError, "runtime settings changed"):
                vision.extract(self.model, self.inputs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def test_projection_removal_is_exact(self):
        state = {f"module.encoder_q.synthetic_{i}": i for i in range(318)}
        state.update({key: 0 for key in vision.MOCO_REMOVED_KEYS})
        result = vision.moco_backbone_state(state)
        self.assertEqual(set(result), {f"synthetic_{i}" for i in range(318)})
        state["module.encoder_k.extra"] = state.pop("module.encoder_q.synthetic_0")
        with self.assertRaisesRegex(ValueError, "Unexpected MoCo"):
            vision.moco_backbone_state(state)

    def test_missing_moco_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "322 original"):
            vision.moco_backbone_state({})

    def test_checkpoint_hash_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pth"
            path.write_bytes(b"software-test-only")
            with mock.patch.object(vision, "configure_runtime") as configure:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    vision.load_moco(path)
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    vision.load_mae(path, folder)
                configure.assert_not_called()

    def test_vendor_hash_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mae"
            path.mkdir()
            (path / "models_vit.py").write_text("raise RuntimeError('must never execute')\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                vision._mae_module(folder)


if __name__ == "__main__":
    unittest.main()
