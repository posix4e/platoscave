# Vision encoder readiness

Both original checkpoints loaded and passed the synthetic checks in
`vision-readiness.json` on an H100 NVL. The fixtures are five deterministic RGB
patterns generated in memory. They are software tests, not experimental samples.
No human photographs, captions, correspondence search, or scores enter this check.

`vision_features.py` exposes:

- `load_moco(checkpoint_path, device="cuda")`: frozen MoCo-v2 ResNet-50, 800 epochs;
  2048-dimensional pooled backbone before the projection head. All 318 backbone
  state entries must match exactly; precisely four projection entries are removed.
- `load_mae(checkpoint_path, vendor_dir, device="cuda")`: frozen original MAE
  ViT-B/16; 768-dimensional CLS after the final encoder LayerNorm. All 150 encoder
  entries load strictly. All 196 patches retain raster order without masking.
- `extract(model, tensors, batch_size=5)`: accepts preprocessed FP32
  `[N,3,224,224]` tensors, returns finite L2-normalized FP32 NumPy rows in order.
  `model.feature_metadata` contains the verified provenance.

Both use the unchanged, hash-checked image helper from `fresh-pilot/extract.py`:
EXIF orientation, RGB, bicubic resize 256, center crop 224, ImageNet normalization.
The common bicubic transform differs from the bilinear resize in MoCo's original
linear-evaluation script. It is fixed for this panel before scoring.

MAE runs the unmodified [publisher's models_vit.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_vit.py)
with `global_pool=False`, using the vendored original `timm==0.3.2` wheel.
The only compatibility adjustment is a temporary import-time alias from
`torch._six.container_abcs` to `collections.abc`; no model operation changes.
The wheel is imported directly, with no package installation. Vendor source,
license, and wheel hashes are in `vision-vendor-lock.json`. MoCo uses the installed
torchvision ResNet implementation, whose version and source hash are checked.

The runtime is Torch 2.7.1/cu128, torchvision 0.22.1/cu128, NumPy 2.5.3, Pillow
12.3.0, deterministic FP32, TF32 disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
All 12 focused software tests passed in that VM runtime.

Repeated and reversed-order runs were bitwise identical for both models.
The largest normalized-coordinate change between batches of five and batches
of one or two was `8.940696716308594e-08` for MoCo and
`4.470348358154297e-07` for MAE, below the prespecified absolute tolerance `2e-6`.
Encoder buffers and both CPU and CUDA random-generator states stayed unchanged.
The report records hashes of the code, generated fixtures, and resulting vectors.
