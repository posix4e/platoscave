#!/usr/bin/env python3
"""Check DINO image preprocessing against an independently expressed reference.

Use torchvision's public transforms when installed; otherwise use the public
resize/center-crop semantics with tensor-based ToTensor/Normalize arithmetic.
The fallback does not import or call any extractor preprocessing helper.
Both paths test RGB, grayscale, and RGBA inputs at even/odd, portrait/landscape,
already-resized, and small image sizes. No model or benchmark score is used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import PIL
import torch


def reference(image_path):
    # These are the documented Resize(int)/CenterCrop/ToTensor/Normalize
    # semantics. Compute normalization through torch, independently of the
    # extractor's numpy float32 operations, so arithmetic drift is visible.
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        short_side = min(image.size)
        long_side = max(image.size)
        new_long = int(long_side * 256 / short_side)
        size = (256, new_long) if image.width <= image.height else (new_long, 256)
        if image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        # Derive the crop origin from the image center rather than reuse the
        # extraction function's crop expression.
        x, y = round(image.width / 2 - 112), round(image.height / 2 - 112)
        pixels = np.array(image.crop((x, y, x + 224, y + 224)), dtype=np.uint8)
    value = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div(255)
    means = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    stds = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return value.sub(means).div(stds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", type=Path, default=Path(__file__).with_name("extract_features.py"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-torchvision", action="store_true")
    parser.add_argument("--real-image", type=Path, action="append", default=[])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("epic_extractor_checked", args.extractor)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    torchvision_transform = None
    torchvision_status = None
    try:
        import torchvision
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
        torchvision_transform = Compose([
            Resize(256, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
            ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        torchvision_status = {"available": True, "version": torchvision.__version__}
    except (ImportError, RuntimeError) as error:
        torchvision_status = {"available": False, "reason": str(error)}
    if args.require_torchvision and torchvision_transform is None:
        raise RuntimeError(f"Official torchvision reference is required: {torchvision_status}")
    rng = np.random.default_rng(918273)
    sizes = [(224, 224), (256, 256), (640, 480), (480, 640), (799, 601),
             (601, 799), (179, 331), (31, 63), (513, 256), (256, 513)]
    checks = []
    with tempfile.TemporaryDirectory(prefix="epic-preprocess-check-") as directory:
        path = Path(directory) / "image.png"
        for width, height in sizes:
            for mode, channels in (("RGB", 3), ("RGBA", 4), ("L", None)):
                shape = (height, width, channels) if channels else (height, width)
                pixels = rng.integers(0, 256, size=shape, dtype=np.uint8)
                Image.fromarray(pixels, mode=mode).save(path)
                actual = extractor.preprocess(path, np, torch, Image)
                expected = reference(path)
                error = float((actual - expected).abs().max())
                # 1e-6 is below one RGB quantization step after normalization.
                assert actual.shape == (3, 224, 224)
                assert actual.dtype == torch.float32
                assert torch.isfinite(actual).all()
                assert torch.allclose(actual, expected, atol=1e-6, rtol=0), (mode, width, height, error)
                result = {"size": [width, height], "mode": mode, "independent_reference_max_abs_error": error}
                if torchvision_transform is not None:
                    with Image.open(path) as image:
                        torchvision_expected = torchvision_transform(image.convert("RGB"))
                    tv_error = float((actual - torchvision_expected).abs().max())
                    assert torch.allclose(actual, torchvision_expected, atol=1e-6, rtol=0), (mode, width, height, tv_error)
                    result["torchvision_max_abs_error"] = tv_error
                checks.append(result)
    for path in args.real_image:
        actual = extractor.preprocess(path, np, torch, Image)
        expected = reference(path)
        error = float((actual - expected).abs().max())
        assert torch.allclose(actual, expected, atol=1e-6, rtol=0), (str(path), error)
        result = {"real_image": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "independent_reference_max_abs_error": error}
        if torchvision_transform is not None:
            with Image.open(path) as image:
                torchvision_expected = torchvision_transform(image.convert("RGB"))
            tv_error = float((actual - torchvision_expected).abs().max())
            assert torch.allclose(actual, torchvision_expected, atol=1e-6, rtol=0), (str(path), tv_error)
            result["torchvision_max_abs_error"] = tv_error
        checks.append(result)
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "case_count": len(checks), "checks": checks,
        "extractor_sha256": hashlib.sha256(args.extractor.read_bytes()).hexdigest(),
        "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__},
        "torchvision": torchvision_status,
        "scope": "Preprocessing transform parity only; not model correctness or experiment validity.",
        "reference_sources": [
            "https://github.com/pytorch/vision/blob/v0.22.1/torchvision/transforms/functional.py",
            "https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/eval_linear.py",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
