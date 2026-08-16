#!/usr/bin/env python3
"""Downloads the ONNX model weights the inference stage needs, into models/.

Model weights aren't committed to git (see .gitignore) - real binary
weights don't belong in version control any more than the uploads/results
runtime images did. Run this once before starting the inference service:

    python scripts/download_model.py

Idempotent: skips any file that already exists at the expected size.
"""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# Pinned to a specific release, not a moving "latest" tag, so a fresh clone
# gets byte-identical weights to what this was developed/benchmarked against.
MODELS = [
    {
        'name': 'yolov8n.onnx',
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx',
        'purpose': 'object detection (COCO 80 classes)',
    },
    {
        'name': 'mobilenetv2-12.onnx',
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        'purpose': 'classification (ImageNet 1000 classes)',
    },
]


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    sys.stdout.write(f"\r  {pct:3d}%  ({downloaded // 1024} KB)")
    sys.stdout.flush()


def download_model(name, url, purpose):
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"{name}: already present ({dest.stat().st_size // 1024} KB), skipping")
        return

    print(f"{name} - {purpose}")
    print(f"  from: {url}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        tmp.rename(dest)
        print(f"\n  done: {dest} ({dest.stat().st_size // 1024} KB)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main():
    for model in MODELS:
        download_model(**model)
    print("\nAll models ready.")


if __name__ == '__main__':
    main()
