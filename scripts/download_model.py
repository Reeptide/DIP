#!/usr/bin/env python3
"""Downloads the ONNX model weights the inference stage needs, into models/.

Model weights aren't committed to git (see .gitignore) - real binary
weights don't belong in version control any more than the uploads/results
runtime images did. Run this once before starting the inference service:

    python scripts/download_model.py

Idempotent: skips any file whose SHA-256 already matches the pinned digest
below - not just "exists and is non-empty," which is all the check used to
do (the docstring claimed "at the expected size" when no expected size was
ever actually defined or checked anywhere - found during a full-project
review, along with the two problems the fix below addresses).
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# Pinned to a specific release / commit, not a moving ref, so a fresh clone
# gets byte-identical weights to what this was developed/benchmarked
# against - verified by sha256, not just assumed. yolov8n's URL was already
# tag-pinned (v8.4.0); mobilenetv2's was NOT - it pointed at `.../raw/main/...`,
# a moving branch ref that could silently start serving different bytes.
# Repinned to the specific commit that path resolved to at review time
# (found via `GET /repos/onnx/models/commits?path=...`), with both
# digests confirmed by actually re-downloading and hashing each file.
MODELS = [
    {
        'name': 'yolov8n.onnx',
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx',
        'purpose': 'object detection (COCO 80 classes)',
        'sha256': 'b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd',
    },
    {
        'name': 'mobilenetv2-12.onnx',
        'url': 'https://github.com/onnx/models/raw/4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177/'
               'validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        'purpose': 'classification (ImageNet 1000 classes)',
        'sha256': 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5',
    },
]


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    sys.stdout.write(f"\r  {pct:3d}%  ({downloaded // 1024} KB)")
    sys.stdout.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(name, url, purpose, sha256):
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        if _sha256(dest) == sha256:
            print(f"{name}: already present and verified ({dest.stat().st_size // 1024} KB), skipping")
            return
        print(f"{name}: present but checksum mismatch - re-downloading")

    print(f"{name} - {purpose}")
    print(f"  from: {url}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        actual = _sha256(tmp)
        if actual != sha256:
            raise ValueError(
                f"{name}: SHA-256 mismatch after download - expected {sha256}, got {actual}. "
                f"The source may have changed; do not use this file without investigating."
            )
        tmp.rename(dest)
        print(f"\n  done: {dest} ({dest.stat().st_size // 1024} KB, checksum verified)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main():
    for model in MODELS:
        download_model(**model)
    print("\nAll models ready.")


if __name__ == '__main__':
    main()
