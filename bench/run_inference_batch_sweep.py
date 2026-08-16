"""Batch-size vs latency sweep for the inference stage (Step 9's last item).

For each INFERENCE_BATCH_SIZE in [1, 8, 16, 32]: restart the inference
container with that env var, upload one large image (144 tiles, enough to
actually fill an 8/16/32 batch), poll /detect/status until complete, record
wall-clock throughput and p50/p95/p99 from tile_latencies_s.

Runs on the HOST like bench/run_bench.py - needs the docker CLI.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_URL = "http://localhost:5000"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

BATCH_SIZES = [1, 8, 16, 32]
IMAGE_SIZE = 6144  # 144 tiles at TILE_SIZE=512


def gen_image(seed=7):
    rng = np.random.default_rng(seed)
    xv, yv = np.meshgrid(np.linspace(0, 255, IMAGE_SIZE), np.linspace(0, 255, IMAGE_SIZE))
    image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.uint8)
    for _ in range(15):
        c = (int(rng.integers(0, IMAGE_SIZE)), int(rng.integers(0, IMAGE_SIZE)))
        r = int(rng.integers(40, 300))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.circle(image, c, r, color, thickness=-1)
    return image


def restart_inference(batch_size):
    print(f"  restarting inference with INFERENCE_BATCH_SIZE={batch_size} ...")
    subprocess.run(
        ["docker", "compose", "stop", "inference"],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    subprocess.run(
        ["docker", "compose", "run", "-d", "--rm", "--name", "dip-inference-bench",
         "-e", f"INFERENCE_BATCH_SIZE={batch_size}", "inference"],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    time.sleep(5)


def stop_bench_container():
    subprocess.run(["docker", "rm", "-f", "dip-inference-bench"],
                    capture_output=True)


def upload_and_wait(image, timeout=120):
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    t0 = time.time()
    resp = requests.post(
        f"{MASTER_URL}/detect",
        files={'image': ('bench.jpg', buf.tobytes(), 'image/jpeg')},
        timeout=30,
    )
    resp.raise_for_status()
    job_id = resp.json()['job_id']
    tiles_count = resp.json()['tiles_count']

    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = requests.get(f"{MASTER_URL}/detect/status/{job_id}", timeout=10).json()
        if status.get('status') in ('completed', 'degraded'):
            break
        time.sleep(0.3)
    else:
        raise TimeoutError(f"job {job_id} did not finish within {timeout}s")

    wall_clock_s = time.time() - t0
    latencies_ms = [t * 1000 for t in status.get('tile_latencies_s', [])]
    return wall_clock_s, latencies_ms, tiles_count, status.get('status')


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def run():
    try:
        requests.get(f"{MASTER_URL}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        print("ERROR: master not reachable. Run `docker compose up -d master` first.", file=sys.stderr)
        sys.exit(1)

    image = gen_image()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for batch_size in BATCH_SIZES:
        stop_bench_container()
        restart_inference(batch_size)

        print(f"batch_size={batch_size:>2}  ({IMAGE_SIZE}x{IMAGE_SIZE}, 144 tiles) ...", end=" ", flush=True)
        try:
            wall_clock_s, latencies_ms, tiles_count, final_status = upload_and_wait(image)
        except (requests.RequestException, TimeoutError) as e:
            print(f"FAILED: {e}")
            continue

        tiles_per_sec = tiles_count / wall_clock_s if wall_clock_s > 0 else 0
        row = {
            'batch_size': batch_size,
            'tile_count': tiles_count,
            'status': final_status,
            'wall_clock_s': round(wall_clock_s, 3),
            'tiles_per_sec': round(tiles_per_sec, 2),
            'p50_ms': round(percentile(latencies_ms, 50), 2),
            'p95_ms': round(percentile(latencies_ms, 95), 2),
            'p99_ms': round(percentile(latencies_ms, 99), 2),
        }
        rows.append(row)
        print(f"{wall_clock_s:.2f}s  {tiles_per_sec:.1f} tiles/s  p50={row['p50_ms']}ms  p99={row['p99_ms']}ms  [{final_status}]")

    stop_bench_container()
    print("Restoring normal inference service (default batch size)...")
    subprocess.run(["docker", "compose", "up", "-d", "inference"],
                    cwd=REPO_ROOT, check=True, capture_output=True)

    out_path = RESULTS_DIR / 'inference_batch_sweep.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    run()
