"""Sweeps TILE_SIZE, measures throughput - the missing data behind
app/config.py's TILE_SIZE default (previously just "512, tuned by
inspection", now a measured result).

Restarts `master` with each TILE_SIZE (master is the only process that
tiles/computes geometry - workers process whatever tile they're handed,
size-agnostic), holds worker count fixed so tile size is the only
variable, and re-uses bench/run_bench.py's own image generator so results
are directly comparable to the worker-count sweeps.

Runs on the HOST like the other bench scripts - needs the docker CLI.

Usage:
    docker compose up -d --scale worker=4   # fixed worker count for the sweep
    python -m bench.run_tile_size_sweep --output bench/results/tile_size_sweep.csv
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.run_bench import gen_test_image, IMAGE_SPECS, DEFAULT_OPERATION, percentile  # noqa: E402

MASTER_URL = "http://localhost:5000"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

# 128 would push the large image (6144x6144) to 2304 tiles, over MAX_TILES
# (1000) - excluded rather than raising the cap just for this sweep.
TILE_SIZES = [256, 384, 512, 768, 1024]

UPLOAD_TIMEOUT_S = 30
JOB_TIMEOUT_S = 120
MASTER_READY_TIMEOUT_S = 60


def restart_master(tile_size):
    print(f"  restarting master with TILE_SIZE={tile_size} ...")
    env = {'TILE_SIZE': str(tile_size)}
    subprocess.run(
        ["docker", "compose", "up", "-d", "master"],
        cwd=REPO_ROOT, check=True, capture_output=True,
        env={**__import__('os').environ, **env},
    )
    deadline = time.time() + MASTER_READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            # A fresh master's heartbeat_tracker is in-process memory (see
            # docs/ARCHITECTURE.md's "master can't be horizontally scaled"
            # section) - /health returns 200 immediately on restart, but
            # /upload still 503s ("No active workers") until real heartbeats
            # repopulate it, a few seconds later. Wait for that, not just
            # for the port to answer.
            resp = requests.get(f"{MASTER_URL}/health", timeout=5)
            if resp.ok and resp.json().get('active_workers', 0) > 0:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def upload_and_wait(image, operation, timeout=JOB_TIMEOUT_S):
    import cv2
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    t0 = time.time()
    resp = requests.post(
        f"{MASTER_URL}/upload",
        files={'image': ('bench.jpg', buf.tobytes(), 'image/jpeg')},
        data={'operation': operation},
        timeout=UPLOAD_TIMEOUT_S,
    )
    resp.raise_for_status()
    job_id = resp.json()['job_id']
    tiles_count = resp.json()['tiles_count']

    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = requests.get(f"{MASTER_URL}/status/{job_id}", timeout=10).json()
        if status.get('status') in ('ready_for_reconstruction', 'completed'):
            break
        time.sleep(0.3)
    else:
        raise TimeoutError(f"job {job_id} did not complete within {timeout}s")

    wall_clock_s = time.time() - t0
    tile_latencies_ms = [t * 1000 for t in status.get('tile_latencies_s', [])]

    # See bench/run_bench.py's identical call for why: /reconstruct is the
    # only thing that deletes this job's MinIO blobs on the OpenCV path.
    try:
        requests.post(f"{MASTER_URL}/reconstruct/{job_id}", timeout=30).raise_for_status()
    except requests.RequestException as e:
        print(f"  warning: reconstruct cleanup failed for job {job_id}: {e}")

    return wall_clock_s, tile_latencies_ms, tiles_count


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', default=DEFAULT_OPERATION)
    parser.add_argument('--output', default=str(RESULTS_DIR / 'tile_size_sweep.csv'))
    args = parser.parse_args()

    try:
        requests.get(f"{MASTER_URL}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        print("ERROR: master not reachable. Run `docker compose up -d --scale worker=4` first.",
              file=sys.stderr)
        sys.exit(1)

    print("Generating test images (deterministic seeds, not written to disk)...")
    images = {name: gen_test_image(w, h, seed=42) for name, w, h in IMAGE_SPECS}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for tile_size in TILE_SIZES:
        if not restart_master(tile_size):
            print(f"  WARNING: master not healthy after restart for TILE_SIZE={tile_size}, skipping", file=sys.stderr)
            continue

        for name, width, height in IMAGE_SPECS:
            image = images[name]
            print(f"tile_size={tile_size:>4}  image={name:<6} ({width}x{height})  op={args.operation} ...",
                  end=" ", flush=True)
            try:
                wall_clock_s, tile_latencies_ms, tiles_count = upload_and_wait(image, args.operation)
            except (requests.RequestException, TimeoutError) as e:
                print(f"FAILED: {e}")
                continue

            tiles_per_sec = tiles_count / wall_clock_s if wall_clock_s > 0 else 0
            row = {
                'tile_size': tile_size,
                'image_name': name,
                'width': width,
                'height': height,
                'tile_count': tiles_count,
                'wall_clock_s': round(wall_clock_s, 3),
                'tiles_per_sec': round(tiles_per_sec, 2),
                'p50_ms': round(percentile(tile_latencies_ms, 50), 2),
                'p99_ms': round(percentile(tile_latencies_ms, 99), 2),
            }
            rows.append(row)
            print(f"{tiles_count} tiles  {wall_clock_s:.2f}s  {tiles_per_sec:.1f} tiles/s  p99={row['p99_ms']}ms")

    print("Restoring default TILE_SIZE...")
    restart_master(256)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == '__main__':
    run()
