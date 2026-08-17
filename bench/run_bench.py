"""Benchmark harness: sweeps worker count, measures throughput and tile
latency, writes a CSV under bench/results/.

Runs on the HOST, not in a container - it drives `docker compose up --scale
worker=N` itself, so it needs the docker CLI.

Usage:
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    docker compose up -d master          # master must already be running
    python -m bench.run_bench --output bench/results/baseline.csv     # before Step 5
    python -m bench.run_bench --output bench/results/claimcheck.csv   # after Step 5

avg_tile_kb/total_kb_in measure different things depending on when you run
this, which is the whole point of the comparison: before Step 5, the task
message itself carried the base64 tile (~700KB); after Step 5 it carries a
MinIO blob_key (~200 bytes) and the image bytes go over HTTP to MinIO
instead. kafka_message_bytes() below always measures what actually would
travel through Kafka for one task, whichever era's message format that is.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import TILE_SIZE  # noqa: E402
from app.imaging import split_image_into_tiles  # noqa: E402

MASTER_URL = "http://localhost:5000"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

WORKER_COUNTS = [1, 2, 4, 6, 8, 12]
IMAGE_SPECS = [
    ("small", 2048, 2048),
    ("medium", 4096, 4096),
    ("large", 6144, 6144),
]
DEFAULT_OPERATION = "edge_canny"

UPLOAD_TIMEOUT_S = 30
JOB_TIMEOUT_S = 120
WORKER_REGISTRATION_TIMEOUT_S = 90


def gen_test_image(width, height, seed):
    """Deterministic synthetic image: gradient + shapes + light noise.
    Not stored on disk or in git - generated fresh every run so the repo
    never repeats the uploads/results bloat mistake from the original code.
    """
    rng = np.random.default_rng(seed)

    xv, yv = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
    image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.uint8)

    for _ in range(20):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(20, min(width, height) // 6))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(image, center, radius, color, thickness=-1)

    noise = rng.integers(0, 30, size=image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    return image


def kafka_task_message_bytes(image, job_id):
    """Size of the actual Kafka task message master/publisher.py would
    produce for each tile of this image - not the tile's image size. Since
    Step 5, image bytes never enter the message at all (they go to MinIO
    over HTTP); the message is job_id/tile_id/operation/blob_key/geometry.
    """
    tiles = split_image_into_tiles(image, TILE_SIZE)
    sizes = []
    for t in tiles:
        blob_key = f"{job_id}/task/{t['tile_id']}.jpg"
        task = {
            'job_id': job_id,
            'tile_id': t['tile_id'],
            'operation': 'edge_canny',
            'blob_key': blob_key,
            'x': t['x'],
            'y': t['y'],
            'width': t['width'],
            'height': t['height'],
            'timestamp': time.time(),
        }
        sizes.append(len(json.dumps(task).encode('utf-8')))
    return sizes, len(tiles)


def scale_workers(n):
    print(f"  scaling workers -> {n} ...")
    subprocess.run(
        ["docker", "compose", "up", "-d", "--scale", f"worker={n}", "worker"],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )


def wait_for_workers(n, timeout=WORKER_REGISTRATION_TIMEOUT_S):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{MASTER_URL}/health", timeout=5)
            if resp.ok and resp.json().get('active_workers') == n:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def upload_and_wait(image, operation, timeout=JOB_TIMEOUT_S):
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("failed to encode test image")

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
    # Step 6 moved this to a top-level field (a Redis LIST, not embedded in
    # the job hash) - see app/jobstore.py.
    tile_latencies_ms = [t * 1000 for t in status.get('tile_latencies_s', [])]

    # Not timed (measured throughput is about the tile pipeline, not
    # reconstruction) but not optional either: /reconstruct is the only
    # thing that calls delete_job_tiles() on the OpenCV path
    # (master/app.py) - skipping it, as this script always did, leaked
    # every tile's MinIO blobs permanently on every sweep. Found during a
    # full-project review; the leak was in the exact benchmark run that
    # regenerated README.md's speedup chart.
    try:
        requests.post(f"{MASTER_URL}/reconstruct/{job_id}", timeout=30).raise_for_status()
    except requests.RequestException as e:
        print(f"  warning: reconstruct cleanup failed for job {job_id}: {e}")

    return wall_clock_s, tile_latencies_ms, tiles_count


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', default=DEFAULT_OPERATION)
    parser.add_argument('--output', default=str(RESULTS_DIR / 'baseline.csv'))
    args = parser.parse_args()

    try:
        requests.get(f"{MASTER_URL}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        print(f"ERROR: master not reachable at {MASTER_URL}. "
              f"Run `docker compose up -d master` first.", file=sys.stderr)
        sys.exit(1)

    print("Generating test images (deterministic seeds, not written to disk)...")
    images = {name: gen_test_image(w, h, seed=42) for name, w, h in IMAGE_SPECS}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    rows = []

    for n in WORKER_COUNTS:
        scale_workers(n)
        if not wait_for_workers(n):
            print(f"  WARNING: {n} workers did not all register in time, skipping", file=sys.stderr)
            continue

        for name, width, height in IMAGE_SPECS:
            image = images[name]
            print(f"workers={n:>2}  image={name:<6} ({width}x{height})  op={args.operation} ...", end=" ", flush=True)

            try:
                wall_clock_s, tile_latencies_ms, tiles_count = upload_and_wait(image, args.operation)
            except (requests.RequestException, TimeoutError) as e:
                print(f"FAILED: {e}")
                continue

            message_sizes, _ = kafka_task_message_bytes(image, job_id=str(uuid.uuid4()))
            avg_tile_kb = (sum(message_sizes) / len(message_sizes)) / 1024
            total_kb_in = sum(message_sizes) / 1024
            tiles_per_sec = tiles_count / wall_clock_s if wall_clock_s > 0 else 0

            row = {
                'worker_count': n,
                'image_name': name,
                'width': width,
                'height': height,
                'tile_count': tiles_count,
                'operation': args.operation,
                'wall_clock_s': round(wall_clock_s, 3),
                'tiles_per_sec': round(tiles_per_sec, 2),
                'p50_ms': round(percentile(tile_latencies_ms, 50), 2),
                'p95_ms': round(percentile(tile_latencies_ms, 95), 2),
                'p99_ms': round(percentile(tile_latencies_ms, 99), 2),
                'avg_tile_kb': round(avg_tile_kb, 2),
                'total_kb_in': round(total_kb_in, 2),
            }
            rows.append(row)
            print(f"{wall_clock_s:.2f}s  {tiles_per_sec:.1f} tiles/s  p99={row['p99_ms']}ms")

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    run()
