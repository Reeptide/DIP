"""Horizontal scaling sweep for the inference stage - the other axis from
bench/run_inference_batch_sweep.py.

That script scaled *batch size* on a single inference container and found
only a modest ~5-10% gain (CPU compute-bound, not call-overhead-bound). This
script instead scales the *number of inference replicas*
(`docker compose up --scale inference=N`), the same Kafka-consumer-group
pattern (`ml-inference-workers` group on `ml_tasks`, 12 partitions) that
gave the OpenCV worker pool near-linear speedup in bench/run_bench.py. If
that pattern holds for a CPU-bound ML workload too, this is the number that
actually justifies the ML pipeline architecturally.

Runs on the HOST like the other bench scripts - needs the docker CLI and a
running `prometheus` service (used to confirm N replicas are actually up
and scraped before each run, since inference/main.py sends no heartbeat
master/ml_routes.py could otherwise check).
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_URL = "http://localhost:5000"
PROMETHEUS_URL = "http://localhost:9090"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

REPLICA_COUNTS = [1, 2, 4, 8]
IMAGE_SIZE = 6144  # 144 tiles at TILE_SIZE=512 - same fixed workload every run
REPLICA_READY_TIMEOUT_S = 90


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


def scale_inference(n):
    print(f"  scaling inference -> {n} replicas ...")
    subprocess.run(
        ["docker", "compose", "up", "-d", "--scale", f"inference={n}", "inference"],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )


def wait_for_replicas(n, timeout=REPLICA_READY_TIMEOUT_S):
    """Poll Prometheus's own target list rather than any app-level health
    check - inference/main.py sends no heartbeat (see master/ml_routes.py's
    comment on why /detect can't gate on worker count the way /upload does),
    so Prometheus's dns_sd-discovered, actually-scraped-successfully target
    count is the only honest signal that N replicas are both running and
    reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=5)
            targets = resp.json()['data']['activeTargets']
            up = [t for t in targets if t['scrapePool'] == 'dip-inference' and t['health'] == 'up']
            if len(up) == n:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def upload_and_wait(image, timeout=180):
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
        requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: master or prometheus not reachable ({e}). "
              f"Run `docker compose up -d` first.", file=sys.stderr)
        sys.exit(1)

    image = gen_image()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for n in REPLICA_COUNTS:
        scale_inference(n)
        if not wait_for_replicas(n):
            print(f"  WARNING: {n} inference replicas did not all come up healthy in time, skipping", file=sys.stderr)
            continue

        print(f"replicas={n:>2}  ({IMAGE_SIZE}x{IMAGE_SIZE}, 144 tiles) ...", end=" ", flush=True)
        try:
            wall_clock_s, latencies_ms, tiles_count, final_status = upload_and_wait(image)
        except (requests.RequestException, TimeoutError) as e:
            print(f"FAILED: {e}")
            continue

        tiles_per_sec = tiles_count / wall_clock_s if wall_clock_s > 0 else 0
        row = {
            'replica_count': n,
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

    print("Restoring inference to 1 replica...")
    scale_inference(1)
    wait_for_replicas(1)

    out_path = RESULTS_DIR / 'inference_replica_sweep.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    run()
