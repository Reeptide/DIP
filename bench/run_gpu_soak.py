"""Sustained-load soak test for the inference stage, used to qualify the GPU path.

The GPU hang this was written to catch never showed up in a single-shot
request - it needed minutes of continuous work before the inference thread
would wedge (100%+ CPU, zero forward progress, no exception, clearing on its
own later). So this keeps CONCURRENCY jobs in flight against POST /detect for
DURATION_S seconds and watches for that exact signature:

  * per-job wall clock, so a stall shows up as an outlier not an average
  * a watchdog that flags any job exceeding STALL_THRESHOLD_S
  * periodic nvidia-smi + docker stats samples, so "busy CPU, idle GPU" (the
    PTX-JIT signature) is distinguishable from "both busy" (real work)

Runs on the HOST like the other bench/ scripts - needs the docker CLI and a
`docker compose up -d` stack already running.

  python bench/run_gpu_soak.py [duration_seconds]
"""
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_URL = "http://localhost:5000"

DURATION_S = 360
CONCURRENCY = 2
IMAGE_SIZE = 2048          # 64 tiles at the current TILE_SIZE=256 default
STALL_THRESHOLD_S = 45     # a job this slow is a stall, not just a slow job
SAMPLE_EVERY_S = 15


def gen_image(seed):
    rng = np.random.default_rng(seed)
    xv, yv = np.meshgrid(np.linspace(0, 255, IMAGE_SIZE), np.linspace(0, 255, IMAGE_SIZE))
    image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.uint8)
    for _ in range(12):
        c = (int(rng.integers(0, IMAGE_SIZE)), int(rng.integers(0, IMAGE_SIZE)))
        r = int(rng.integers(30, 200))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.circle(image, c, r, color, thickness=-1)
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def submit_and_wait(payload, timeout):
    t0 = time.time()
    resp = requests.post(
        f"{MASTER_URL}/detect",
        files={'image': ('soak.jpg', payload, 'image/jpeg')},
        timeout=30,
    )
    resp.raise_for_status()
    job_id = resp.json()['job_id']
    tiles = resp.json()['tiles_count']

    deadline = time.time() + timeout
    while time.time() < deadline:
        st = requests.get(f"{MASTER_URL}/detect/status/{job_id}", timeout=10).json()
        if st.get('status') in ('completed', 'degraded'):
            return time.time() - t0, tiles, st.get('status')
        time.sleep(0.25)
    return time.time() - t0, tiles, 'TIMEOUT'


def sample_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return tuple(x.strip() for x in out.split(','))
    except Exception:
        return ('?', '?')


def sample_cpu():
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}",
             "dip-inference-1"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return out or '?'
    except Exception:
        return '?'


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else DURATION_S
    requests.get(f"{MASTER_URL}/health", timeout=5).raise_for_status()

    print(f"generating {CONCURRENCY} source images ({IMAGE_SIZE}x{IMAGE_SIZE}) ...")
    payloads = [gen_image(i) for i in range(CONCURRENCY)]

    stop_at = time.time() + duration
    lock = threading.Lock()
    results = []   # (elapsed_s, tiles, status)
    stalls = []

    def worker(idx):
        while time.time() < stop_at:
            try:
                elapsed, tiles, status = submit_and_wait(payloads[idx], STALL_THRESHOLD_S * 2)
            except Exception as e:
                with lock:
                    stalls.append(f"t+{time.time() - t_start:6.0f}s EXCEPTION {e}")
                time.sleep(1)
                continue
            with lock:
                results.append((elapsed, tiles, status))
                if elapsed > STALL_THRESHOLD_S or status != 'completed':
                    stalls.append(
                        f"t+{time.time() - t_start:6.0f}s job took {elapsed:.1f}s status={status}"
                    )

    t_start = time.time()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(CONCURRENCY)]
    for t in threads:
        t.start()

    next_sample = time.time()
    while any(t.is_alive() for t in threads):
        if time.time() >= next_sample:
            gpu_util, gpu_mem = sample_gpu()
            with lock:
                done = len(results)
                recent = results[-6:]
            avg = statistics.mean(r[0] for r in recent) if recent else 0
            print(f"t+{time.time() - t_start:6.0f}s  jobs={done:4d}  "
                  f"recent_job_wall={avg:6.2f}s  gpu={gpu_util}%  gpu_mem={gpu_mem}MiB  "
                  f"inference_cpu={sample_cpu()}", flush=True)
            next_sample = time.time() + SAMPLE_EVERY_S
        time.sleep(0.5)
    for t in threads:
        t.join()

    total = time.time() - t_start
    walls = [r[0] for r in results]
    tiles = sum(r[1] for r in results)
    print("\n" + "=" * 70)
    print(f"soak duration      : {total:.0f}s")
    print(f"jobs completed     : {len(results)}")
    print(f"tiles processed    : {tiles}  ({tiles / total:.1f} tiles/s aggregate)")
    if walls:
        print(f"job wall clock     : min={min(walls):.2f}s  "
              f"p50={statistics.median(walls):.2f}s  "
              f"p95={np.percentile(walls, 95):.2f}s  max={max(walls):.2f}s")
    print(f"non-completed/stalls: {len(stalls)}")
    for s in stalls:
        print("   " + s)
    print("=" * 70)
    return 1 if stalls else 0


if __name__ == '__main__':
    sys.exit(main())
