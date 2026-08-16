"""Renders bench/results/<csv> into a speedup-vs-workers chart.

Two panels: raw throughput (tiles/sec vs worker count, one line per image
size) and speedup relative to the 1-worker run, against an ideal-linear
reference line. Where the real line pulls away from ideal is the
interesting part - Step 3's fix removes the *hard* ceiling at 2 workers,
but reconstruction and Kafka overhead still bend the curve away from
linear at higher counts. That bend is worth explaining, not hiding.

Usage: python -m bench.plot [--input bench/results/baseline.csv]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless - no X server needed on the host
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def load(csv_path):
    by_image = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            by_image[row['image_name']].append({
                'worker_count': int(row['worker_count']),
                'tiles_per_sec': float(row['tiles_per_sec']),
            })
    for rows in by_image.values():
        rows.sort(key=lambda r: r['worker_count'])
    return by_image


def plot(csv_path, out_path, title_suffix=""):
    by_image = load(csv_path)
    if not by_image:
        raise SystemExit(f"No data in {csv_path}")

    fig, (ax_throughput, ax_speedup) = plt.subplots(1, 2, figsize=(12, 5))

    for name, rows in by_image.items():
        workers = [r['worker_count'] for r in rows]
        tps = [r['tiles_per_sec'] for r in rows]
        ax_throughput.plot(workers, tps, marker='o', label=name)

        base = tps[0] if tps[0] > 0 else 1
        speedup = [t / base for t in tps]
        ax_speedup.plot(workers, speedup, marker='o', label=name)

    all_workers = sorted({r['worker_count'] for rows in by_image.values() for r in rows})
    ax_speedup.plot(all_workers, [w / all_workers[0] for w in all_workers],
                     linestyle='--', color='gray', label='ideal (linear)')

    ax_throughput.set_xlabel('Worker count')
    ax_throughput.set_ylabel('Tiles / sec')
    ax_throughput.set_title(f'Throughput{title_suffix}')
    ax_throughput.legend()
    ax_throughput.grid(alpha=0.3)

    ax_speedup.set_xlabel('Worker count')
    ax_speedup.set_ylabel('Speedup vs 1 worker')
    ax_speedup.set_title(f'Speedup{title_suffix}')
    ax_speedup.legend()
    ax_speedup.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(RESULTS_DIR / 'baseline.csv'))
    parser.add_argument('--output', default=str(RESULTS_DIR / 'speedup.png'))
    parser.add_argument('--title-suffix', default='')
    args = parser.parse_args()
    plot(args.input, args.output, args.title_suffix)


if __name__ == '__main__':
    main()
