"""Re-run the existing inference sweeps against the GPU-enabled service and
save the results under *_gpu.csv, leaving the CPU baseline CSVs untouched.

bench/run_inference_batch_sweep.py and bench/run_inference_replica_sweep.py
both hardcode their output filename under bench/results/. Rather than edit
either script (and risk clobbering the CPU baselines the whole Step 9
write-up is built on), this wrapper redirects their module-level RESULTS_DIR
at a scratch directory, runs them unmodified, then copies the output to the
_gpu name. The comparison stays apples-to-apples because the sweep logic
itself is byte-identical to the run that produced the CPU numbers.

  python bench/run_gpu_sweeps.py batch     # -> results/inference_batch_sweep_gpu.csv
  python bench/run_gpu_sweeps.py replica   # -> results/inference_replica_sweep_gpu.csv
  python bench/run_gpu_sweeps.py both
"""
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "bench" / "results"


def run_sweep(module_name, produced_name, gpu_name):
    import importlib

    mod = importlib.import_module(module_name)
    scratch = Path(tempfile.mkdtemp(prefix="gpu_sweep_"))
    mod.RESULTS_DIR = scratch
    try:
        mod.run()
    finally:
        produced = scratch / produced_name
        if produced.exists():
            dest = RESULTS_DIR / gpu_name
            shutil.copyfile(produced, dest)
            print(f"--> saved {dest}")
        else:
            print(f"!!! {module_name} produced no CSV at {produced}", file=sys.stderr)


SWEEPS = {
    'batch': ("bench.run_inference_batch_sweep",
              "inference_batch_sweep.csv", "inference_batch_sweep_gpu.csv"),
    'replica': ("bench.run_inference_replica_sweep",
                "inference_replica_sweep.csv", "inference_replica_sweep_gpu.csv"),
}


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    targets = list(SWEEPS) if which == 'both' else [which]
    for t in targets:
        print(f"\n########## {t} sweep (GPU) ##########")
        run_sweep(*SWEEPS[t])
