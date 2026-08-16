"""Shared configuration for master, worker, and inference processes.

All values are env-var overridable so the same image runs unmodified in
docker-compose, in a benchmark sweep, or directly on the host.
"""
import os
import socket

# --- Kafka -------------------------------------------------------------
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')

TASK_TOPIC = 'tasks'
RESULT_TOPIC = 'results'
HEARTBEAT_TOPIC = 'heartbeats'
# Terminal failures are parked here instead of being silently committed and
# lost (Step 7). Nothing consumes it in normal operation - it exists so a
# permanently-failing tile is inspectable after the fact rather than
# vanishing from the system with no trace.
TASK_DLQ_TOPIC = 'tasks.dlq'

# ML inference stage (Step 9) - separate topics and consumer group from the
# OpenCV pipeline above, so detection jobs don't compete with filter jobs
# for the same partitions.
ML_TASK_TOPIC = 'ml_tasks'
ML_RESULT_TOPIC = 'ml_results'

# --- Redis ---------------------------------------------------------------
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

# --- MinIO (claim-check blob store, wired in Step 5) --------------------
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'dipadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'dipadmin123')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'dip-tiles')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').lower() == 'true'

# --- Node identity ---------------------------------------------------------
# Hostname fallback, not a fixed default - master is now horizontally
# scalable (`docker compose up --scale master=N`, see app/leader.py), and a
# fixed INSTANCE_ID across replicas would collide (Kafka consumer group IDs,
# log line attribution). Same pattern WORKER_ID already used.
INSTANCE_ID = os.getenv('INSTANCE_ID', f"master-{socket.gethostname()}")
WORKER_ID = os.getenv('WORKER_ID', f"worker-{socket.gethostname()}")

# --- Filesystem ------------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# --- Image tiling ------------------------------------------------------
# TILE_SIZE's default (256) is not a guess - see bench/run_tile_size_sweep.py
# and docs/ARCHITECTURE.md's "Tile size" section: measured throughput across
# 256/384/512/768/1024 for this system's 3 benchmark image sizes. 256 won
# outright and by a wide margin (~2-2.5x over the old 512 default on every
# image size, e.g. 143.9 vs 58.1 tiles/sec on the large image) - smaller
# tiles mean more of them to spread across the worker pool and less
# per-tile OpenCV work, and that outweighs the added per-tile Kafka/MinIO
# overhead at every size tested.
TILE_SIZE = int(os.getenv('TILE_SIZE', 256))
MIN_IMAGE_SIZE = 1024
MAX_IMAGE_DIMENSION = 8192
# 1000 was sized against the old TILE_SIZE=512 default's worst case
# (8192x8192 / 512 = 256 tiles) with generous headroom. Dropping the
# default to 256 raises that worst case to 1024 tiles - just over the old
# cap - which would have silently broken the largest previously-valid
# upload (`ValueError: Image too large`) as a side effect of a throughput
# fix. 1200 keeps the same margin-over-worst-case ratio the original value had.
MAX_TILES = int(os.getenv('MAX_TILES', 1200))  # prevent memory exhaustion on huge images

# --- Heartbeats --------------------------------------------------------
HEARTBEAT_INTERVAL = 5   # worker: how often it sends a heartbeat
HEARTBEAT_TIMEOUT = 15   # master: TTL on each heartbeat:{worker_id} Redis key
                          # (see master/heartbeat.py) - Redis's own expiry
                          # IS the dead-worker sweep now, no polling loop
                          # needed for it.

# --- Fault tolerance / reaper (Step 7) -----------------------------------
# How often the reaper sweeps Redis for jobs with outstanding tiles.
REAPER_SCAN_INTERVAL = int(os.getenv('REAPER_SCAN_INTERVAL', 5))

# How long a job may sit in 'processing' with tiles still missing before
# those tiles are assumed lost and republished. Tiles process in tens of
# milliseconds here, so even a several-hundred-tile job finishes well
# inside this - anything still missing after it really is gone (worker
# killed mid-tile, before its offset was committed). Also the clock between
# successive requeues of the same job, so a requeued tile gets a full
# window to land before being requeued again.
TILE_OUTSTANDING_TIMEOUT_SECONDS = int(os.getenv('TILE_OUTSTANDING_TIMEOUT_SECONDS', 20))

# Total dispatch attempts per tile, counting the master's original publish
# as attempt 1. Past this the tile is given up on and the job is marked
# 'degraded' rather than being requeued forever.
MAX_TILE_ATTEMPTS = int(os.getenv('MAX_TILE_ATTEMPTS', 3))

# --- Rate limiting (master's /upload endpoint) --------------------------
RATE_LIMIT = 10
RATE_WINDOW = 60

# --- Kafka producer batching (master's task publisher) -------------------
BATCH_SIZE = 5
BATCH_FLUSH_INTERVAL = 0.05

# Threads used to encode tiles and PUT them to MinIO while publishing a job.
# The publish loop used to do this one tile at a time on the request thread,
# which made the master - not the worker pool - the throughput ceiling: a
# 144-tile job spent ~30ms per tile serially, slower than a single worker's
# own per-tile latency, so scaling workers changed nothing. The MinIO PUT is
# pure I/O wait, so a modest thread pool hides nearly all of it.
PUBLISH_POOL_SIZE = int(os.getenv('PUBLISH_POOL_SIZE', 12))

# --- ML inference (Step 9) -----------------------------------------------
DETECT_MODEL_PATH = os.getenv('DETECT_MODEL_PATH', '/srv/models/yolov8n.onnx')
CLASSIFY_MODEL_PATH = os.getenv('CLASSIFY_MODEL_PATH', '/srv/models/mobilenetv2-12.onnx')
DETECT_INPUT_SIZE = 640      # YOLOv8's export input resolution (square)
CLASSIFY_INPUT_SIZE = 224    # MobileNetV2's expected input resolution
DETECT_CONF_THRESHOLD = 0.4
DETECT_IOU_THRESHOLD = 0.45  # for non-max suppression

# Whether inference/engine.py attempts CUDAExecutionProvider. Centralized
# here (not just read locally in engine.py) because INFERENCE_BATCH_TIMEOUT_MS
# below needs to know it too - the two are coupled by hardware reality.
INFERENCE_USE_GPU = os.getenv('INFERENCE_USE_GPU', 'false').lower() == 'true'

# Dynamic batching: a tile is inferred as soon as either threshold is hit,
# whichever comes first - INFERENCE_BATCH_SIZE bounds latency under heavy
# load (a full batch fires immediately), INFERENCE_BATCH_TIMEOUT_MS bounds
# latency under light load (a lone tile doesn't wait forever for company).
#
# The timeout's right default value depends on hardware, not just load: a
# stock YOLOv8n forward pass costs ~120-150ms/tile on CPU, so waiting up to
# 50ms to accumulate company barely matters - but on this project's GPU
# (after the sm_120 JIT-stall fix, see inference/engine.py and
# docs/ARCHITECTURE.md) it dropped to ~7ms/tile, and measured directly via
# bench/run_inference_batch_sweep.py against the GPU-enabled service: a
# 50ms wait made batch=1 the FASTEST configuration end-to-end, because the
# wait itself became the dominant cost at low load, not compute. 5ms keeps
# a lone tile from waiting 7x its own compute cost for company that may
# never come, while still coalescing genuinely concurrent tiles that land
# within that much shorter window.
INFERENCE_BATCH_SIZE = int(os.getenv('INFERENCE_BATCH_SIZE', 16))
INFERENCE_BATCH_TIMEOUT_MS = int(os.getenv(
    'INFERENCE_BATCH_TIMEOUT_MS', 5 if INFERENCE_USE_GPU else 50,
))

# Backpressure: pause the Kafka consumer once this many tasks are buffered
# waiting for inference (GPU/CPU can't keep up with ingest), resume once it
# drains back below the low-watermark. Without this, an unbounded consumer
# just keeps pulling from Kafka and building an ever-growing in-memory
# backlog until the process runs out of memory.
INFERENCE_MAX_QUEUE = int(os.getenv('INFERENCE_MAX_QUEUE', 200))
INFERENCE_RESUME_QUEUE = int(os.getenv('INFERENCE_RESUME_QUEUE', 50))


def configure_logging(prefix: str) -> None:
    """Call once per process entrypoint - master and worker each use their
    own prefix so log lines are attributable when running under compose."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{prefix}] %(asctime)s - %(levelname)s - %(message)s'
    )
