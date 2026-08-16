"""Prometheus metrics, shared across master, worker, and inference processes.

Each process is a separate container with its own default CollectorRegistry
(prometheus_client's global registry), so metrics never mix between
replicas - Prometheus scrapes each container's own /metrics (master) or
metrics port (worker, inference) independently and aggregates server-side.

`pipeline` label distinguishes the OpenCV path (worker/main.py) from the ML
path (inference/main.py) sharing these same metric names, since both are
"tiles processed" in the same sense.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server

METRICS_PORT = 9200

# Buckets tuned for this system's actual observed range: OpenCV tiles run
# ~15-30ms, ML inference tiles (per-tile YOLO fallback included) ~130-300ms.
TILE_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5,
)

tiles_processed_total = Counter(
    'dip_tiles_processed_total', 'Tiles processed, by pipeline and outcome',
    ['pipeline', 'status'],
)

tile_processing_seconds = Histogram(
    'dip_tile_processing_seconds', 'Per-tile processing time',
    ['pipeline'], buckets=TILE_LATENCY_BUCKETS,
)

dlq_total = Counter(
    'dip_dlq_total', 'Tiles parked on the dead-letter queue after exhausting retries',
    ['pipeline'],
)

dlq_depth = Gauge(
    'dip_dlq_depth', 'Current message count on tasks.dlq (high - low watermark, '
    'summed across partitions) - the live queue size, distinct from dip_dlq_total '
    'which only ever grows. See master/dlq_monitor.py.',
)

tiles_requeued_total = Counter(
    'dip_tiles_requeued_total', 'Tiles republished by the reaper after appearing lost',
)

jobs_created_total = Counter(
    'dip_jobs_created_total', 'Jobs created', ['pipeline'],
)

jobs_finished_total = Counter(
    'dip_jobs_finished_total', 'Jobs that reached a terminal status',
    ['pipeline', 'status'],  # status: completed | degraded
)

active_workers = Gauge(
    'dip_active_workers', 'Workers currently considered alive by the heartbeat tracker',
)

is_leader_gauge = Gauge(
    'dip_is_leader', '1 if this master replica currently holds the Redis-backed '
    'leader lock (see app/leader.py), 0 otherwise - exactly one replica should '
    'read 1 at any given time; summed across replicas this should always be 1.'
    # Named *_gauge, not is_leader, to avoid shadowing app.leader.is_leader()
    # when both are imported into the same module (master/app.py does).
)

queue_depth = Gauge(
    'dip_queue_depth', 'Tasks buffered in-process waiting to be handled',
    ['pipeline'],
)


def serve_metrics(port: int = METRICS_PORT) -> None:
    """Start a background HTTP server exposing /metrics on `port`. Called
    once per worker/inference process - master exposes /metrics as a normal
    Flask route instead, since it already has an HTTP server."""
    start_http_server(port)
