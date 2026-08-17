"""Redis-backed job/tile state.

Before this rewrite, job state lived in two JSON-blob string keys
(`job:{id}` and `job:{id}:tiles`) that every single tile result rewrote in
full: GET the whole blob, deserialize, mutate one entry, reserialize,
SETEX the whole thing back. That's O(n) work per tile, O(n^2) over a job,
and a read-modify-write race the moment two processes touch the same job
concurrently. The Step 4 benchmark showed exactly this signature -
throughput flat regardless of worker count, because a single-threaded
consumer doing this was the real ceiling, not tile-processing parallelism.

Now:
- `job:{job_id}` is a Redis HASH. Per-tile updates are single-field
  operations (HINCRBY for results_count, HSET for status) - O(1) per call,
  no read-modify-write, no serialized snapshot of the whole job to rewrite.
- `job:{job_id}:tiles` is a Redis HASH keyed by tile_id. Each tile's result
  is one HSET call, independent of how many tiles came before it.
- `job:{job_id}:latencies` is a Redis LIST. Each tile appends with RPUSH -
  O(1) amortized, no rewriting prior entries.

job_id is always the full UUID, not the 8-character prefix the original
code truncated IDs to (`job_id[:8]`) - that scheme had roughly 50% odds of
a collision around 77,000 jobs. The HTTP API still accepts either (see
master/app.py's routes), but Redis keys and MinIO blob paths only ever use
the full ID.
"""
import json
import logging
import time

import redis

from app.config import REDIS_HOST, REDIS_PORT, TILE_SIZE

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 86400


class RedisClient:
    """Wrapper for Redis operations with error handling and retries."""

    def __init__(self, host, port, max_retries=3):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        self.max_retries = max_retries

    def _retry_operation(self, operation, *args, idempotent=True, **kwargs):
        """Retry Redis operation with exponential backoff.

        idempotent=False (incr/hincrby/rpush/sadd - anything where re-running
        the same call changes the result, unlike a plain set/get) narrows
        what counts as safe to retry: a ConnectionError means the command
        never reached the server (retrying is exactly as safe as the first
        attempt), but a TimeoutError only means the client gave up waiting
        for the reply - the command may have already been applied server-
        side. Retrying an hincrby after a timeout can silently double-count.
        Found during a full-project review: incr_tile_attempts() retrying
        blindly on any redis.RedisError (TimeoutError included) could burn
        a tile's retry budget and get it abandoned after fewer than
        MAX_TILE_ATTEMPTS real failures - the same class of bug leader
        election was added to prevent, arriving through Redis instead of
        two masters. Non-idempotent operations now raise immediately on
        TimeoutError instead of retrying blind.
        """
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except redis.TimeoutError as e:
                if not idempotent:
                    logger.error(f"Redis operation timed out (not retrying - not idempotent): {e}")
                    raise
                if attempt == self.max_retries - 1:
                    logger.error(f"Redis operation failed after {self.max_retries} attempts: {e}")
                    raise
                wait_time = 0.1 * (2 ** attempt)
                logger.warning(f"Redis timeout, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            except redis.RedisError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Redis operation failed after {self.max_retries} attempts: {e}")
                    raise
                wait_time = 0.1 * (2 ** attempt)
                logger.warning(f"Redis error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

    def set(self, key, value, ex=None):
        return self._retry_operation(self.client.set, key, value, ex=ex)

    def get(self, key):
        return self._retry_operation(self.client.get, key)

    def incr(self, key):
        return self._retry_operation(self.client.incr, key, idempotent=False)

    def delete(self, *keys):
        return self._retry_operation(self.client.delete, *keys)

    def hset(self, key, mapping):
        return self._retry_operation(self.client.hset, key, mapping=mapping)

    def hget(self, key, field):
        return self._retry_operation(self.client.hget, key, field)

    def hgetall(self, key):
        return self._retry_operation(self.client.hgetall, key)

    def hincrby(self, key, field, amount=1):
        return self._retry_operation(self.client.hincrby, key, field, amount, idempotent=False)

    def hlen(self, key):
        return self._retry_operation(self.client.hlen, key)

    def rpush(self, key, value):
        return self._retry_operation(self.client.rpush, key, value, idempotent=False)

    def sadd(self, key, value):
        return self._retry_operation(self.client.sadd, key, value, idempotent=False)

    def smembers(self, key):
        return self._retry_operation(self.client.smembers, key)

    def lrange(self, key, start, end):
        return self._retry_operation(self.client.lrange, key, start, end)

    def expire(self, key, seconds):
        return self._retry_operation(self.client.expire, key, seconds)

    def ping(self):
        try:
            return self.client.ping()
        except redis.RedisError:
            return False


redis_client = RedisClient(REDIS_HOST, REDIS_PORT)


# ============================================================================
# Job-level metadata (job:{job_id} hash)
# ============================================================================

def create_job(job_id, *, original_filename, operation, original_width,
                original_height, expected_tiles, active_workers, tile_size=TILE_SIZE):
    """Create a job's metadata hash. Called once, at upload time.

    tile_size is stored per-job, not read from the global TILE_SIZE default
    at reap time - TILE_SIZE is now a runtime-overridable config value (see
    app/config.py's tile-size sweep story), so a job created under one
    value and reaped after a config change (or, in this repo, literally
    reaped by bench/run_tile_size_sweep.py restarting master with a
    different TILE_SIZE) would have its tile geometry recomputed wrong -
    tile_geometries() would silently produce a different tile_id layout
    than what was actually dispatched.
    """
    now = time.time()
    key = f"job:{job_id}"
    redis_client.hset(key, {
        'job_id': job_id,
        'original_filename': original_filename,
        'operation': operation,
        'original_width': original_width,
        'original_height': original_height,
        'tile_size': tile_size,
        'timestamp': now,
        'processing_start': now,
        'status': 'processing',
        'expected_tiles': expected_tiles,
        'active_workers': active_workers,
        'results_count': 0,
        'failed_tasks': 0,
    })
    redis_client.expire(key, JOB_TTL_SECONDS)


def get_job(job_id):
    """Fetch job metadata as a dict with proper types, or None if missing."""
    raw = redis_client.hgetall(f"job:{job_id}")
    if not raw:
        return None
    return {
        'job_id': raw.get('job_id', job_id),
        'original_filename': raw.get('original_filename', 'unknown'),
        'operation': raw.get('operation', 'unknown'),
        'original_width': int(raw.get('original_width', 0)),
        'original_height': int(raw.get('original_height', 0)),
        # Fallback to the current default only for jobs created before this
        # field existed - a live job always has it set at create_job() time.
        'tile_size': int(raw.get('tile_size', TILE_SIZE)),
        'timestamp': float(raw.get('timestamp', 0)),
        'processing_start': float(raw.get('processing_start', 0)),
        'status': raw.get('status', 'unknown'),
        'expected_tiles': int(raw.get('expected_tiles', 0)),
        'active_workers': int(raw.get('active_workers', 0)),
        'results_count': int(raw.get('results_count', 0)),
        'failed_tasks': int(raw.get('failed_tasks', 0)),
        'completion_time': float(raw['completion_time']) if 'completion_time' in raw else None,
        'result_path': raw.get('result_path'),
    }


def mark_job_completed(job_id, result_path):
    key = f"job:{job_id}"
    redis_client.hset(key, {
        'status': 'completed',
        'result_path': result_path,
        'completion_time': time.time(),
    })
    # A normal HSET on an already-live hash preserves its existing TTL, but
    # if job:{id} had already expired (job older than JOB_TTL_SECONDS) and
    # a straggler write like this one arrives anyway, HSET recreates it
    # fresh with NO TTL - an immortal key. Re-applying the TTL on every
    # write closes that gap; found during a full-project review, alongside
    # the same fix in set_job_status()/set_last_reap_time() below.
    redis_client.expire(key, JOB_TTL_SECONDS)


# ============================================================================
# Per-tile results (job:{job_id}:tiles hash, job:{job_id}:latencies list)
# ============================================================================

def record_tile_result(job_id, tile_id, tile_info: dict, processing_time: float) -> int:
    """Store one tile's result. Returns the count of *distinct* tiles received.

    Delivery is at-least-once by design (the worker commits its offset only
    after publishing a result, and the Step 7 reaper may requeue a tile that
    turns out to have succeeded after all), so the same tile_id can legitimately
    arrive twice. The HSET is naturally idempotent - a repeat just overwrites
    the same field - but the results_count HINCRBY was not: it counted
    *messages*, so a duplicate inflated it past the number of distinct tiles
    actually held. Since completion was decided by comparing that counter to
    expected_tiles, a job could be declared complete while a distinct tile was
    still genuinely missing, and reconstruction would then fail with "Not all
    tiles received" on a job whose status already said ready_for_reconstruction.

    Fix: HSET reports whether the field was new (1) or an overwrite (0), so
    the counter and the latency sample are only recorded for first delivery,
    and the returned completion signal is HLEN - the true distinct count -
    rather than a counter that can only be trusted if nothing is ever redelivered.
    """
    tiles_key = f"job:{job_id}:tiles"
    latencies_key = f"job:{job_id}:latencies"

    is_new_tile = redis_client.hset(tiles_key, {str(tile_id): json.dumps(tile_info)})
    redis_client.expire(tiles_key, JOB_TTL_SECONDS)

    if is_new_tile:
        redis_client.rpush(latencies_key, processing_time)
        redis_client.expire(latencies_key, JOB_TTL_SECONDS)
        job_key = f"job:{job_id}"
        redis_client.hincrby(job_key, 'results_count', 1)
        redis_client.expire(job_key, JOB_TTL_SECONDS)  # see mark_job_completed()
    else:
        logger.info(f"Duplicate result for job {job_id[:8]} tile {tile_id} ignored (at-least-once delivery)")

    return redis_client.hlen(tiles_key)


def set_job_status(job_id, status):
    key = f"job:{job_id}"
    redis_client.hset(key, {'status': status})
    redis_client.expire(key, JOB_TTL_SECONDS)  # see mark_job_completed()


def get_tile_results(job_id) -> dict:
    """All received tile results for a job: {tile_id (int) -> tile_info dict}."""
    raw = redis_client.hgetall(f"job:{job_id}:tiles")
    return {int(tile_id): json.loads(value) for tile_id, value in raw.items()}


def count_tile_results(job_id) -> int:
    return redis_client.hlen(f"job:{job_id}:tiles")


def get_tile_latencies_ms(job_id) -> list:
    raw = redis_client.lrange(f"job:{job_id}:latencies", 0, -1)
    return [float(v) * 1000 for v in raw]


# ============================================================================
# Requeue bookkeeping (Step 7 reaper)
# ============================================================================

def scan_job_ids() -> list:
    """Every job id currently in Redis, via SCAN (never KEYS - this runs on a
    loop against a live server). Matches only the `job:{id}` metadata hashes,
    not the `:tiles` / `:latencies` companions."""
    job_ids = []
    cursor = '0'
    while cursor != 0:
        cursor, keys = redis_client.client.scan(cursor=cursor, match='job:*', count=1000)
        job_ids.extend(key[4:] for key in keys if ':' not in key[4:])
    return job_ids


def incr_tile_attempts(job_id, tile_id) -> int:
    """Count how many times the reaper has requeued this tile. Kept in Redis
    rather than in the reaper's memory so the cap survives a master restart -
    otherwise a crash-looping master would requeue a poison tile forever."""
    key = f"job:{job_id}:tile:{tile_id}:attempts"
    count = redis_client.incr(key)
    redis_client.expire(key, JOB_TTL_SECONDS)
    return count


def add_abandoned_tile(job_id, tile_id) -> None:
    """Record a tile the reaper has permanently given up on (exceeded
    MAX_TILE_ATTEMPTS, or its source blob is gone). Persisted so a poison
    tile is never retried again, and - the reason this exists as its own
    set rather than just being implied by the job flipping to 'degraded' -
    so ONE unrecoverable tile doesn't have to mean giving up on every OTHER
    tile still legitimately in flight in the same job. See reap_once()'s
    completion check."""
    key = f"job:{job_id}:abandoned"
    redis_client.sadd(key, str(tile_id))
    redis_client.expire(key, JOB_TTL_SECONDS)


def get_abandoned_tiles(job_id) -> set:
    members = redis_client.smembers(f"job:{job_id}:abandoned") or set()
    return {int(t) for t in members}


def get_last_reap_time(job_id) -> float:
    raw = redis_client.hget(f"job:{job_id}", 'last_reap_time')
    return float(raw) if raw else 0.0


def set_last_reap_time(job_id, when: float):
    key = f"job:{job_id}"
    redis_client.hset(key, {'last_reap_time': when})
    redis_client.expire(key, JOB_TTL_SECONDS)  # see mark_job_completed()


def delete_job_tile_state(job_id):
    """Delete the tiles/latencies keys after reconstruction - the job's own
    metadata hash (job:{job_id}) is kept so completed jobs still show up in
    /metadata's job listing."""
    redis_client.delete(f"job:{job_id}:tiles", f"job:{job_id}:latencies", f"job:{job_id}:abandoned")

    # The reaper's per-tile attempt counters are only meaningful while the
    # job is in flight; they carry a TTL as a backstop but a completed job
    # shouldn't leave them behind for a day.
    cursor = '0'
    attempt_keys = []
    while cursor != 0:
        cursor, keys = redis_client.client.scan(
            cursor=cursor, match=f"job:{job_id}:tile:*:attempts", count=1000)
        attempt_keys.extend(keys)
    if attempt_keys:
        redis_client.delete(*attempt_keys)
