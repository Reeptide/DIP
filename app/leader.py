"""Redis-backed leader election for master's singleton duties.

Master is horizontally scalable (`docker compose up --scale master=N`) - its
HTTP routes are already safe to run on every replica simultaneously, since
they only ever read/write Redis, Kafka, and MinIO, never in-process state
(see master/app.py). But a handful of background duties are NOT safe to run
on every replica at once:

- master/reaper.py increments a per-tile attempt counter
  (app.jobstore.incr_tile_attempts) and republishes missing tasks. If N
  replicas all reaped the same stalled job in the same scan window, a tile
  would get its attempt counter incremented N times for one real outage,
  hitting MAX_TILE_ATTEMPTS and giving up after a single genuine failure
  instead of the intended number of retries - a real correctness bug, not
  just wasted work.
- master/results_consumer.py and master/ml_results_consumer.py would each
  redundantly process every result if run on every replica under a shared
  consumer group (see below for why the group is shared) - safe, since
  record_tile_result() is idempotent, but N-way wasted Kafka consumption
  for no benefit.

Elected via a simple single-Redis-instance lock (SET NX PX + a
compare-and-extend Lua script for renewal) - not a multi-node Redlock
consensus, since Redis itself isn't the thing being made highly available
here; it's already a single point of failure for job state (Step 6) and
this doesn't change that. TTL is short (2s, renewed every 0.5s) so a dead
leader's lock expires and a new one is elected within ~2s of a real crash,
not "eventually, once a human notices."
"""
import logging
import threading
import time
import uuid

from app.jobstore import redis_client

logger = logging.getLogger(__name__)

LEADER_KEY = 'dip:master:leader'
LEADER_TTL_MS = 2000
RENEW_INTERVAL_S = 0.5

# A random token, not INSTANCE_ID - uniquely identifies this PROCESS across
# restarts, which the renewal script needs to be safe: it must only extend
# the lock if this exact process still holds it, never a different process
# that happens to share the same INSTANCE_ID (e.g. after a fast restart).
_TOKEN = str(uuid.uuid4())

_is_leader = threading.Event()

# Classic single-instance distributed lock renewal: only extend the TTL if
# the value still matches this holder's token - otherwise another process
# already acquired the lock (this one's previous renewal was too slow, or
# it was never really the leader) and extending would steal it back
# incorrectly.
_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""


def is_leader() -> bool:
    return _is_leader.is_set()


def run_leader_election():
    """Background thread: forever try to become or remain leader."""
    logger.info(f"Leader election started (token={_TOKEN[:8]})")
    renew = redis_client.client.register_script(_RENEW_SCRIPT)

    while True:
        try:
            if _is_leader.is_set():
                renewed = renew(keys=[LEADER_KEY], args=[_TOKEN, LEADER_TTL_MS])
                if not renewed:
                    logger.warning("Lost leadership (renewal failed - lock no longer ours)")
                    _is_leader.clear()
            else:
                acquired = redis_client.client.set(
                    LEADER_KEY, _TOKEN, nx=True, px=LEADER_TTL_MS,
                )
                if acquired:
                    logger.warning(f"Acquired leadership (token={_TOKEN[:8]})")
                    _is_leader.set()
        except Exception as e:
            logger.error(f"Leader election error: {e}", exc_info=True)
            _is_leader.clear()

        time.sleep(RENEW_INTERVAL_S)
