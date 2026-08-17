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
  just wasted work. This is the hard requirement leader election exists
  for.
- master/results_consumer.py and master/ml_results_consumer.py are also
  gated, though the reason is narrower than it might look: a shared Kafka
  consumer group across N master replicas would NOT duplicate work -
  Kafka's normal group protocol splits partitions among concurrent
  members, same as it does for `worker`. record_tile_result() being
  idempotent means even genuinely-concurrent processing across replicas
  would stay correct. Gating to the leader here is a simplicity choice -
  one process owns all Kafka-side job-completion logic at a time, so
  there's one place to reason about the completion-status transition and
  offset-commit ordering, not N possibly-interleaved ones - not a bug fix
  the way the reaper's gating is.

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

# The actual source of truth for is_leader() - a monotonic deadline, not
# just the Event above. Without this, is_leader() could keep returning True
# for up to a full RENEW_INTERVAL_S (or longer, if a renewal call itself
# blocks - the Redis client's socket_timeout is 5s, longer than the 2s
# lock TTL) after this process's lock has already expired in Redis and a
# different replica has legitimately acquired it. That window is a real
# split-brain gap, not a theoretical one: two processes can both believe
# they're leader simultaneously. time.monotonic(), not time.time(), since
# it can't jump backwards on an NTP correction.
_expires_at = 0.0

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

# Same compare-then-act shape as renewal, for releasing on our own way out:
# only delete the key if it still holds our token, never someone else's
# (e.g. a process that already won the lock legitimately after we lost it).
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def is_leader() -> bool:
    return _is_leader.is_set() and time.monotonic() < _expires_at


def run_leader_election():
    """Background thread: forever try to become or remain leader."""
    logger.info(f"Leader election started (token={_TOKEN[:8]})")
    renew = redis_client.client.register_script(_RENEW_SCRIPT)
    release = redis_client.client.register_script(_RELEASE_SCRIPT)

    global _expires_at

    while True:
        try:
            if _is_leader.is_set():
                renewed = renew(keys=[LEADER_KEY], args=[_TOKEN, LEADER_TTL_MS])
                if renewed:
                    _expires_at = time.monotonic() + LEADER_TTL_MS / 1000
                else:
                    logger.warning("Lost leadership (renewal failed - lock no longer ours)")
                    _is_leader.clear()
                    _expires_at = 0.0
            else:
                acquired = redis_client.client.set(
                    LEADER_KEY, _TOKEN, nx=True, px=LEADER_TTL_MS,
                )
                if acquired:
                    logger.warning(f"Acquired leadership (token={_TOKEN[:8]})")
                    _expires_at = time.monotonic() + LEADER_TTL_MS / 1000
                    _is_leader.set()
        except Exception as e:
            logger.error(f"Leader election error: {e}", exc_info=True)
            # If this process still holds the lock in Redis (the failure was
            # e.g. a transient timeout on this specific call, not on every
            # call), an unconditional clear() here without releasing would
            # leave OUR token sitting in Redis - blocking this same process
            # from re-acquiring via SET NX until the full TTL naturally
            # expires, even though nothing else has actually failed over.
            # Best-effort release closes that self-inflicted gap.
            if _is_leader.is_set():
                try:
                    release(keys=[LEADER_KEY], args=[_TOKEN])
                except Exception:
                    pass  # Redis itself may be the thing that's down - fine, TTL still covers us
            _is_leader.clear()
            _expires_at = 0.0

        time.sleep(RENEW_INTERVAL_S)
