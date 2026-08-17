"""Shared @rate_limit decorator - its own module (not master/app.py) so
master/ml_routes.py can import it without a circular import (master/app.py
already imports ml_bp FROM ml_routes.py). Originally lived only in
master/app.py, gating /upload but not /detect - a full-project review
found /detect had no rate limit applied at all.

Two known, honest limitations, not silently fixed away here:
1. In-process only. With `docker compose up --scale master=N` behind
   master-proxy, each replica has its own request_counts dict - the
   effective limit is N x RATE_LIMIT, not RATE_LIMIT. A real fix means a
   Redis-backed shared counter (INCR + EXPIRE per window), which is a
   bigger change than a bug fix belongs to be - flagged plainly instead of
   built.
2. X-Forwarded-For is trusted as-is once past nginx, with no allowlist of
   trusted proxies - a client could set its own X-Forwarded-For to evade
   the limit by IP-hopping in the header. Fine for a single-nginx demo
   deployment; not a substitute for real abuse protection.
"""
import threading
import time
from collections import defaultdict
from functools import wraps

from flask import jsonify, request

from app.config import RATE_LIMIT, RATE_WINDOW

request_counts = defaultdict(list)
# Flask runs threaded (master/app.py's app.run(threaded=True)), and
# defaultdict's __getitem__ mutates the dict (auto-vivifies a missing key)
# - so decorated_function() below inserting a new IP mid-sweep could hit
# _sweep_stale_ips()'s `request_counts.items()` iteration and raise
# "RuntimeError: dictionary changed size during iteration". Found during a
# full-project review: this only fired on the 1-in-500-call sweep under
# real concurrency, so it looked like a random flake rather than a
# deterministic bug. One lock around every access is simpler and cheap
# enough here (this is a rate limiter, not a hot inner loop) rather than
# trying to make the sweep iteration itself safe against concurrent
# mutation.
_lock = threading.Lock()
_rate_limit_calls = 0
_SWEEP_EVERY_N_CALLS = 500  # bound dict growth without sweeping on every request


def _sweep_stale_ips(now):
    """Drop IPs with nothing left in-window - without this, request_counts
    keeps one entry per distinct IP ever seen for the process's lifetime,
    unbounded. Only the current IP's own list gets pruned per-request (see
    rate_limit() below); this periodic full sweep is what actually bounds
    total memory. Caller must hold _lock."""
    stale = [ip for ip, timestamps in request_counts.items()
             if not any(now - t < RATE_WINDOW for t in timestamps)]
    for ip in stale:
        del request_counts[ip]


def _client_ip():
    """request.remote_addr alone is master-proxy's own container IP for
    EVERY client once traffic goes through nginx (docker-compose.yml's
    master-proxy service) - every request would share one bucket. nginx
    sets X-Forwarded-For (monitoring/nginx/master-proxy.conf); use its
    first (left-most / original-client) entry when present."""
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr


def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global _rate_limit_calls
        ip = _client_ip()
        now = time.time()

        with _lock:
            pruned = [t for t in request_counts[ip] if now - t < RATE_WINDOW]

            if len(pruned) >= RATE_LIMIT:
                request_counts[ip] = pruned
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

            pruned.append(now)
            request_counts[ip] = pruned

            _rate_limit_calls += 1
            if _rate_limit_calls % _SWEEP_EVERY_N_CALLS == 0:
                _sweep_stale_ips(now)

        return f(*args, **kwargs)

    return decorated_function
