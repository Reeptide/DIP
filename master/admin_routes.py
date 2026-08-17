"""Diagnostic and maintenance routes, split out of the core Flask app so
app.py stays focused on the actual upload -> process -> reconstruct path.

Reads job:{id} via app.jobstore.get_job() (a Redis HASH since Step 6), not
raw GET+json.loads - that stopped working the moment job:{id} became a
hash instead of a JSON-blob string (Redis raises WRONGTYPE on GET against
a hash key).
"""
import logging
import time
from datetime import datetime
from functools import wraps

from flask import Blueprint, jsonify, request

from app.config import ADMIN_TOKEN
from app.jobstore import redis_client, get_job, JOB_TTL_SECONDS
from master.heartbeat import heartbeat_tracker

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)


def require_admin_token(f):
    """Gate a destructive admin route behind a shared secret. See
    app/config.py's ADMIN_TOKEN for why this exists and what it isn't."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not ADMIN_TOKEN:
            return jsonify({'error': 'Admin routes are disabled (ADMIN_TOKEN not configured)'}), 503
        if request.headers.get('X-Admin-Token') != ADMIN_TOKEN:
            return jsonify({'error': 'Missing or invalid X-Admin-Token header'}), 401
        return f(*args, **kwargs)
    return wrapped


def _job_ids_from_keys(all_keys):
    """job:{id} keys (not :tiles or :latencies suffixed) -> bare job ids."""
    return [
        key[4:] for key in all_keys
        if key.startswith('job:') and ':' not in key[4:]
    ]


def _scan_and_classify_keys():
    """One full Redis key scan, classified into job/tile-tracking/system/
    counter/orphaned buckets. Shared by /metadata and /metadata/cleanup so
    the two can't drift on what counts as an orphan - they used to (this
    function didn't exist; /metadata/cleanup didn't either) until a review
    found the orphan-detecting UI button called an endpoint that was never
    implemented."""
    all_keys = []
    cursor = '0'
    while cursor != 0:
        cursor, keys = redis_client.client.scan(cursor=cursor, count=1000)
        all_keys.extend(keys)

    job_ids = set(_job_ids_from_keys(all_keys))
    job_keys = [f"job:{jid}" for jid in job_ids]
    # ':attempts' keys are the reaper's per-tile requeue counters (Step 7);
    # ':abandoned' is the reaper's per-job unrecoverable-tile set (see
    # app/jobstore.py's add_abandoned_tile) - both are job-scoped state,
    # not orphans.
    tile_tracking_keys = [
        k for k in all_keys
        if k.endswith(':tiles') or k.endswith(':latencies')
        or k.endswith(':attempts') or k.endswith(':abandoned')
    ]
    # heartbeat:{worker_id} (master/heartbeat.py) and dip:master:leader
    # (app/leader.py) are both real, expected, short-TTL system state, not
    # orphans - without this they'd get swept up and reported (or on a
    # real cleanup run, deleted) as garbage.
    system_keys = [
        k for k in all_keys
        if k.startswith('heartbeat:') or k == 'dip:master:leader'
    ]
    counter_keys = [k for k in all_keys if k in ('active_workers_count', 'active_workers')]

    classified = set(job_keys) | set(tile_tracking_keys) | set(system_keys) | set(counter_keys)
    orphaned_keys = [k for k in all_keys if k not in classified]

    return {
        'all_keys': all_keys,
        'job_ids': job_ids,
        'job_keys': job_keys,
        'tile_tracking_keys': tile_tracking_keys,
        'system_keys': system_keys,
        'counter_keys': counter_keys,
        'orphaned_keys': orphaned_keys,
    }


@admin_bp.route('/metadata', methods=['GET'])
def get_metadata():
    """Comprehensive metadata about jobs and Redis storage."""
    try:
        try:
            scan = _scan_and_classify_keys()
        except Exception as e:
            logger.error(f"Error scanning Redis keys: {e}")
            scan = {'all_keys': [], 'job_ids': set(), 'job_keys': [],
                    'tile_tracking_keys': [], 'system_keys': [],
                    'counter_keys': [], 'orphaned_keys': []}

        all_keys = scan['all_keys']
        job_ids = scan['job_ids']
        job_keys = scan['job_keys']
        tile_tracking_keys = scan['tile_tracking_keys']
        counter_keys = scan['counter_keys']
        orphaned_keys = scan['orphaned_keys']

        jobs = []
        for jid in job_ids:
            try:
                job_data = get_job(jid)
                if not job_data:
                    continue

                created_at = job_data['timestamp'] or time.time()
                completion_time = job_data['completion_time']
                processing_time = (completion_time - created_at) if completion_time else None

                tiles_count = job_data['expected_tiles']
                rows = int(tiles_count ** 0.5) or 1
                cols = (tiles_count + rows - 1) // rows

                jobs.append({
                    'job_id': jid[:8],
                    'full_job_id': jid,
                    'operation': job_data['operation'],
                    'original_filename': job_data['original_filename'],
                    'status': job_data['status'],
                    'total_tiles': str(job_data['expected_tiles']),
                    'tiles_received': str(job_data['results_count']),
                    'failed_tasks': str(job_data['failed_tasks']),
                    'rows': str(rows),
                    'cols': str(cols),
                    'created_at': str(int(created_at)),
                    'created_at_readable': datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S'),
                    'image_size': f"{job_data['original_width']}x{job_data['original_height']}",
                    'active_workers': str(job_data['active_workers']),
                    'processing_time': f"{processing_time:.2f}s" if processing_time else 'N/A',
                    'completion_time': completion_time,
                    'result_available': bool(job_data['result_path']),
                    'progress': int((job_data['results_count'] / max(job_data['expected_tiles'], 1)) * 100)
                })
            except Exception as e:
                logger.error(f"Error processing job {jid}: {e}")

        jobs.sort(key=lambda x: float(x['created_at']), reverse=True)

        current_counters = {}
        for key in counter_keys:
            try:
                current_counters[key] = redis_client.get(key) or '0'
            except Exception:
                current_counters[key] = '0'

        active_workers = heartbeat_tracker.get_active_workers()

        return jsonify({
            'total_jobs': len(jobs),
            'jobs': jobs,
            'redis_summary': {
                'total_keys': len(all_keys),
                'job_keys': len(job_keys),
                'tile_keys': len(tile_tracking_keys),
                'counter_keys': len(counter_keys),
                'orphaned_keys': len(orphaned_keys)
            },
            'current_counters': current_counters,
            'active_workers': {
                'count': len(active_workers),
                'workers': active_workers
            },
            'cleanup_info': {
                'orphaned_keys': orphaned_keys[:20],
                'total_orphaned': len(orphaned_keys),
                'cleanup_recommended': len(orphaned_keys) > 5,
            },
            'all_keys': sorted(all_keys)[:50],
            'timestamp': time.time(),
            'timestamp_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        }), 200

    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/metadata/cleanup', methods=['POST'])
@require_admin_token
def cleanup_orphaned_keys():
    """Delete only orphaned Redis keys (job-scoped keys whose job:{id} no
    longer exists, plus anything else _scan_and_classify_keys() can't
    explain). `dry_run: true` (the default, and what templates/index.html's
    "Cleanup Orphaned Keys" button always calls first) reports what would
    be deleted without deleting anything.

    This route didn't exist until now - the UI button already called it
    and expected this exact {dry_run, orphaned_keys_found, valid_jobs,
    total_keys} response shape, silently 404ing every time, found during a
    full-project review. Implemented to match what the frontend already
    expected rather than changing the frontend to match a different,
    already-existing route (nuclear-cleanup and clear both do something
    materially different - a full flush-and-restore and an age-based
    purge, respectively - neither is "just delete the orphans").
    """
    data = request.get_json() or {}
    dry_run = data.get('dry_run', True)

    try:
        scan = _scan_and_classify_keys()
        orphaned = scan['orphaned_keys']

        if not dry_run and orphaned:
            redis_client.delete(*orphaned)
            logger.warning(f"Deleted {len(orphaned)} orphaned key(s)")

        return jsonify({
            'dry_run': dry_run,
            'orphaned_keys_found': len(orphaned),
            'valid_jobs': len(scan['job_ids']),
            'total_keys': len(scan['all_keys']),
            'deleted': 0 if dry_run else len(orphaned),
        }), 200
    except Exception as e:
        logger.error(f"Cleanup orphaned keys failed: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/metadata/nuclear-cleanup', methods=['POST'])
@require_admin_token
def nuclear_cleanup():
    """Delete ALL non-essential Redis keys, keeping only job records.

    Gated behind require_admin_token: this is a FLUSHDB. Found completely
    unauthenticated during a full-project review, reachable through the
    proxy with nothing but a JSON body - and it deletes app/leader.py's
    `dip:master:leader` key along with everything else, which is a
    deterministic way to trigger split-brain (a follower can immediately
    SET NX and win while the old leader's local state hasn't caught up),
    not just data loss.
    """
    try:
        data = request.get_json() or {}
        if not data.get('confirm_nuclear', False):
            return jsonify({
                'error': 'Nuclear cleanup requires confirmation',
                'message': 'Add "confirm_nuclear": true to proceed',
                'warning': 'This will delete ALL Redis data except job definitions'
            }), 400

        logger.warning("NUCLEAR CLEANUP INITIATED!")

        all_keys = []
        cursor = '0'
        while cursor != 0:
            cursor, keys = redis_client.client.scan(cursor=cursor, match='job:*', count=100)
            all_keys.extend(keys)
        job_keys = [f"job:{jid}" for jid in _job_ids_from_keys(all_keys)]

        job_backup = {key: redis_client.hgetall(key) for key in job_keys}
        job_backup = {k: v for k, v in job_backup.items() if v}

        redis_client.client.flushdb()
        logger.warning("ALL REDIS DATA DELETED!")

        restored = 0
        for key, mapping in job_backup.items():
            redis_client.hset(key, mapping)
            # hset alone leaves the restored hash with no TTL - it would
            # otherwise live forever, unlike every other job hash in the
            # system (see app/jobstore.py's JOB_TTL_SECONDS). Missed in the
            # original version of this route; found during review.
            redis_client.expire(key, JOB_TTL_SECONDS)
            restored += 1

        logger.warning(f"NUCLEAR CLEANUP COMPLETE: Restored {restored} jobs")

        return jsonify({
            'message': 'Nuclear cleanup completed',
            'jobs_restored': restored,
            'warning': 'All processing history and results have been deleted',
        }), 200

    except Exception as e:
        logger.error(f"Nuclear cleanup failed: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/metadata/clear', methods=['POST'])
@require_admin_token
def clear_old_jobs():
    """Clear completed jobs older than N hours (default 24)."""
    try:
        data = request.get_json() or {}
        older_than_hours = data.get('older_than_hours', 24)
        cutoff_time = time.time() - (older_than_hours * 3600)
        cleared_jobs = []

        all_keys = []
        cursor = '0'
        while cursor != 0:
            cursor, keys = redis_client.client.scan(cursor=cursor, match='job:*', count=100)
            all_keys.extend(keys)
        job_ids = _job_ids_from_keys(all_keys)

        for jid in job_ids:
            try:
                job_data = get_job(jid)
                if not job_data:
                    continue

                if job_data['timestamp'] < cutoff_time:
                    redis_client.delete(f"job:{jid}", f"job:{jid}:tiles", f"job:{jid}:latencies")
                    cleared_jobs.append(jid[:8])
            except Exception as e:
                logger.error(f"Error clearing job {jid}: {e}")

        return jsonify({
            'message': f'Cleared {len(cleared_jobs)} old jobs',
            'cleared_jobs': cleared_jobs,
            'older_than_hours': older_than_hours,
        }), 200

    except Exception as e:
        logger.error(f"Error clearing jobs: {str(e)}")
        return jsonify({'error': str(e)}), 500

# /debug/redis-test used to live here: unauthenticated, did a blocking
# `KEYS '*'` (the exact command app/jobstore.py's own scan_job_ids()
# docstring says to never use against a live server), and returned raw key
# names to anyone who asked. Pure leftover debug scaffolding with no
# purpose /health doesn't already serve - removed rather than gated,
# during the same review that found the auth gap above.
