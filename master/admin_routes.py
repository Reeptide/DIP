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

from flask import Blueprint, jsonify, request

from app.jobstore import redis_client, get_job
from master.heartbeat import heartbeat_tracker

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)


def _job_ids_from_keys(all_keys):
    """job:{id} keys (not :tiles or :latencies suffixed) -> bare job ids."""
    return [
        key[4:] for key in all_keys
        if key.startswith('job:') and ':' not in key[4:]
    ]


@admin_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """List recent jobs (for debugging)."""
    try:
        return jsonify({
            'message': 'Job listing not implemented. Use /status/<job_id> to check specific jobs.',
            'active_workers': len(heartbeat_tracker.get_active_workers())
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/metadata', methods=['GET'])
def get_metadata():
    """Comprehensive metadata about jobs and Redis storage."""
    try:
        all_keys = []
        counter_keys = []
        orphaned_keys = []
        job_ids = set()
        job_keys = []
        tile_tracking_keys = []

        try:
            cursor = '0'
            while cursor != 0:
                cursor, keys = redis_client.client.scan(cursor=cursor, count=1000)
                all_keys.extend(keys)

            job_ids = set(_job_ids_from_keys(all_keys))
            job_keys = [f"job:{jid}" for jid in job_ids]
            # ':attempts' keys are the reaper's per-tile requeue counters
            # (Step 7) - job-scoped state, not orphans.
            tile_tracking_keys = [
                k for k in all_keys
                if k.endswith(':tiles') or k.endswith(':latencies') or k.endswith(':attempts')
            ]

            for key in all_keys:
                if key in job_keys or key in tile_tracking_keys:
                    continue
                elif key in ['active_workers_count', 'active_workers']:
                    counter_keys.append(key)
                else:
                    orphaned_keys.append(key)

        except Exception as e:
            logger.error(f"Error scanning Redis keys: {e}")

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


@admin_bp.route('/metadata/nuclear-cleanup', methods=['POST'])
def nuclear_cleanup():
    """Delete ALL non-essential Redis keys, keeping only job records."""
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
            restored += 1

        redis_client.set('active_workers_count', '0')
        redis_client.set('active_workers', '[]')

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


@admin_bp.route('/debug/redis-test', methods=['GET'])
def debug_redis_test():
    """Test Redis connectivity and basic operations."""
    from app.config import REDIS_HOST, REDIS_PORT
    try:
        ping_result = redis_client.ping()
        test_key = f"test_key_{int(time.time())}"
        redis_client.set(test_key, "test_value", ex=60)
        read_result = redis_client.get(test_key)
        db_size = redis_client.client.dbsize()
        all_keys = redis_client.client.keys('*')

        return jsonify({
            'redis_ping': ping_result,
            'test_write': test_key,
            'test_read': read_result,
            'database_size': db_size,
            'all_keys': all_keys[:20],
            'redis_config': {'host': REDIS_HOST, 'port': REDIS_PORT}
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'redis_config': {'host': REDIS_HOST, 'port': REDIS_PORT}}), 500
