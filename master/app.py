"""MASTER NODE - Flask web UI + image tiling + task dispatch + reconstruction.

Split out of the original single 1855-line master.py. Diagnostic/cleanup
routes live in master/admin_routes.py; heartbeat tracking in
master/heartbeat.py; task publishing in master/publisher.py; result
collection in master/results_consumer.py.

create_dashboard_template(), present in the original file, is dropped: it
regenerated templates/dashboard.html from an inline HTML string on every
master startup, silently overwriting the checked-in template file. The
template is tracked in git now and edited directly.
"""
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime

import cv2
from flask import Flask, render_template, request, jsonify, send_file, Response
from confluent_kafka import Producer
from werkzeug.utils import secure_filename

from app.blobstore import get_tile, delete_job_tiles
from app.config import (
    KAFKA_BROKER, REDIS_HOST, REDIS_PORT, INSTANCE_ID,
    UPLOAD_FOLDER, RESULT_FOLDER, ALLOWED_EXTENSIONS,
    TILE_SIZE, MIN_IMAGE_SIZE, MAX_IMAGE_DIMENSION, MAX_TILES,
    HEARTBEAT_TIMEOUT, BATCH_SIZE,
    configure_logging,
)
from app.imaging import split_image_into_tiles, reconstruct_image_from_tiles, probe_image_dimensions
from app.jobstore import (
    redis_client, create_job, get_job, mark_job_completed,
    get_tile_results, get_tile_latencies_ms, delete_job_tile_state,
)
from app.kafkaio import producer_pool
from app.metrics import jobs_created_total, jobs_finished_total, active_workers, is_leader_gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.ops import PROCESSING_OPERATIONS
from app.leader import run_leader_election, is_leader
from master.admin_routes import admin_bp
from master.dlq_monitor import monitor_dlq_depth
from master.heartbeat import heartbeat_tracker, monitor_heartbeats
from master.ml_results_consumer import listen_for_ml_results
from master.ml_routes import ml_bp
from master.publisher import publish_tile_tasks
from master.rate_limit import rate_limit
from master.reaper import run_reaper
from master.results_consumer import listen_for_results

configure_logging('MASTER')
logger = logging.getLogger(__name__)

# OpenCV otherwise fans every operation out across all host cores. In a
# horizontally-scaled deployment the parallelism comes from running more
# containers (`docker compose up --scale worker=N`), so a process that also
# threads internally just fights every other container for the same cores -
# measured as per-tile p50 latency climbing 20.8ms -> 28.6ms from 1 to 12
# workers, which is contention, not saturation.
cv2.setNumThreads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Absolute, not the bare 'uploads'/'results' from app.config: Flask's
# send_file() resolves relative paths against app.root_path (the directory
# containing this module, /srv/master), not the process cwd (/srv). That
# mismatch only appeared once master.py moved into master/app.py - a
# relative RESULT_FOLDER worked fine when this file sat at the repo root.
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, UPLOAD_FOLDER)
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, RESULT_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.register_blueprint(admin_bp)
app.register_blueprint(ml_bp)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Rate limiting: see master/rate_limit.py (its own module so
# master/ml_routes.py can use the same decorator without a circular
# import - master/app.py already imports ml_bp FROM ml_routes.py).


# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html', operations=PROCESSING_OPERATIONS)


@app.route('/upload', methods=['POST'])
@rate_limit
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    operation = request.form.get('operation', 'grayscale')

    if operation not in PROCESSING_OPERATIONS:
        return jsonify({'error': f'Invalid operation. Valid options: {list(PROCESSING_OPERATIONS.keys())}'}), 400

    if not file or file.filename == '':
        return jsonify({'error': 'Invalid file'}), 400

    if '.' not in file.filename:
        return jsonify({'error': 'File must have an extension'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    try:
        active_workers = heartbeat_tracker.get_active_workers()
        if len(active_workers) == 0:
            return jsonify({'error': 'No active workers available. Please start workers first.'}), 503

        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")

        file.save(filepath)

        # Probe dimensions from the header BEFORE cv2.imread() decodes
        # pixels - a small, highly-compressed file can claim a huge
        # resolution, and imread() would allocate the full pixel buffer
        # (potentially multiple GB) before the MAX_IMAGE_DIMENSION check
        # below ever ran. probe_image_dimensions() returns None on an
        # unrecognized/malformed header, in which case imread() below is
        # left to reject the file the way it always did.
        with open(filepath, 'rb') as f:
            header = f.read(65536)
        probed = probe_image_dimensions(header)
        if probed and (probed[0] > MAX_IMAGE_DIMENSION or probed[1] > MAX_IMAGE_DIMENSION):
            os.remove(filepath)
            return jsonify({
                'error': f'Image too large. Maximum dimension: {MAX_IMAGE_DIMENSION}px'
            }), 400

        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to load image or corrupted file'}), 400

        height, width = image.shape[:2]

        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            os.remove(filepath)
            return jsonify({
                'error': f'Image too small. Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}, got: {width}x{height}'
            }), 400

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            os.remove(filepath)
            return jsonify({
                'error': f'Image too large. Maximum dimension: {MAX_IMAGE_DIMENSION}px'
            }), 400

        try:
            tiles = split_image_into_tiles(image, TILE_SIZE)
        except ValueError as e:
            os.remove(filepath)
            return jsonify({'error': str(e)}), 400

        expected_tiles = len(tiles)

        # image is already decoded into memory (line 129) and tiled from
        # there - the on-disk upload is never read again on this path.
        os.remove(filepath)

        create_job(
            job_id,
            original_filename=filename,
            operation=operation,
            original_width=width,
            original_height=height,
            expected_tiles=expected_tiles,
            active_workers=len(active_workers),
            tile_size=TILE_SIZE,
        )

        success = publish_tile_tasks(job_id, tiles, operation, BATCH_SIZE)

        if not success:
            return jsonify({'error': 'Failed to publish tasks to workers'}), 500

        jobs_created_total.labels(pipeline='opencv').inc()
        logger.info(f"Job {job_id[:8]} created: {expected_tiles} tiles")

        return jsonify({
            'message': 'Image uploaded and processing started',
            'job_id': job_id,
            'tiles_count': expected_tiles,
            'operation': operation,
            'active_workers': len(active_workers),
            'image_size': f"{width}x{height}"
        }), 200

    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    try:
        job_data = get_job(job_id)
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        results_count = job_data['results_count']
        expected_tiles = job_data['expected_tiles'] or 1
        progress = int((results_count / expected_tiles) * 100)

        return jsonify({
            'job_id': job_id,
            'status': job_data['status'],
            'progress': progress,
            'received_tiles': results_count,
            'expected_tiles': expected_tiles,
            'failed_tasks': job_data['failed_tasks'],
            'tile_latencies_s': [ms / 1000 for ms in get_tile_latencies_ms(job_id)],
            'job_data': job_data
        }), 200

    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    try:
        job_data = get_job(job_id)
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        if job_data['status'] != 'completed':
            return jsonify({'error': f'Job not completed yet. Current status: {job_data["status"]}'}), 400

        result_path = job_data['result_path']

        if not result_path or not os.path.exists(result_path):
            return jsonify({'error': 'Result file not found'}), 404

        return send_file(result_path, mimetype='image/jpeg')

    except Exception as e:
        logger.error(f"Error retrieving result: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/reconstruct/<job_id>', methods=['POST'])
def reconstruct_job(job_id):
    try:
        job_data = get_job(job_id)
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        received_tiles = get_tile_results(job_id)
        expected_count = job_data['expected_tiles']

        if len(received_tiles) < expected_count:
            return jsonify({
                'error': f'Not all tiles received. Expected: {expected_count}, Got: {len(received_tiles)}'
            }), 400

        # Fetch each tile's processed bytes from MinIO by key - Redis only
        # ever held the pointer (see master/results_consumer.py).
        tiles_for_reconstruction = [
            {
                'tile_id': tile_id,
                'processed_tile': get_tile(tile_info['result_blob_key']),
                'x': tile_info['x'],
                'y': tile_info['y'],
                'width': tile_info['width'],
                'height': tile_info['height']
            }
            for tile_id, tile_info in received_tiles.items()
        ]
        tiles_for_reconstruction.sort(key=lambda t: t['tile_id'])

        final_image = reconstruct_image_from_tiles(
            tiles_for_reconstruction,
            job_data['original_width'],
            job_data['original_height']
        )

        result_filename = f"{job_id}_processed.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, final_image)

        mark_job_completed(job_id, result_path)
        jobs_finished_total.labels(pipeline='opencv', status='completed').inc()

        delete_job_tile_state(job_id)
        # Also sweep the MinIO blobs (input + result tiles) for this job -
        # otherwise every job leaves its tile bytes in the bucket forever.
        delete_job_tiles(job_id)

        logger.info(f"Job {job_id[:8]} reconstruction completed and tile data cleaned")

        return jsonify({
            'message': 'Image reconstructed successfully',
            'job_id': job_id,
            'result_path': result_path,
            'tiles_processed': len(tiles_for_reconstruction)
        }), 200

    except Exception as e:
        logger.error(f"Error reconstructing image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')


@app.route('/dashboard/data', methods=['GET'])
def dashboard_data():
    try:
        active_workers = heartbeat_tracker.get_active_workers()
        all_heartbeats = heartbeat_tracker.get_all_heartbeats()

        worker_details = []
        current_time = time.time()

        for wid in active_workers:
            hb = all_heartbeats.get(wid, {})
            last_seen = hb.get('last_seen', 0)
            last_seen_ago = current_time - last_seen

            worker_details.append({
                'worker_id': wid,
                'last_seen': last_seen,
                'last_seen_ago': last_seen_ago,
                'status': hb.get('status', 'unknown'),
                'heartbeat_timestamp': datetime.fromtimestamp(last_seen).strftime('%H:%M:%S') if last_seen > 0 else 'Never',
                'is_healthy': last_seen_ago < HEARTBEAT_TIMEOUT,
                'response_time_ms': round(last_seen_ago * 1000, 0)
            })

        worker_details.sort(key=lambda w: w['last_seen'], reverse=True)

        total_workers = len(worker_details)
        healthy_workers = len([w for w in worker_details if w['is_healthy']])
        health_percentage = (healthy_workers / total_workers * 100) if total_workers > 0 else 100

        return jsonify({
            'active_workers': len(active_workers),
            'worker_list': worker_details,
            'timestamp': current_time,
            'kafka_broker': KAFKA_BROKER,
            'redis_connected': redis_client.ping(),
            'operations_available': len(PROCESSING_OPERATIONS),
            'heartbeat_stats': {
                'total_workers': total_workers,
                'healthy_workers': healthy_workers,
                'health_percentage': health_percentage,
                'heartbeat_timeout': HEARTBEAT_TIMEOUT,
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
        }), 200

    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    active_workers.set(len(heartbeat_tracker.get_active_workers()))
    is_leader_gauge.set(1 if is_leader() else 0)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health', methods=['GET'])
def health():
    try:
        active_workers = heartbeat_tracker.get_active_workers()
        redis_healthy = redis_client.ping()

        # producer_pool.get_producer() alone never proves anything -
        # confluent_kafka.Producer() doesn't connect eagerly, so this
        # returned a truthy object (and reported "healthy") even against a
        # broker that was never reachable at all. list_topics() actually
        # talks to the broker and raises on failure - a real check, found
        # missing during a full-project review.
        kafka_healthy = False
        try:
            producer_pool.get_producer().list_topics(timeout=5)
            kafka_healthy = True
        except Exception as e:
            logger.warning(f"Kafka health check failed: {e}")

        overall_status = 'healthy' if (redis_healthy and kafka_healthy) else 'degraded'

        return jsonify({
            'status': overall_status,
            'kafka_broker': KAFKA_BROKER,
            'kafka_healthy': kafka_healthy,
            'redis_host': REDIS_HOST,
            'redis_healthy': redis_healthy,
            'active_workers': len(active_workers),
            'operations': len(PROCESSING_OPERATIONS),
            'instance_id': INSTANCE_ID,
            'is_leader': is_leader(),
            'timestamp': time.time()
        }), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    try:
        producer_pool.cleanup()
    except Exception as e:
        logger.error(f"Error during producer cleanup: {e}")
    logger.info("Shutdown complete")
    os._exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config():
    logger.info("Validating configuration...")
    errors = []

    if not KAFKA_BROKER:
        errors.append("KAFKA_BROKER not configured")
    else:
        try:
            # Producer() doesn't connect eagerly and flush() on an empty
            # queue returns immediately regardless of broker reachability -
            # this "test" always passed, even against an unreachable
            # broker, until list_topics() (which actually round-trips to
            # the broker) replaced it. Found during a full-project review.
            test_producer = Producer({'bootstrap.servers': KAFKA_BROKER})
            test_producer.list_topics(timeout=5)
            logger.info("Kafka connectivity test passed")
        except Exception as e:
            errors.append(f"Cannot connect to Kafka at {KAFKA_BROKER}: {str(e)}")

    if not REDIS_HOST:
        errors.append("REDIS_HOST not configured")
    else:
        try:
            if redis_client.ping():
                logger.info("Redis connectivity test passed")
            else:
                errors.append("Redis ping failed")
        except Exception as e:
            errors.append(f"Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}: {str(e)}")

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
        logger.info("Upload and result directories ready")
    except Exception as e:
        errors.append(f"Cannot create directories: {str(e)}")

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("All configuration checks passed")
    return True


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    logger.info("=" * 80)
    logger.info("MASTER NODE - Distributed Image Processing Pipeline")
    logger.info(f"Instance ID: {INSTANCE_ID}")
    logger.info(f"Operations: {len(PROCESSING_OPERATIONS)}")
    logger.info(f"Kafka Broker: {KAFKA_BROKER}")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Tile Size: {TILE_SIZE}x{TILE_SIZE}")
    logger.info(f"Max Tiles: {MAX_TILES}")
    logger.info(f"Batching: Enabled (batch size: {BATCH_SIZE})")
    logger.info("=" * 80)

    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        raise SystemExit(1)

    try:
        producer_pool.initialize()
        logger.info("Producer pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize producer pool: {e}")
        raise SystemExit(1)

    logger.info("Starting background threads...")

    # Leader election first - the singleton-duty threads below (results
    # consumers, reaper) gate their own work on is_leader() internally, but
    # they still start unconditionally on every replica; only the elected
    # leader's copies actually do anything. See app/leader.py.
    threading.Thread(target=run_leader_election, daemon=False, name="LeaderElection").start()
    logger.info("Leader election thread started")

    threading.Thread(target=listen_for_results, daemon=False, name="ResultListener").start()
    logger.info("Result listener thread started")

    threading.Thread(target=listen_for_ml_results, daemon=False, name="MLResultListener").start()
    logger.info("ML result listener thread started")

    threading.Thread(target=monitor_heartbeats, daemon=False, name="HeartbeatMonitor").start()
    logger.info("Heartbeat monitor thread started")

    threading.Thread(target=run_reaper, daemon=False, name="Reaper").start()
    logger.info("Reaper thread started")

    threading.Thread(target=monitor_dlq_depth, daemon=False, name="DLQDepthMonitor").start()
    logger.info("DLQ depth monitor thread started")

    logger.info("=" * 80)
    logger.info("Master Node is ready! Starting Flask web server on 0.0.0.0:5000")
    logger.info("=" * 80)

    try:
        # This IS what actually runs in the shipped image, not a
        # placeholder - `gunicorn` sat unused in requirements.txt claiming
        # otherwise (found during a full-project review; removed it there
        # rather than switch WSGI servers here, which isn't a drop-in
        # change: gunicorn's default worker model forks separate OS
        # processes, and this module starts background threads - the
        # reaper, leader election, result consumers - at import time,
        # before app.run() - each forked worker would independently
        # restart all of them, multiplying exactly the kind of
        # concurrent-duty problem app/leader.py's election exists to
        # prevent, just within one container instead of across replicas).
        # debug=False is what keeps this reasonably production-safe as-is;
        # threaded=True lets it serve overlapping requests without a
        # separate WSGI layer.
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")


if __name__ == '__main__':
    main()
