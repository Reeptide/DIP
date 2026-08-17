"""Consumes 'ml_results' and stores per-tile detections in Redis.

Reuses app.jobstore's generic tile-result storage (job:{id}:tiles as an
atomic Redis hash, results_count via HINCRBY) exactly as-is - it was never
specific to the OpenCV pipeline's schema, it just stores whatever dict a
tile produced. Here that's {'detections': [...], 'worker_id', 'x', 'y',
'width', 'height'} instead of {'result_blob_key'}.

Leader-gated with a shared consumer group, same reasoning as
master/results_consumer.py - see that module's docstring.
"""
import json
import logging
import time

from confluent_kafka import Consumer, KafkaError

from app.blobstore import delete_job_tiles
from app.config import ML_RESULT_TOPIC
from app.kafkaio import master_consumer_conf
from app.jobstore import get_job, record_tile_result, redis_client
from app.leader import is_leader
from app.metrics import jobs_finished_total

logger = logging.getLogger(__name__)

ML_RESULT_CONSUMER_GROUP = 'master-ml-result-consumer'


def _process_message(msg):
    result = json.loads(msg.value().decode('utf-8'))
    job_id = result['job_id']
    tile_id = result['tile_id']

    tile_info = {
        'detections': result.get('detections', []),
        'worker_id': result['worker_id'],
        'x': result.get('x', 0),
        'y': result.get('y', 0),
        'width': result.get('width', 0),
        'height': result.get('height', 0),
    }
    processing_time = result.get('processing_time', 0)

    received_count = record_tile_result(job_id, tile_id, tile_info, processing_time)

    job_data = get_job(job_id)
    # A degraded job gave up on tiles it will never receive; don't
    # resurrect it if a straggler from a requeue lands afterwards. This
    # guard exists in master/results_consumer.py (the OpenCV sibling) but
    # was missing here - a review found it, and the missing check meant a
    # degraded ML job could flip back to 'completed' on a late duplicate,
    # and every such duplicate on an already-finished job would re-fire
    # jobs_finished_total{status="completed"}.inc() with no bound.
    if job_data and job_data['status'] == 'processing' and received_count >= job_data['expected_tiles']:
        redis_client.hset(f"job:{job_id}", {'status': 'completed'})
        jobs_finished_total.labels(pipeline='ml', status='completed').inc()
        logger.info(f"ML job {job_id[:8]} complete - all {received_count} tiles received")

        # Only the MinIO input blobs (the raw uploaded tile images
        # inference actually reads), not the Redis tile state - unlike the
        # OpenCV pipeline, an ML job's "final output" IS its Redis
        # job:{id}:tiles hash (detections are pure JSON, there's no
        # separate reconstructed-image artifact the way /reconstruct
        # produces one), and /detect/status keeps reading that hash for as
        # long as a client keeps polling it. Deleting it here the way
        # master/app.py's /reconstruct deletes OpenCV's would make every
        # subsequent status poll silently lose the detection results.
        # Found unswept entirely during a full-project review - MinIO
        # blobs for /detect uploads leaked forever before this.
        delete_job_tiles(job_id)


def listen_for_ml_results():
    logger.info("ML result listener started (leader-gated)")

    while True:
        if not is_leader():
            time.sleep(1)
            continue

        logger.warning("Acquired leadership - starting ML result consumption")
        consumer_config = master_consumer_conf()
        consumer_config['group.id'] = ML_RESULT_CONSUMER_GROUP
        # Manual commit - see master/results_consumer.py's identical fix
        # for the full reasoning (auto-commit can drop a fetched-but-not-
        # yet-processed result silently on a leadership handoff).
        consumer_config['enable.auto.commit'] = False
        consumer = Consumer(consumer_config)
        consumer.subscribe([ML_RESULT_TOPIC])

        try:
            while is_leader():
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"ML result consumer error: {msg.error()}")
                    continue

                try:
                    _process_message(msg)
                    consumer.commit(message=msg, asynchronous=True)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in ML result: {e}")
                except Exception as e:
                    logger.error(f"Error processing ML result: {str(e)}")
        finally:
            consumer.close()
            logger.warning("Lost leadership - stopped ML result consumption")
