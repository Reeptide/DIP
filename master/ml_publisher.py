"""Publishes tile tasks to the ML inference topic.

Mirrors master/publisher.py's thread-pooled staging (JPEG encode + MinIO
PUT off the request thread) for the same reason: that work is blocking
network I/O and was the master's real throughput ceiling for the OpenCV
pipeline (see the plan's Step 6 addendum). No `operation` field is needed
here - there's exactly one thing an ML task means (run detection then
classification), unlike the OpenCV pipeline's 13 interchangeable filters.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.blobstore import put_tile
from app.config import ML_TASK_TOPIC, PUBLISH_POOL_SIZE
from app.imaging import encode_image
from app.jobstore import redis_client
from app.kafkaio import producer_pool

logger = logging.getLogger(__name__)


def _record_publish_failure(job_id):
    """See master/publisher.py's identical helper - this module had no
    equivalent at all (no delivery callback, no staging-failure tracking),
    so a failed ML tile publish was invisible in /detect/status and
    /metadata's failed_tasks. Found during a full-project review."""
    try:
        redis_client.hincrby(f"job:{job_id}", 'failed_tasks', 1)
    except Exception as e:
        logger.error(f"Failed to track ML publish error: {e}")


def _delivery_callback(err, msg, tile_id, job_id):
    if err is not None:
        logger.error(f"ML task delivery FAILED - Job: {job_id}, Tile: {tile_id}, Error: {err}")
        _record_publish_failure(job_id)


def _stage_tile(job_id, tile):
    return put_tile(job_id, tile['tile_id'], 'ml_task', encode_image(tile['tile_data']))


def publish_ml_tasks(job_id, tiles, batch_size):
    logger.info(f"Publishing {len(tiles)} ML tile tasks for job {job_id}")

    producer = producer_pool.get_producer()
    published_count = 0
    failed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=PUBLISH_POOL_SIZE) as executor:
            futures = {executor.submit(_stage_tile, job_id, tile): tile for tile in tiles}

            for future in as_completed(futures):
                tile = futures[future]
                try:
                    blob_key = future.result()

                    task = {
                        'job_id': job_id,
                        'tile_id': tile['tile_id'],
                        'blob_key': blob_key,
                        'x': tile['x'],
                        'y': tile['y'],
                        'width': tile['width'],
                        'height': tile['height'],
                        'timestamp': time.time()
                    }

                    producer.produce(
                        ML_TASK_TOPIC,
                        key=f"{job_id}:{tile['tile_id']}",
                        value=json.dumps(task).encode('utf-8'),
                        callback=lambda err, msg, tid=tile['tile_id']:
                            _delivery_callback(err, msg, tid, job_id)
                    )
                    producer.poll(0)
                    published_count += 1

                    if published_count % batch_size == 0:
                        producer.flush(timeout=1)

                except BufferError:
                    producer.flush(timeout=5)
                    failed_count += 1
                    _record_publish_failure(job_id)
                except Exception as e:
                    logger.error(f"Error publishing ML tile {tile['tile_id']}: {e}")
                    failed_count += 1
                    _record_publish_failure(job_id)

        producer.flush(timeout=10)
        logger.info(f"ML task publishing complete - Published: {published_count}, Failed: {failed_count}")
        return published_count > 0

    except Exception as e:
        logger.error(f"Critical error in publish_ml_tasks: {str(e)}")
        return False
