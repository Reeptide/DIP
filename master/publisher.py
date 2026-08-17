"""Publishes tile-processing tasks to Kafka.

Partitioning: no explicit `partition=` is passed to produce(). Kafka's
default partitioner hashes the message key - `f"{job_id}:{tile_id}"`, not
bare tile_id (a bare tile_id would collide across different jobs' tiles
sharing the same id, e.g. every job's tile 0) - and spreads keys across
however many partitions the topic actually has. The original code
hardcoded `partition = tile_id % 2`, which silently capped usable
parallelism at 2 consumers no matter how many partitions the topic had or
how many workers were started - Kafka assigns whole partitions to
consumers in a group, so partition 2 onward would just sit unread. Letting
the default partitioner do this means the code needs no change if the
topic is ever repartitioned again.

Payload: the task message carries a MinIO blob_key, not the tile's image
bytes (the claim-check pattern, app/blobstore.py). Before this, each task
JSON embedded the tile as base64 - roughly 700KB per message - which is why
message.max.bytes had to be raised to 50MB (see app/kafkaio.py history).
Now each message is a couple hundred bytes.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.blobstore import put_tile
from app.config import TASK_TOPIC, PUBLISH_POOL_SIZE
from app.imaging import encode_image
from app.jobstore import redis_client
from app.kafkaio import producer_pool

logger = logging.getLogger(__name__)


def delivery_callback(err, msg, tile_id, job_id):
    if err is not None:
        logger.error(f"Task delivery FAILED - Job: {job_id}, Tile: {tile_id}, Error: {err}")
        try:
            # HINCRBY on the job:{job_id} hash's own 'failed_tasks' field -
            # NOT a separate `failed_tasks:{job_id}` string key. That used
            # to be a different key than the one app/jobstore.py's
            # get_job() reads ('failed_tasks' as a hash field), so nothing
            # ever wrote the field get_job() actually returns: /status and
            # /metadata's failed_tasks were hardcoded zeros in practice,
            # and the orphan string keys leaked with no TTL. Found during a
            # full-project review.
            redis_client.hincrby(f"job:{job_id}", 'failed_tasks', 1)
        except Exception as e:
            logger.error(f"Failed to track delivery error: {e}")
    else:
        logger.debug(f"Task delivered - Tile: {tile_id} -> Partition: {msg.partition()}")


def _record_publish_failure(job_id):
    """Mirror of delivery_callback()'s HINCRBY, for tiles that failed before
    ever reaching Kafka (blob staging or producer.produce() itself) rather
    than an async delivery error - see publish_tile_tasks()'s call sites."""
    try:
        redis_client.hincrby(f"job:{job_id}", 'failed_tasks', 1)
    except Exception as e:
        logger.error(f"Failed to track publish error: {e}")


def _stage_tile(job_id, tile):
    """Encode one tile and upload it to the blob store, returning its key."""
    return put_tile(job_id, tile['tile_id'], 'task', encode_image(tile['tile_data']))


def publish_tile_tasks(job_id, tiles, operation, batch_size):
    """Publish one Kafka task per tile, batching flushes for throughput.

    Blob staging (JPEG encode + MinIO PUT) runs across a thread pool because
    it is blocking network I/O and was the master's real bottleneck; Kafka
    produce/poll/flush stay on this thread. produce() is thread-safe, but
    keeping the poll/flush coordination single-threaded avoids interleaving
    delivery callbacks with the batch-flush accounting below.

    Tiles are published in completion order, not tile-id order - workers
    process tiles independently, so ordering carries no meaning here.
    """
    logger.info(f"Publishing {len(tiles)} tile tasks for job {job_id} (with batching)")

    producer = producer_pool.get_producer()
    published_count = 0
    failed_count = 0
    batch_count = 0

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
                        'operation': operation,
                        'blob_key': blob_key,
                        'x': tile['x'],
                        'y': tile['y'],
                        'width': tile['width'],
                        'height': tile['height'],
                        'timestamp': time.time()
                    }

                    producer.produce(
                        TASK_TOPIC,
                        # job_id prefix, not bare tile_id: tile_id restarts at 0
                        # for every job, so a bare tile_id key would send every
                        # job's tile 0 to the same partition - worse spread
                        # across concurrent jobs.
                        key=f"{job_id}:{tile['tile_id']}",
                        value=json.dumps(task).encode('utf-8'),
                        callback=lambda err, msg, tid=tile['tile_id']:
                            delivery_callback(err, msg, tid, job_id)
                    )

                    producer.poll(0)  # trigger delivery callbacks
                    published_count += 1

                    if published_count % batch_size == 0:
                        producer.flush(timeout=1)
                        batch_count += 1
                        logger.debug(f"Batch {batch_count} flushed ({batch_size} tasks)")

                except BufferError:
                    logger.warning(f"Producer queue full at tile {tile['tile_id']}, flushing...")
                    producer.flush(timeout=5)
                    failed_count += 1
                    _record_publish_failure(job_id)
                except Exception as e:
                    logger.error(f"Error publishing tile {tile['tile_id']}: {e}")
                    failed_count += 1
                    _record_publish_failure(job_id)

        producer.flush(timeout=10)

        logger.info(f"Task publishing complete - Published: {published_count}, Failed: {failed_count}, Batches: {batch_count}")
        # published_count > 0, not published_count == len(tiles): a tile
        # already staged and produced can't be un-dispatched, so returning
        # False on a partial failure here would tell /upload's caller
        # nothing happened when most tiles actually did - the reaper
        # recovers the rest either way. What WAS missing (found during a
        # full-project review): staging/produce failures here only updated
        # the local failed_count variable, never Redis - unlike an async
        # Kafka delivery failure (delivery_callback above), which already
        # HINCRBYs job:{id}'s failed_tasks field. A tile that failed to
        # even get produced was invisible in /status and /metadata.
        # _record_publish_failure() closes that gap.
        return published_count > 0

    except Exception as e:
        logger.error(f"Critical error in publish_tile_tasks: {str(e)}")
        return False
