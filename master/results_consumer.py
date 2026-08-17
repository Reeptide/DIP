"""Consumes the 'results' topic and stores tile results in Redis.

Rewritten in Step 6: each result is now one atomic HSET (tile) + RPUSH
(latency) + HINCRBY (results_count) - no more GET-the-whole-blob,
mutate-in-Python, SETEX-it-back per tile. See app/jobstore.py for the
schema and why the old approach was the real bottleneck (confirmed by the
Step 4 benchmark: throughput was flat regardless of worker count, because
this single-threaded consumer, not tile processing, was the ceiling).

Leader-gated (app/leader.py): only the elected master replica actually
consumes. The consumer group is now a FIXED name shared across all master
replicas, not INSTANCE_ID-suffixed - this matters for failover
correctness, not just efficiency. A per-instance group would mean a newly
elected leader starts with zero committed offsets for its own group
(`auto.offset.reset: earliest`), replaying and re-processing the ENTIRE
results topic history from scratch on every failover. A shared group means
whichever replica is leader continues from exactly where the previous
leader's last commit left off.
"""
import json
import logging
import time

from confluent_kafka import Consumer, KafkaError

from app.config import RESULT_TOPIC
from app.kafkaio import master_consumer_conf
from app.jobstore import get_job, record_tile_result, redis_client
from app.leader import is_leader

logger = logging.getLogger(__name__)

RESULT_CONSUMER_GROUP = 'master-result-consumer'


def _process_message(msg):
    result = json.loads(msg.value().decode('utf-8'))
    job_id = result['job_id']
    tile_id = result['tile_id']

    logger.info(f"Processing result for job {job_id[:8]}, tile {tile_id}")

    tile_info = {
        'result_blob_key': result['result_blob_key'],
        'worker_id': result['worker_id'],
        'x': result.get('x', 0),
        'y': result.get('y', 0),
        'width': result.get('width', 0),
        'height': result.get('height', 0),
    }
    processing_time = result.get('processing_time', 0)

    # Distinct tile count, not a message count - see
    # app.jobstore.record_tile_result. Duplicates are expected
    # (at-least-once delivery + reaper requeues) and must not be
    # able to push this past the number of tiles actually held.
    distinct_count = record_tile_result(job_id, tile_id, tile_info, processing_time)

    job_data = get_job(job_id)
    # A degraded job gave up on tiles it will never receive; don't
    # resurrect it if a straggler from a requeue lands afterwards.
    if job_data and job_data['status'] == 'processing' and distinct_count >= job_data['expected_tiles']:
        redis_client.hset(f"job:{job_id}", {'status': 'ready_for_reconstruction'})
        logger.info(f"Job {job_id[:8]} complete - all {distinct_count} tiles received")

    logger.info(f"Result stored - Job: {job_id[:8]}, Tile: {tile_id}, Distinct tiles: {distinct_count}")


def listen_for_results():
    """Background thread: listen for processed tile results.

    Runs on every replica, but only constructs/holds a live Consumer while
    leader - see module docstring. The outer loop waits for leadership; the
    inner loop consumes until leadership is lost, then closes the consumer
    cleanly and goes back to waiting."""
    logger.info("Result listener started (leader-gated)")

    while True:
        if not is_leader():
            time.sleep(1)
            continue

        logger.warning("Acquired leadership - starting result consumption")
        consumer_config = master_consumer_conf()
        consumer_config['group.id'] = RESULT_CONSUMER_GROUP
        # Manual commit, overriding master_consumer_conf()'s default
        # enable.auto.commit=True (fine for the monitoring-only consumers
        # it was designed for - heartbeats, DLQ depth - but wrong here).
        # Auto-commit advances offsets on a timer, independent of whether
        # _process_message() actually succeeded - so a result that was
        # fetched but not yet processed could already be committed when a
        # leadership handoff happens, silently dropping it rather than
        # replaying it on the new leader. That directly contradicted this
        # module's own docstring claim about clean failover continuation;
        # found during a full-project review. record_tile_result() is
        # idempotent, so committing only after a successful process is
        # safe even under at-least-once redelivery from a slow/failed
        # commit.
        consumer_config['enable.auto.commit'] = False
        consumer = Consumer(consumer_config)
        consumer.subscribe([RESULT_TOPIC])

        try:
            while is_leader():
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Result consumer error: {msg.error()}")
                    continue

                try:
                    _process_message(msg)
                    # Async, not sync - see inference/main.py's
                    # _drain_commits for why: a synchronous commit blocks
                    # until the broker acks, and if that takes long enough
                    # this loop stops calling poll() for the same stretch,
                    # risking a session-timeout eviction from the consumer
                    # group (silent - no crash, just a new member ID).
                    consumer.commit(message=msg, asynchronous=True)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in result: {e}")
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
        finally:
            consumer.close()
            logger.warning("Lost leadership - stopped result consumption")
