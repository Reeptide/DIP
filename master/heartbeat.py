"""Worker heartbeat tracking: consumes the 'heartbeats' topic and exposes
which workers are currently alive.

Backed by Redis (`heartbeat:{worker_id}` keys with a TTL), not an in-process
dict - master is horizontally scalable now (`docker compose up --scale
master=N`, see app/leader.py), and an in-memory tracker would mean a freshly
started or newly-elected-leader replica sees zero active workers until it
personally observes fresh heartbeats, several seconds later - exactly the
gap that caused a real 503 in this project's own master-failure-recovery
testing. Redis makes worker liveness a fact any replica can see the instant
ANY replica (or a previous instance of the same one) last observed it.

Every master replica runs monitor_heartbeats() unconditionally, not gated by
leadership - it's not a singleton duty. Each replica has its own consumer
group ID (INSTANCE_ID-derived), so Kafka fans the heartbeats topic out to
every replica independently (this is desired: every replica should have its
own live view, not just the leader), and the Redis writes are naturally
idempotent (SET with an expiry), so redundant writes from N replicas
observing the same heartbeat cost nothing beyond the write itself.

heartbeat_checker() (a periodic in-memory-to-Redis mirror thread) is gone -
there's no in-memory state left to mirror, the tracker IS Redis now.
"""
import json
import logging

from confluent_kafka import Consumer, KafkaError

from app.config import INSTANCE_ID, HEARTBEAT_TOPIC, HEARTBEAT_TIMEOUT
from app.kafkaio import master_consumer_conf
from app.jobstore import redis_client

logger = logging.getLogger(__name__)

HEARTBEAT_KEY_PREFIX = 'heartbeat:'


class WorkerHeartbeatTracker:
    """Redis-backed - see module docstring. Kept as a class (rather than
    bare module functions) only so master/app.py's existing
    `heartbeat_tracker.get_active_workers()` call sites don't need to
    change."""

    def update(self, worker_id, timestamp):
        key = f"{HEARTBEAT_KEY_PREFIX}{worker_id}"
        value = json.dumps({'last_seen': timestamp, 'status': 'alive'})
        # px, not ex: HEARTBEAT_TIMEOUT is seconds-granularity but short
        # (15s default) - ms precision avoids a worker looking dead for up
        # to a full extra second after a genuine timeout.
        redis_client.client.set(key, value, px=int(HEARTBEAT_TIMEOUT * 1000))

    def get_active_workers(self, timeout=HEARTBEAT_TIMEOUT):
        """Redis's own TTL expiry IS the liveness check now - any key
        present is active by definition, no manual timestamp comparison or
        explicit dead-worker sweep needed (both were real code in the old
        in-memory version; Redis does it for free and can't race with a
        concurrent update the way a manual sweep-then-delete could)."""
        keys = redis_client.client.keys(f"{HEARTBEAT_KEY_PREFIX}*")
        return [k[len(HEARTBEAT_KEY_PREFIX):] for k in keys]

    def get_all_heartbeats(self):
        keys = redis_client.client.keys(f"{HEARTBEAT_KEY_PREFIX}*")
        if not keys:
            return {}
        values = redis_client.client.mget(keys)
        result = {}
        for key, value in zip(keys, values):
            if value is None:  # expired between KEYS and MGET - ignore
                continue
            worker_id = key[len(HEARTBEAT_KEY_PREFIX):]
            result[worker_id] = json.loads(value)
        return result


heartbeat_tracker = WorkerHeartbeatTracker()


def monitor_heartbeats():
    """Background thread: consume the heartbeats topic and feed the tracker.
    Runs on every replica unconditionally - see module docstring."""
    logger.info("Heartbeat monitor started")

    consumer_config = master_consumer_conf()
    consumer_config['group.id'] = f'master-heartbeat-monitor-{INSTANCE_ID}'

    consumer = Consumer(consumer_config)

    try:
        consumer.subscribe([HEARTBEAT_TOPIC])
        logger.info(f"Subscribed to heartbeat topic: {HEARTBEAT_TOPIC}")
    except Exception as e:
        logger.error(f"Failed to subscribe to heartbeat topic: {e}")
        return

    message_count = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Heartbeat consumer error: {msg.error()}")
                continue

            message_count += 1

            try:
                heartbeat = json.loads(msg.value().decode('utf-8'))
                worker_id = heartbeat.get('worker_id')
                timestamp = heartbeat.get('timestamp')

                if worker_id and timestamp:
                    heartbeat_tracker.update(worker_id, timestamp)
                    logger.debug(f"Heartbeat from {worker_id} at {timestamp} (msg #{message_count})")
                else:
                    logger.warning(f"Invalid heartbeat data: worker_id={worker_id}, timestamp={timestamp}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in heartbeat: {e}")
            except Exception as e:
                logger.error(f"Error processing heartbeat: {str(e)}")

    except KeyboardInterrupt:
        logger.info("Heartbeat monitor stopped")
    finally:
        consumer.close()
        logger.info(f"Heartbeat monitor closed. Total messages processed: {message_count}")
