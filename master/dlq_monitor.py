"""Polls tasks.dlq's real depth and exposes it as a live Prometheus gauge.

dip_dlq_total (app/metrics.py, incremented in worker/main.py's send_to_dlq)
is a Counter - it only ever grows, and it's a per-process in-memory count
that resets on a worker restart and can't see what other worker replicas
sent. It answers "how many tiles have ever failed permanently," not "how
many are sitting there right now" - and this system has no consumer on
tasks.dlq in normal operation (it exists purely as an inspectable/replay
paper trail, see master/reaper.py's own docstring), so there's no other
code path that could answer the second question.

Kafka itself already knows the real number: the high watermark (next
offset to be written) minus the low watermark (oldest offset still
retained) on a partition is exactly its current message count, no separate
bookkeeping needed. Polling that via get_watermark_offsets() is standalone
- unlike a consumer, it does not join the DLQ's consumer group or affect
delivery to any future DLQ-draining tool that might be added later.
"""
import logging
import time

from confluent_kafka import Consumer, TopicPartition

from app.config import TASK_DLQ_TOPIC, configure_logging
from app.kafkaio import master_consumer_conf
from app.leader import is_leader
from app.metrics import dlq_depth

logger = logging.getLogger(__name__)

DLQ_POLL_INTERVAL_SECONDS = 5


def get_dlq_depth(consumer) -> int:
    metadata = consumer.list_topics(TASK_DLQ_TOPIC, timeout=10)
    partitions = metadata.topics[TASK_DLQ_TOPIC].partitions.keys()

    total = 0
    for p in partitions:
        low, high = consumer.get_watermark_offsets(
            TopicPartition(TASK_DLQ_TOPIC, p), timeout=10, cached=False,
        )
        total += high - low
    return total


def monitor_dlq_depth():
    """Background thread: keep dip_dlq_depth current forever.

    Runs on every master replica, gated on is_leader() - harmless either
    way (read-only Kafka admin calls, idempotent gauge set), but there's no
    reason for every replica to redundantly hit the broker's metadata API
    on the same schedule."""
    logger.info(f"DLQ depth monitor started (polling every {DLQ_POLL_INTERVAL_SECONDS}s)")

    consumer_config = master_consumer_conf()
    consumer_config['group.id'] = 'master-dlq-depth-monitor'
    consumer = Consumer(consumer_config)

    while True:
        if is_leader():
            try:
                depth = get_dlq_depth(consumer)
                dlq_depth.set(depth)
            except Exception as e:
                logger.error(f"Error polling DLQ depth: {e}", exc_info=True)
        time.sleep(DLQ_POLL_INTERVAL_SECONDS)


if __name__ == '__main__':
    configure_logging('dlq-monitor')
    monitor_dlq_depth()
