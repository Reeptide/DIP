"""Kafka producer/consumer config, centralized so master and worker stop
maintaining separate copies.

Master and worker intentionally use different consumer settings and are
kept as separate functions rather than unified: the worker manually commits
offsets only after a tile is fully processed and its result published
(at-least-once delivery - required for the failure-recovery story in
Step 7). master_consumer_conf()'s enable.auto.commit=True default fits
master's genuinely best-effort monitoring streams (heartbeats, DLQ depth
polling) - but master/results_consumer.py and master/ml_results_consumer.py
override it back to manual commit, for the same at-least-once reason the
worker needs it: they're not just monitoring, they're the only place a
Kafka result becomes durable Redis state, and a leadership handoff
shouldn't be able to silently drop one mid-flight.
"""
import threading
import logging

from confluent_kafka import Producer

from app.config import KAFKA_BROKER

logger = logging.getLogger(__name__)


def master_producer_conf() -> dict:
    return {
        'bootstrap.servers': KAFKA_BROKER,
        'compression.type': 'gzip',
        'acks': 'all',
        # With enable.idempotence=True, librdkafka needs a large retry
        # budget to actually behave idempotently under transient failures -
        # a low retries count (this used to be 3) just makes the producer
        # give up and surface a send failure to the app well before the
        # idempotence machinery gets a chance to matter. delivery.timeout.ms
        # (default 120s) is the real bound on how long a send can be
        # retried, not this count, so it's safe to set this high.
        'retries': 2147483647,
        # No message.max.bytes override (Kafka's 1MB default is plenty):
        # messages carry a MinIO blob_key, not tile bytes, since Step 5.
        # This used to be set to 50MB to fit ~700KB base64 tile payloads.
        'request.timeout.ms': 30000,
        'linger.ms': 50,
        'batch.size': 500000,
        'enable.idempotence': True,
    }


def master_consumer_conf() -> dict:
    return {
        'bootstrap.servers': KAFKA_BROKER,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'session.timeout.ms': 30000,
        'max.poll.interval.ms': 300000,
    }


def worker_producer_conf() -> dict:
    return {
        'bootstrap.servers': KAFKA_BROKER,
        'compression.type': 'gzip',
        'acks': 'all',
        # With enable.idempotence=True, librdkafka needs a large retry
        # budget to actually behave idempotently under transient failures -
        # a low retries count (this used to be 3) just makes the producer
        # give up and surface a send failure to the app well before the
        # idempotence machinery gets a chance to matter. delivery.timeout.ms
        # (default 120s) is the real bound on how long a send can be
        # retried, not this count, so it's safe to set this high.
        'retries': 2147483647,
        # No message.max.bytes override - see master_producer_conf().
        'request.timeout.ms': 30000,
        'linger.ms': 100,
        'enable.idempotence': True,
    }


def worker_consumer_conf() -> dict:
    return {
        'bootstrap.servers': KAFKA_BROKER,
        # 'earliest', not 'latest': with 'latest' a worker that joins after
        # tasks were already published (e.g. scaling up mid-job, or a
        # container restart) would silently miss them - it only sees
        # messages produced after it connects.
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,  # manual commit for reliability
        'session.timeout.ms': 30000,
        'max.poll.interval.ms': 300000,
    }


class KafkaProducerPool:
    """Thread-safe singleton producer, shared across Flask request threads
    on the master (Confluent's Producer is itself thread-safe for produce();
    this just avoids constructing a new one per request)."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._producer = None
                    cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    try:
                        self._producer = Producer(master_producer_conf())
                        self._initialized = True
                        logger.info("Global Kafka producer initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize producer: {e}")
                        raise

    def get_producer(self):
        if not self._initialized:
            self.initialize()
        return self._producer

    def cleanup(self):
        if self._producer:
            logger.info("Flushing producer...")
            self._producer.flush(timeout=30)
            logger.info("Producer flushed")


producer_pool = KafkaProducerPool()
