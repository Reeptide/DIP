"""WORKER NODE - Image Processing Worker

Consumes image tiles from Kafka 'tasks', applies one of 13 OpenCV
operations, publishes results to 'results', and sends periodic heartbeats.

This file replaces worker1.py and worker2.py, which were identical except
for whitespace - two files standing in for what should have been one
process run twice. Identity now comes entirely from the WORKER_ID env var
(docker-compose sets a distinct one per replica); run N of these via
`docker compose up --scale worker=N`.
"""
import json
import logging
import signal
import time
from threading import Thread, Event

import cv2
from confluent_kafka import Producer, Consumer, KafkaError

from app.blobstore import get_tile, put_tile
from app.config import (
    KAFKA_BROKER, WORKER_ID, HEARTBEAT_INTERVAL, HEARTBEAT_TOPIC,
    TASK_TOPIC, TASK_DLQ_TOPIC, RESULT_TOPIC, configure_logging,
)
from app.kafkaio import worker_producer_conf, worker_consumer_conf
from app.imaging import decode_image, encode_image
from app.metrics import (
    tiles_processed_total, tile_processing_seconds, dlq_total, serve_metrics,
)
from app.ops import OPERATION_MAP

configure_logging(WORKER_ID)
logger = logging.getLogger(__name__)

# One OpenCV thread per worker process: scale-out comes from running more
# worker containers, not from each one spreading a single tile across every
# host core. At 12 replicas the default (host core count) meant ~192 OpenCV
# threads contending for 16 real cores, which is why per-tile p50 latency
# rose as replicas were added instead of holding flat.
cv2.setNumThreads(1)


class HeartbeatSender:
    """Sends periodic heartbeat messages to Kafka."""

    def __init__(self, worker_id, interval=HEARTBEAT_INTERVAL):
        self.worker_id = worker_id
        self.interval = interval
        self.running = False
        self.producer = None
        self.stop_event = Event()

    def initialize(self):
        try:
            self.producer = Producer(worker_producer_conf())
            logger.info("Heartbeat producer initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize heartbeat producer: {str(e)}")
            return False

    def delivery_report(self, err, msg):
        if err is not None:
            logger.debug(f"Heartbeat delivery failed: {err}")
        else:
            logger.debug(f"Heartbeat delivered to partition {msg.partition()}")

    def send_heartbeat(self):
        try:
            heartbeat_message = {
                'worker_id': self.worker_id,
                'timestamp': int(time.time()),
                'status': 'alive'
            }
            self.producer.produce(
                HEARTBEAT_TOPIC,
                key=self.worker_id,
                value=json.dumps(heartbeat_message).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)  # trigger delivery callbacks
            logger.debug(f"Heartbeat sent at {heartbeat_message['timestamp']}")
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {str(e)}")

    def run(self):
        logger.info(f"Heartbeat sender thread started (interval: {self.interval}s)")
        self.running = True
        while self.running and not self.stop_event.is_set():
            try:
                self.send_heartbeat()
                self.stop_event.wait(timeout=self.interval)
            except Exception as e:
                logger.error(f"Error in heartbeat sender: {str(e)}")
                self.stop_event.wait(timeout=self.interval)

    def stop(self):
        logger.info("Stopping heartbeat sender...")
        self.running = False
        self.stop_event.set()
        if self.producer:
            self.producer.flush(timeout=5)
        logger.info("Heartbeat sender stopped")


class Worker:
    """Consumes tasks, processes tiles, publishes results."""

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.consumer = None
        self.task_producer = None
        self.heartbeat_sender = None
        self.running = False
        self.shutdown_event = Event()

        self.metrics = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0,
            'start_time': time.time()
        }

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.running = False

    def initialize_kafka(self):
        try:
            consumer_config = worker_consumer_conf()
            consumer_config['group.id'] = 'image-processing-workers'

            self.consumer = Consumer(consumer_config)
            self.consumer.subscribe([TASK_TOPIC])

            self.task_producer = Producer(worker_producer_conf())

            logger.info(f"Worker {self.worker_id} connected to Kafka broker: {KAFKA_BROKER}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {str(e)}")
            return False

    def process_tile(self, task, max_retries=2):
        """Process a single image tile. Returns a result dict, or None if
        every retry failed."""
        job_id = task['job_id']
        tile_id = task['tile_id']
        operation = task['operation']
        blob_key = task['blob_key']

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                logger.info(f"Processing tile {tile_id} for job {job_id} with operation '{operation}' (attempt {attempt + 1})")

                if operation not in OPERATION_MAP:
                    logger.error(f"Unknown operation: {operation}")
                    self.metrics['tasks_failed'] += 1
                    return None

                tile_image = decode_image(get_tile(blob_key))
                processed_tile = OPERATION_MAP[operation](tile_image)
                result_blob_key = put_tile(job_id, tile_id, 'result', encode_image(processed_tile))
                processing_time = time.time() - start_time

                result = {
                    'job_id': job_id,
                    'tile_id': tile_id,
                    'operation': operation,
                    'result_blob_key': result_blob_key,
                    'worker_id': self.worker_id,
                    'processing_time': processing_time,
                    'x': task['x'],
                    'y': task['y'],
                    'width': task['width'],
                    'height': task['height'],
                    'timestamp': time.time()
                }

                self.metrics['tasks_processed'] += 1
                self.metrics['total_processing_time'] += processing_time
                tiles_processed_total.labels(pipeline='opencv', status='success').inc()
                tile_processing_seconds.labels(pipeline='opencv').observe(processing_time)

                logger.info(f"Tile {tile_id} processed successfully in {processing_time:.2f}s")
                return result

            except Exception as e:
                logger.error(f"Error processing tile {tile_id} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))  # exponential backoff
                else:
                    self.metrics['tasks_failed'] += 1
                    tiles_processed_total.labels(pipeline='opencv', status='failed').inc()
                    return None

    def publish_result(self, result):
        """Publish tile result to Kafka.

        No explicit partition= - see master/publisher.py for why (Kafka's
        default partitioner hashes the key across all available partitions;
        hardcoding tile_id % 2 was the parallelism ceiling).
        """
        try:
            job_id = result['job_id']
            tile_id = result['tile_id']

            self.task_producer.produce(
                RESULT_TOPIC,
                key=f"{job_id}:{tile_id}",  # see master/publisher.py for why
                value=json.dumps(result).encode('utf-8'),
                callback=lambda err, msg: self._result_callback(err, msg, tile_id)
            )
            self.task_producer.poll(0)
            return True

        except BufferError:
            logger.warning(f"Producer queue full for tile {result['tile_id']}, flushing...")
            self.task_producer.flush(timeout=5)
            return False
        except Exception as e:
            logger.error(f"Failed to publish result for tile {result['tile_id']}: {str(e)}")
            return False

    def send_to_dlq(self, task, reason):
        """Park a task that exhausted its retries on tasks.dlq.

        Before this, a tile that failed every retry had its offset committed
        with nothing but a warning - the tile was gone, and the job it
        belonged to sat at 'processing' forever because expected_tiles could
        never be met. The DLQ keeps the evidence: the original task plus how
        many dispatch attempts it has now survived, so it can be inspected
        (or replayed) instead of silently disappearing.
        """
        job_id = task.get('job_id', 'unknown')
        tile_id = task.get('tile_id', -1)
        try:
            # The reaper stamps requeued tasks with 'attempt'; the master's
            # original dispatch carries none, which is attempt 1.
            dlq_record = dict(task)
            dlq_record['attempt'] = int(task.get('attempt', 1)) + 1
            dlq_record['failed_by'] = self.worker_id
            dlq_record['failure_reason'] = reason
            dlq_record['failed_at'] = time.time()

            self.task_producer.produce(
                TASK_DLQ_TOPIC,
                key=f"{job_id}:{tile_id}",
                value=json.dumps(dlq_record).encode('utf-8'),
            )
            # Flush rather than poll(0): the offset commit that follows is
            # what makes this tile unrecoverable from 'tasks', so the DLQ
            # copy has to be durable before that happens.
            self.task_producer.flush(timeout=5)
            dlq_total.labels(pipeline='opencv').inc()
            logger.error(f"Tile {tile_id} of job {job_id[:8]} FAILED permanently - sent to DLQ "
                         f"({TASK_DLQ_TOPIC}, attempt {dlq_record['attempt']}, reason: {reason})")
            return True
        except Exception as e:
            logger.error(f"Failed to send tile {tile_id} to DLQ: {str(e)}")
            return False

    def _result_callback(self, err, msg, tile_id):
        if err is not None:
            logger.error(f"Result delivery FAILED for tile {tile_id}: {err}")
        else:
            logger.debug(f"Result delivered: tile {tile_id} -> partition {msg.partition()}")

    def get_health_status(self):
        uptime = time.time() - self.metrics['start_time']
        avg_time = (self.metrics['total_processing_time'] /
                    self.metrics['tasks_processed']
                    if self.metrics['tasks_processed'] > 0 else 0)
        return {
            'worker_id': self.worker_id,
            'status': 'healthy' if self.running else 'stopped',
            'uptime_seconds': uptime,
            'tasks_processed': self.metrics['tasks_processed'],
            'tasks_failed': self.metrics['tasks_failed'],
            'avg_processing_time': avg_time
        }

    def run(self):
        logger.info("=" * 80)
        logger.info(f"WORKER NODE - {self.worker_id}")
        logger.info(f"Kafka Broker: {KAFKA_BROKER}")
        logger.info("=" * 80)

        if not self.initialize_kafka():
            logger.error("Failed to initialize Kafka. Exiting.")
            return

        self.heartbeat_sender = HeartbeatSender(self.worker_id)
        if self.heartbeat_sender.initialize():
            heartbeat_thread = Thread(target=self.heartbeat_sender.run, daemon=False)
            heartbeat_thread.start()
            logger.info("Heartbeat sender thread started")

        self.running = True
        logger.info(f"Worker {self.worker_id} started and waiting for tasks...")

        try:
            while self.running and not self.shutdown_event.is_set():
                msg = self.consumer.poll(timeout=1.0)

                if self.shutdown_event.is_set():
                    logger.info("Shutdown requested, finishing current task...")
                    break

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Kafka error: {msg.error()}")
                    break

                try:
                    task = json.loads(msg.value().decode('utf-8'))
                    logger.info(f"Task received - Job: {task['job_id']}, Tile: {task['tile_id']}")

                    result = self.process_tile(task)

                    if result:
                        success = self.publish_result(result)
                        if success:
                            self.consumer.commit(message=msg)
                            logger.info(f"Result published and offset committed for tile {task['tile_id']}")
                        else:
                            logger.error(f"Failed to publish result for tile {task['tile_id']}")
                    else:
                        # Retries exhausted. Park it on the DLQ first, then
                        # commit so 'tasks' doesn't redeliver it - the retry
                        # budget inside process_tile() has already been spent,
                        # so redelivery here would just burn it again.
                        self.send_to_dlq(task, 'retries exhausted in process_tile')
                        self.consumer.commit(message=msg)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in task message: {e}")
                    self.consumer.commit(message=msg)
                except Exception as e:
                    logger.error(f"Error in worker loop: {str(e)}", exc_info=True)

        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_id} interrupted...")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Starting cleanup...")
        self.running = False

        if self.heartbeat_sender:
            self.heartbeat_sender.stop()

        if self.task_producer:
            logger.info("Flushing producer...")
            self.task_producer.flush(timeout=10)

        if self.consumer:
            logger.info("Closing consumer...")
            self.consumer.close()

        health = self.get_health_status()
        logger.info(f"Final metrics - Processed: {health['tasks_processed']}, "
                    f"Failed: {health['tasks_failed']}, "
                    f"Avg time: {health['avg_processing_time']:.2f}s")
        logger.info(f"Worker {self.worker_id} cleanup completed")


def validate_config():
    errors = []
    if not KAFKA_BROKER:
        errors.append("KAFKA_BROKER not configured")

    try:
        test_producer = Producer({'bootstrap.servers': KAFKA_BROKER})
        test_producer.flush(timeout=5)
        logger.info("Kafka connectivity test passed")
    except Exception as e:
        errors.append(f"Cannot connect to Kafka: {str(e)}")

    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        return False
    return True


def main():
    logger.info("=" * 80)
    logger.info(f"Starting Worker {WORKER_ID}")
    logger.info(f"Kafka Broker: {KAFKA_BROKER}")
    logger.info(f"Heartbeat Interval: {HEARTBEAT_INTERVAL}s")
    logger.info("=" * 80)

    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        raise SystemExit(1)

    serve_metrics()
    logger.info(f"Metrics server listening on :9200/metrics")

    worker = Worker(WORKER_ID)
    worker.run()


if __name__ == '__main__':
    main()
