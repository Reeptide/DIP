"""ML INFERENCE WORKER - two-stage detect -> crop -> classify -> aggregate.

Separate consumer group from the OpenCV pipeline (worker/main.py) - this
subscribes to ml_tasks/ml_results, not tasks/results, so ML jobs and filter
jobs never compete for the same partitions.

Two threads share one Kafka Consumer, which is not safe to call
concurrently from confluent-kafka - so every Consumer method (poll, pause,
resume, commit) is called ONLY from the poller thread. The batcher thread
never touches the Consumer directly; it hands finished message objects
back through a queue for the poller to commit.

- Poller thread: polls Kafka, enqueues (task, msg) pairs, and applies
  backpressure - once INFERENCE_MAX_QUEUE tasks are buffered waiting for
  inference, it calls consumer.pause() so Kafka stops handing it more work
  (the alternative, an unbounded in-memory queue, just delays an OOM crash
  under sustained overload instead of preventing it). Resumes once the
  backlog drains below INFERENCE_RESUME_QUEUE.
- Batcher thread: accumulates a batch bounded by size OR time (whichever
  hits first), runs both ONNX models, publishes results.
"""
import json
import logging
import queue
import signal
import threading
import time

from confluent_kafka import Consumer, Producer, KafkaError

from app.blobstore import get_tile
from app.config import (
    WORKER_ID, ML_TASK_TOPIC, ML_RESULT_TOPIC,
    DETECT_MODEL_PATH, CLASSIFY_MODEL_PATH,
    INFERENCE_BATCH_SIZE, INFERENCE_BATCH_TIMEOUT_MS,
    INFERENCE_MAX_QUEUE, INFERENCE_RESUME_QUEUE,
    configure_logging,
)
from app.imaging import decode_image
from app.kafkaio import worker_producer_conf, worker_consumer_conf
from app.metrics import tiles_processed_total, tile_processing_seconds, queue_depth, serve_metrics
from inference.engine import DetectionEngine, ClassificationEngine

configure_logging(f"inference-{WORKER_ID}")
logger = logging.getLogger(__name__)


class InferenceWorker:
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.detector = None
        self.classifier = None

        self.pending = queue.Queue()     # (task: dict, msg: kafka Message)
        self.to_commit = queue.Queue()   # kafka Message, batcher -> poller
        self.paused = False
        self.shutdown_event = threading.Event()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()

    def initialize(self):
        consumer_conf = worker_consumer_conf()
        consumer_conf['group.id'] = 'ml-inference-workers'
        self.consumer = Consumer(consumer_conf)
        self.consumer.subscribe([ML_TASK_TOPIC])

        self.producer = Producer(worker_producer_conf())

        logger.info(f"Loading detection model: {DETECT_MODEL_PATH}")
        self.detector = DetectionEngine(DETECT_MODEL_PATH)
        logger.info(f"Loading classification model: {CLASSIFY_MODEL_PATH}")
        self.classifier = ClassificationEngine(CLASSIFY_MODEL_PATH)

    # ------------------------------------------------------------------
    # Poller thread - the only thread that touches self.consumer
    # ------------------------------------------------------------------
    def poll_loop(self):
        logger.info("Poller thread started")
        while not self.shutdown_event.is_set():
            try:
                self._drain_commits()
                self._apply_backpressure()

                msg = self.consumer.poll(timeout=0.2)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                try:
                    task = json.loads(msg.value().decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in ML task: {e}")
                    self.consumer.commit(message=msg, asynchronous=False)
                    continue

                self.pending.put((task, msg))

            except Exception as e:
                # Without this, an exception here silently kills the whole
                # poller thread - no traceback in the logs (this is how the
                # "consumer never processes anything, no crash visible" bug
                # was found: a bare loop body with no per-iteration guard).
                logger.error(f"Error in poller loop: {e}", exc_info=True)

        self._drain_commits()
        self.consumer.close()
        logger.info("Poller thread stopped")

    def _drain_commits(self):
        # Async, not sync: a synchronous commit() blocks this thread until
        # the broker acknowledges, and this loop can have many queued at
        # once (one per tile in a finished batch). Blocking here means not
        # calling consumer.poll() for that whole stretch - and if that gap
        # is long enough, the broker times out this consumer's session and
        # evicts it from the group. It silently rejoins on the next poll()
        # (a new member ID, no crash, no traceback - which is exactly what
        # made this bug hard to see: the process looked alive throughout).
        while True:
            try:
                msg = self.to_commit.get_nowait()
            except queue.Empty:
                return
            try:
                self.consumer.commit(message=msg, asynchronous=True)
            except Exception as e:
                logger.error(f"Commit failed: {e}")

    def _apply_backpressure(self):
        qsize = self.pending.qsize()
        queue_depth.labels(pipeline='ml').set(qsize)
        if not self.paused and qsize >= INFERENCE_MAX_QUEUE:
            self.consumer.pause(self.consumer.assignment())
            self.paused = True
            logger.warning(f"Backpressure engaged: {qsize} tasks pending, pausing consumer")
        elif self.paused and qsize <= INFERENCE_RESUME_QUEUE:
            self.consumer.resume(self.consumer.assignment())
            self.paused = False
            logger.info(f"Backpressure released: {qsize} tasks pending, resuming consumer")

    # ------------------------------------------------------------------
    # Batcher thread
    # ------------------------------------------------------------------
    def batch_loop(self):
        logger.info(f"Batcher thread started (batch_size={INFERENCE_BATCH_SIZE}, "
                     f"timeout={INFERENCE_BATCH_TIMEOUT_MS}ms)")
        while not self.shutdown_event.is_set():
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)
        logger.info("Batcher thread stopped")

    def _collect_batch(self):
        """Block for the first item (no upper bound - nothing to batch
        until something arrives), then keep pulling until either
        INFERENCE_BATCH_SIZE items are collected or INFERENCE_BATCH_TIMEOUT_MS
        has elapsed since the first item landed."""
        batch = []
        deadline = None

        while len(batch) < INFERENCE_BATCH_SIZE and not self.shutdown_event.is_set():
            timeout = 0.5 if deadline is None else max(0.0, deadline - time.time())
            if deadline is not None and timeout == 0.0:
                break
            try:
                item = self.pending.get(timeout=timeout)
            except queue.Empty:
                if deadline is not None:
                    break
                continue

            batch.append(item)
            if deadline is None:
                deadline = time.time() + INFERENCE_BATCH_TIMEOUT_MS / 1000

        return batch

    def _process_batch(self, batch):
        tasks = [t for t, _ in batch]
        msgs = [m for _, m in batch]
        start = time.time()

        images = []
        for task in tasks:
            try:
                images.append(decode_image(get_tile(task['blob_key'])))
            except Exception as e:
                logger.error(f"Failed to fetch/decode tile {task.get('tile_id')}: {e}")
                images.append(None)

        detect_results = [[] for _ in tasks]
        valid_idx = [i for i, im in enumerate(images) if im is not None]
        if valid_idx:
            batch_detections = self.detector.infer_batch([images[i] for i in valid_idx])
            for local_i, dets in zip(valid_idx, batch_detections):
                detect_results[local_i] = dets

        # Stage 2: crop every detected box across the WHOLE batch and
        # classify in one call - this is the point of dynamic batching:
        # one forward pass for N crops instead of N forward passes.
        crop_refs, crops = [], []
        for i, dets in enumerate(detect_results):
            image = images[i]
            for di, det in enumerate(dets):
                x1, y1, x2, y2 = [int(v) for v in det['box']]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                crops.append(image[y1:y2, x1:x2])
                crop_refs.append((i, di))

        if crops:
            classifications = self.classifier.infer_batch(crops)
            for (task_i, det_i), (label, conf) in zip(crop_refs, classifications):
                detect_results[task_i][det_i]['classify_label'] = label
                detect_results[task_i][det_i]['classify_confidence'] = round(conf, 4)

        processing_time = time.time() - start
        per_tile_time = processing_time / len(tasks)

        for _ in tasks:
            tiles_processed_total.labels(pipeline='ml', status='success').inc()
            tile_processing_seconds.labels(pipeline='ml').observe(per_tile_time)

        for task, dets in zip(tasks, detect_results):
            result = {
                'job_id': task['job_id'],
                'tile_id': task['tile_id'],
                'detections': dets,
                'worker_id': WORKER_ID,
                'processing_time': per_tile_time,
                'x': task['x'], 'y': task['y'],
                'width': task['width'], 'height': task['height'],
                'timestamp': time.time(),
            }
            try:
                self.producer.produce(
                    ML_RESULT_TOPIC,
                    key=f"{task['job_id']}:{task['tile_id']}",
                    value=json.dumps(result).encode('utf-8'),
                )
            except BufferError:
                self.producer.flush(timeout=5)
                self.producer.produce(
                    ML_RESULT_TOPIC,
                    key=f"{task['job_id']}:{task['tile_id']}",
                    value=json.dumps(result).encode('utf-8'),
                )
        self.producer.poll(0)
        self.producer.flush(timeout=10)

        for msg in msgs:
            self.to_commit.put(msg)

        total_dets = sum(len(d) for d in detect_results)
        logger.info(f"Batch of {len(tasks)} tiles inferred in {processing_time*1000:.1f}ms "
                     f"({per_tile_time*1000:.1f}ms/tile, {total_dets} detections)")

    def run(self):
        logger.info("=" * 80)
        logger.info(f"ML INFERENCE WORKER - {WORKER_ID}")
        logger.info("=" * 80)

        self.initialize()

        poller = threading.Thread(target=self.poll_loop, daemon=False, name="MLPoller")
        batcher = threading.Thread(target=self.batch_loop, daemon=False, name="MLBatcher")
        poller.start()
        batcher.start()

        poller.join()
        batcher.join()

        if self.producer:
            self.producer.flush(timeout=10)
        logger.info(f"Inference worker {WORKER_ID} stopped")


def main():
    serve_metrics()
    logger.info("Metrics server listening on :9200/metrics")
    InferenceWorker().run()


if __name__ == '__main__':
    main()
