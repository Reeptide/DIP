# WORKER NODES (Nodes #3 & #4) - FIXED VERSION
# File: worker.py
# Complete Implementation with Confluent Kafka API

"""
WORKER NODE - Image Processing Worker
Uses Confluent Kafka API
- Consumes image tiles from Kafka 'tasks' topic
- Processes tiles using 12 OpenCV operations
- Publishes results to 'results' topic
- Sends periodic heartbeat messages to 'heartbeats' topic
Nodes #3 and #4 in architecture
"""

import os
import json
import time
import base64
import socket
import logging
import signal
from threading import Thread, Event
import numpy as np
import cv2
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

# ============================================================================
# CONFIGURATION
# ============================================================================
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
WORKER_ID = os.getenv('WORKER_ID', f"worker-{socket.gethostname()}")
HEARTBEAT_INTERVAL = 5  # Send heartbeat every N seconds

# Kafka Topics
TASK_TOPIC = 'tasks'
RESULT_TOPIC = 'results'
HEARTBEAT_TOPIC = 'heartbeats'

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format=f'[{WORKER_ID}] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFLUENT KAFKA CONFIGURATION
# ============================================================================
# Producer Configuration
producer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'compression.type': 'gzip',
    'acks': 'all',
    'retries': 3,
    'message.max.bytes': 52428800,  # 50MB
    'request.timeout.ms': 30000,
    'linger.ms': 100,
    'enable.idempotence': True  # Prevent duplicates
}

# Consumer Configuration
consumer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'auto.offset.reset': 'latest',
    'enable.auto.commit': False,  # Manual commit for reliability
    'session.timeout.ms': 30000,
    'max.poll.interval.ms': 300000,
}

# ============================================================================
# IMAGE PROCESSING CLASS (12 OPERATIONS)
# ============================================================================
class ImageProcessor:
    """Image processing operations using OpenCV"""
   
    # ===== BASIC OPERATIONS =====
   
    @staticmethod
    def grayscale(tile_image):
        """Convert tile to grayscale"""
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   
    @staticmethod
    def color_inversion(tile_image):
        """Invert colors of image"""
        return cv2.bitwise_not(tile_image)
   
    # ===== BLUR OPERATIONS =====
   
    @staticmethod
    def blur_gaussian(tile_image, kernel_size=51):
        """Apply Gaussian blur to tile - FIXED: Increased kernel size for visibility"""
        return cv2.GaussianBlur(tile_image, (kernel_size, kernel_size), 0)
   
    @staticmethod
    def blur_median(tile_image, kernel_size=15):
        """Apply median blur to tile"""
        return cv2.medianBlur(tile_image, kernel_size)
   
    @staticmethod
    def blur_bilateral(tile_image, diameter=9, sigma_color=75, sigma_space=75):
        """Apply bilateral blur (edge-preserving smoothing)"""
        return cv2.bilateralFilter(tile_image, diameter, sigma_color, sigma_space)
   
    # ===== EDGE DETECTION OPERATIONS =====
   
    @staticmethod
    def edge_canny(tile_image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection to tile"""
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
   
    @staticmethod
    def edge_sobel(tile_image):
        """Apply Sobel edge detection to tile"""
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(255 * sobel / np.max(sobel))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
   
    @staticmethod
    def edge_laplacian(tile_image):
        """Laplacian edge detection"""
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
   
    # ===== ENHANCEMENT & THRESHOLDING =====
   
    @staticmethod
    def sharpen(tile_image):
        """Sharpen image using unsharp mask technique"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(tile_image, -1, kernel)
   
    @staticmethod
    def histogram_equalization(tile_image):
        """Apply histogram equalization for contrast enhancement"""
        if len(tile_image.shape) == 3:
            ycrcb = cv2.cvtColor(tile_image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(tile_image)
   
    @staticmethod
    def threshold_binary(tile_image, threshold_value=127):
        """Apply binary thresholding"""
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
   
    # ===== ADVANCED OPERATIONS =====
   
    @staticmethod
    def contour_detection(tile_image):
        """Detect and draw contours"""
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = tile_image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result
   
    @staticmethod
    def morphological_opening(tile_image, kernel_size=5):
        """Morphological opening (erosion followed by dilation)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.morphologyEx(tile_image, cv2.MORPH_OPEN, kernel)
        return result

# ============================================================================
# IMAGE ENCODING/DECODING WITH VALIDATION
# ============================================================================
def decode_image(base64_string):
    """Decode base64 string to OpenCV image with validation"""
    try:
        if not base64_string or len(base64_string) < 100:
            raise ValueError("Invalid or too small base64 string")
       
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        if image is None or image.size == 0:
            raise ValueError("Failed to decode image or empty image")
       
        return image
       
    except Exception as e:
        logger.error(f"Image decode error: {str(e)}")
        raise

def encode_image(image):
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')

# ============================================================================
# HEARTBEAT SENDER
# ============================================================================
class HeartbeatSender:
    """Sends periodic heartbeat messages to Kafka using Confluent Kafka"""
   
    def __init__(self, worker_id, kafka_broker, interval=HEARTBEAT_INTERVAL):
        self.worker_id = worker_id
        self.kafka_broker = kafka_broker
        self.interval = interval
        self.running = False
        self.producer = None
        self.stop_event = Event()
   
    def initialize(self):
        """Initialize Kafka producer for heartbeats"""
        try:
            self.producer = Producer(producer_conf)
            logger.info("Heartbeat producer initialized (Confluent Kafka)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize heartbeat producer: {str(e)}")
            return False
   
    def delivery_report(self, err, msg):
        """Callback for delivery confirmation"""
        if err is not None:
            logger.debug(f"Heartbeat delivery failed: {err}")
        else:
            logger.debug(f"Heartbeat delivered to partition {msg.partition()}")
   
    def send_heartbeat(self):
        """Send a single heartbeat message"""
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
           
            # CRITICAL: Poll to trigger callbacks
            self.producer.poll(0)
           
            logger.debug(f"Heartbeat sent at {heartbeat_message['timestamp']}")
           
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {str(e)}")
   
    def run(self):
        """Background thread: Send heartbeats periodically"""
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
        """Stop sending heartbeats"""
        logger.info("Stopping heartbeat sender...")
        self.running = False
        self.stop_event.set()
        if self.producer:
            self.producer.flush(timeout=5)
        logger.info("Heartbeat sender stopped")

# ============================================================================
# WORKER CLASS WITH HEALTH METRICS
# ============================================================================
class Worker:
    """Main worker class for processing image tiles"""
   
    def __init__(self, worker_id, kafka_broker):
        self.worker_id = worker_id
        self.kafka_broker = kafka_broker
        self.consumer = None
        self.task_producer = None
        self.heartbeat_sender = None
        self.running = False
        self.shutdown_event = Event()
       
        # Metrics tracking
        self.metrics = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0,
            'start_time': time.time()
        }
       
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
   
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.running = False
   
    def initialize_kafka(self):
        """Initialize Kafka consumer and producer using Confluent Kafka"""
        try:
            consumer_config = consumer_conf.copy()
            consumer_config['group.id'] = 'image-processing-workers'
           
            self.consumer = Consumer(consumer_config)
            self.consumer.subscribe([TASK_TOPIC])
           
            self.task_producer = Producer(producer_conf)
           
            logger.info(f"Worker {self.worker_id} connected to Kafka broker: {self.kafka_broker} (Confluent Kafka)")
            return True
           
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {str(e)}")
            return False
   
    def process_tile(self, task, max_retries=2):
        """
        Process a single image tile with all 12 operations
       
        Args:
            task: Task message containing tile data
            max_retries: Number of retry attempts
       
        Returns:
            Result message with processed tile
        """
        job_id = task['job_id']
        tile_id = task['tile_id']
        operation = task['operation']
        tile_base64 = task['tile_image']
       
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                logger.info(f"Processing tile {tile_id} for job {job_id} with operation '{operation}' (attempt {attempt + 1})")
               
                tile_image = decode_image(tile_base64)
                processor = ImageProcessor()
               
                # Support all 12 operations
                operation_map = {
                    'grayscale': processor.grayscale,
                    'color_inversion': processor.color_inversion,
                    'blur_gaussian': processor.blur_gaussian,
                    'blur_median': processor.blur_median,
                    'blur_bilateral': processor.blur_bilateral,
                    'edge_canny': processor.edge_canny,
                    'edge_sobel': processor.edge_sobel,
                    'edge_laplacian': processor.edge_laplacian,
                    'sharpen': processor.sharpen,
                    'histogram_equalization': processor.histogram_equalization,
                    'threshold_binary': processor.threshold_binary,
                    'contour_detection': processor.contour_detection,
                    'morphological_opening': processor.morphological_opening
                }
               
                if operation not in operation_map:
                    logger.error(f"Unknown operation: {operation}")
                    self.metrics['tasks_failed'] += 1
                    return None
               
                processed_tile = operation_map[operation](tile_image)
                processed_base64 = encode_image(processed_tile)
                processing_time = time.time() - start_time
               
                result = {
                    'job_id': job_id,
                    'tile_id': tile_id,
                    'operation': operation,
                    'processed_tile': processed_base64,
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
               
                logger.info(f"Tile {tile_id} processed successfully in {processing_time:.2f}s")
                return result
               
            except Exception as e:
                logger.error(f"Error processing tile {tile_id} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    self.metrics['tasks_failed'] += 1
                    return None
   
    def publish_result(self, result):
        """Publish tile result to Kafka"""
        try:
            job_id = result['job_id']
            tile_id = result['tile_id']
            partition = tile_id % 2  # 2 partitions as per requirements
           
            self.task_producer.produce(
                RESULT_TOPIC,
                key=str(tile_id),
                value=json.dumps(result).encode('utf-8'),
                partition=partition,
                callback=lambda err, msg: self._result_callback(err, msg, tile_id)
            )
           
            # CRITICAL: Poll to trigger callbacks
            self.task_producer.poll(0)
           
            return True
           
        except BufferError:
            logger.warning(f"Producer queue full for tile {tile_id}, flushing...")
            self.task_producer.flush(timeout=5)
            return False
        except Exception as e:
            logger.error(f"Failed to publish result for tile {tile_id}: {str(e)}")
            return False
   
    def _result_callback(self, err, msg, tile_id):
        """Enhanced delivery callback"""
        if err is not None:
            logger.error(f"Result delivery FAILED for tile {tile_id}: {err}")
        else:
            logger.debug(f"Result delivered: tile {tile_id} -> partition {msg.partition()}")
   
    def get_health_status(self):
        """Return worker health metrics"""
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
        """Main worker loop"""
        logger.info("=" * 80)
        logger.info(f"WORKER NODE - {self.worker_id}")
        logger.info(f"Kafka API: Confluent Kafka")
        logger.info(f"Kafka Broker: {self.kafka_broker}")
        logger.info("=" * 80)
       
        if not self.initialize_kafka():
            logger.error("Failed to initialize Kafka. Exiting.")
            return
       
        # Start heartbeat sender thread
        self.heartbeat_sender = HeartbeatSender(self.worker_id, self.kafka_broker)
        if self.heartbeat_sender.initialize():
            heartbeat_thread = Thread(target=self.heartbeat_sender.run, daemon=False)
            heartbeat_thread.start()
            logger.info("Heartbeat sender thread started")
       
        self.running = True
        logger.info(f"Worker {self.worker_id} started and waiting for tasks...")
       
        try:
            while self.running and not self.shutdown_event.is_set():
                msg = self.consumer.poll(timeout=1.0)
               
                # Check for shutdown
                if self.shutdown_event.is_set():
                    logger.info("Shutdown requested, finishing current task...")
                    break
               
                if msg is None:
                    continue
               
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        break
               
                try:
                    task = json.loads(msg.value().decode('utf-8'))
                    logger.info(f"Task received - Job: {task['job_id']}, Tile: {task['tile_id']}")
                   
                    result = self.process_tile(task)
                   
                    if result:
                        success = self.publish_result(result)
                        if success:
                            # Commit offset only after successful processing
                            self.consumer.commit(message=msg)
                            logger.info(f"Result published and offset committed for tile {task['tile_id']}")
                        else:
                            logger.error(f"Failed to publish result for tile {task['tile_id']}")
                    else:
                        # Still commit to avoid reprocessing failed tasks indefinitely
                        self.consumer.commit(message=msg)
                        logger.warning(f"Task processing failed for tile {task['tile_id']}, offset committed")
                   
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in task message: {e}")
                    self.consumer.commit(message=msg)  # Skip bad message
                except Exception as e:
                    logger.error(f"Error in worker loop: {str(e)}", exc_info=True)
       
        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_id} interrupted...")
       
        finally:
            self.cleanup()
   
    def cleanup(self):
        """Cleanup resources"""
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
       
        # Log final metrics
        health = self.get_health_status()
        logger.info(f"Final metrics - Processed: {health['tasks_processed']}, "
                   f"Failed: {health['tasks_failed']}, "
                   f"Avg time: {health['avg_processing_time']:.2f}s")
       
        logger.info(f"Worker {self.worker_id} cleanup completed")

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration before starting"""
    errors = []
   
    if not KAFKA_BROKER:
        errors.append("KAFKA_BROKER not configured")
   
    if HEARTBEAT_INTERVAL < 1:
        errors.append("HEARTBEAT_INTERVAL too small (minimum 1 second)")
   
    # Test Kafka connectivity
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

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info(f"Starting Worker {WORKER_ID}")
    logger.info(f"Kafka Broker: {KAFKA_BROKER}")
    logger.info(f"Kafka API: Confluent Kafka")
    logger.info(f"Heartbeat Interval: {HEARTBEAT_INTERVAL}s")
    logger.info(f"Partitions: 2 (as per requirements)")
    logger.info("=" * 80)
   
    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        exit(1)
   
    worker = Worker(WORKER_ID, KAFKA_BROKER)
    worker.run()
