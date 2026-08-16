# MASTER NODE (Node #1) - FIXED VERSION
# File: master.py
# Complete Implementation with Confluent Kafka API

"""
MASTER NODE - Image Processing Pipeline Coordinator
Uses Confluent Kafka, Redis, Flask
- Image tiling and task distribution
- Result collection and reconstruction
- Worker heartbeat monitoring
- Web UI for image upload and management
Node #1 in architecture
"""

import os
import json
import time
import uuid
import base64
import logging
import signal
import threading
from threading import Thread, RLock
from datetime import datetime
from functools import wraps
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_file
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
import redis
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
INSTANCE_ID = os.getenv('INSTANCE_ID', 'master-1')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Kafka Topics (3 required topics)
TASK_TOPIC = 'tasks'
RESULT_TOPIC = 'results'
HEARTBEAT_TOPIC = 'heartbeats'

# Image Tiling Configuration
TILE_SIZE = 512  # Minimum tile size requirement
MIN_IMAGE_SIZE = 1024  # Minimum image size requirement
MAX_IMAGE_DIMENSION = 8192  # Maximum dimension
MAX_TILES = 1000  # Prevent memory exhaustion

# Heartbeat Configuration
HEARTBEAT_TIMEOUT = 15  # seconds - consider worker dead if no heartbeat after this
HEARTBEAT_CHECK_INTERVAL = 2  # seconds

# Rate Limiting
RATE_LIMIT = 10  # requests per minute
RATE_WINDOW = 60

# Batching Configuration - PERFORMANCE OPTIMIZATION
BATCH_SIZE = 5  # Number of tasks to batch together
BATCH_FLUSH_INTERVAL = 0.05  # Flush batch every 50ms

# ============================================================================
# FLASK APP SETUP
# ============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Create templates directory if it doesn't exist
TEMPLATES_FOLDER = 'templates'
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[MASTER] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PROCESSING OPERATIONS (12 TOTAL)
# ============================================================================
PROCESSING_OPERATIONS = {
    # Basic Operations
    'grayscale': 'Grayscale Conversion',
    'color_inversion': 'Color Inversion',
    
    # Blur Operations
    'blur_gaussian': 'Gaussian Blur',
    'blur_median': 'Median Blur',
    'blur_bilateral': 'Bilateral Blur (Edge-Preserving)',
    
    # Edge Detection Operations
    'edge_canny': 'Canny Edge Detection',
    'edge_sobel': 'Sobel Edge Detection',
    'edge_laplacian': 'Laplacian Edge Detection',
    
    # Enhancement & Thresholding
    'sharpen': 'Image Sharpening',
    'histogram_equalization': 'Histogram Equalization',
    'threshold_binary': 'Binary Thresholding',
    
    # Advanced Operations
    'contour_detection': 'Contour Detection',
    'morphological_opening': 'Morphological Opening'
}

# ============================================================================
# REDIS CLIENT WITH ERROR HANDLING
# ============================================================================
class RedisClient:
    """Wrapper for Redis operations with error handling and retries"""
    
    def __init__(self, host, port, max_retries=3):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        self.max_retries = max_retries
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry Redis operation with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except redis.RedisError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Redis operation failed after {self.max_retries} attempts: {e}")
                    raise
                wait_time = 0.1 * (2 ** attempt)
                logger.warning(f"Redis error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    def set(self, key, value, ex=None):
        return self._retry_operation(self.client.set, key, value, ex=ex)
    
    def get(self, key):
        return self._retry_operation(self.client.get, key)
    
    def incr(self, key):
        return self._retry_operation(self.client.incr, key)
    
    def delete(self, *keys):
        return self._retry_operation(self.client.delete, *keys)
    
    def setex(self, key, time, value):
        return self._retry_operation(self.client.setex, key, time, value)
    
    def ping(self):
        try:
            return self.client.ping()
        except redis.RedisError:
            return False

redis_client = RedisClient(REDIS_HOST, REDIS_PORT)

# ============================================================================
# CONFLUENT KAFKA CONFIGURATION
# ============================================================================
# Producer Configuration - OPTIMIZED FOR BATCHING
producer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'compression.type': 'gzip',
    'acks': 'all',
    'retries': 3,
    'message.max.bytes': 52428800,  # 50MB
    'request.timeout.ms': 30000,
    'linger.ms': 50,  # Reduced for faster batching
    'batch.size': 500000,  # 500KB batches
    'enable.idempotence': True
}

# Consumer Configuration
consumer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True,
    'session.timeout.ms': 30000,
    'max.poll.interval.ms': 300000,
}

# ============================================================================
# KAFKA PRODUCER POOL (SIMPLIFIED AND FIXED)
# ============================================================================
class KafkaProducerPool:
    """Thread-safe singleton producer pool"""
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
        """Initialize the producer once"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    try:
                        self._producer = Producer(producer_conf)
                        self._initialized = True
                        logger.info("Global Kafka producer initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize producer: {e}")
                        raise
    
    def get_producer(self):
        """Get the shared producer instance"""
        if not self._initialized:
            self.initialize()
        return self._producer
    
    def cleanup(self):
        """Cleanup producer on shutdown"""
        if self._producer:
            logger.info("Flushing producer...")
            self._producer.flush(timeout=30)
            logger.info("Producer flushed")

# Create the singleton instance and initialize it immediately
producer_pool = KafkaProducerPool()

# ============================================================================
# WORKER HEALTH TRACKING (THREAD-SAFE)
# ============================================================================
class WorkerHeartbeatTracker:
    """Thread-safe worker heartbeat tracking"""
    
    def __init__(self):
        self.heartbeats = {}
        self.lock = RLock()
    
    def update(self, worker_id, timestamp):
        """Update worker heartbeat"""
        with self.lock:
            self.heartbeats[worker_id] = {
                'last_seen': timestamp,
                'status': 'alive',
                'message_time': time.time()
            }
    
    def get_active_workers(self, timeout=HEARTBEAT_TIMEOUT):
        """Get list of active workers (thread-safe)"""
        now = time.time()
        active = []
        dead = []
        
        with self.lock:
            for worker_id, data in list(self.heartbeats.items()):
                time_since_heartbeat = now - data['last_seen']
                if time_since_heartbeat < timeout:
                    active.append(worker_id)
                else:
                    dead.append(worker_id)
            
            # Remove dead workers
            for worker_id in dead:
                del self.heartbeats[worker_id]
                logger.warning(f"Worker {worker_id} marked as dead (no heartbeat for {timeout}s)")
        
        return active
    
    def get_all_heartbeats(self):
        """Get copy of all heartbeat data"""
        with self.lock:
            return dict(self.heartbeats)

heartbeat_tracker = WorkerHeartbeatTracker()

# ============================================================================
# HEARTBEAT MONITORING
# ============================================================================
def monitor_heartbeats():
    """Background thread: Listen to worker heartbeats using Confluent Kafka"""
    logger.info("Heartbeat monitor started (Confluent Kafka)")
    
    consumer_config = consumer_conf.copy()
    consumer_config['group.id'] = f'master-heartbeat-monitor-{INSTANCE_ID}'
    
    consumer = Consumer(consumer_config)
    
    try:
        # Subscribe to heartbeat topic
        consumer.subscribe([HEARTBEAT_TOPIC])
        logger.info(f"✓ Subscribed to heartbeat topic: {HEARTBEAT_TOPIC}")
        
        # Add debug: List all available topics
        metadata = consumer.list_topics(timeout=10)
        available_topics = list(metadata.topics.keys())
        logger.info(f"Available Kafka topics: {available_topics}")
        
        if HEARTBEAT_TOPIC not in available_topics:
            logger.error(f"❌ Heartbeat topic '{HEARTBEAT_TOPIC}' does not exist!")
            logger.info("Available topics: " + ", ".join(available_topics))
    
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
                else:
                    logger.error(f"Heartbeat consumer error: {msg.error()}")
                    continue
            
            message_count += 1
            
            try:
                # Debug: Log raw message
                raw_message = msg.value().decode('utf-8')
                logger.debug(f"Raw heartbeat message #{message_count}: {raw_message}")
                
                heartbeat = json.loads(raw_message)
                worker_id = heartbeat.get('worker_id')
                timestamp = heartbeat.get('timestamp')
                
                if worker_id and timestamp:
                    heartbeat_tracker.update(worker_id, timestamp)
                    logger.info(f"✓ Heartbeat from {worker_id} at {timestamp} (msg #{message_count})")
                else:
                    logger.warning(f"Invalid heartbeat data: worker_id={worker_id}, timestamp={timestamp}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in heartbeat: {e}")
                logger.error(f"Raw data: {msg.value()}")
            except Exception as e:
                logger.error(f"Error processing heartbeat: {str(e)}")
    
    except KeyboardInterrupt:
        logger.info("Heartbeat monitor stopped")
    finally:
        consumer.close()
        logger.info(f"Heartbeat monitor closed. Total messages processed: {message_count}")

def heartbeat_checker():
    """Background thread: Periodically check for dead workers"""
    logger.info("Heartbeat checker started")
    
    while True:
        try:
            # Debug: Show all heartbeat data
            all_heartbeats = heartbeat_tracker.get_all_heartbeats()
            active = heartbeat_tracker.get_active_workers()
            
            # Enhanced logging
            logger.info(f"Heartbeat check - Total tracked: {len(all_heartbeats)}, Active: {len(active)}")
            
            if all_heartbeats:
                logger.info("All tracked workers:")
                current_time = time.time()
                for worker_id, data in all_heartbeats.items():
                    age = current_time - data.get('last_seen', 0)
                    logger.info(f"  - {worker_id}: {age:.1f}s ago, status: {data.get('status', 'unknown')}")
            else:
                logger.warning("No workers in heartbeat tracker")
            
            redis_client.set('active_workers_count', len(active))
            redis_client.set('active_workers', json.dumps(active))
            
            logger.info(f"Active workers: {len(active)} - {active}")
            time.sleep(HEARTBEAT_CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in heartbeat checker: {str(e)}")
            time.sleep(HEARTBEAT_CHECK_INTERVAL)

# ============================================================================
# IMAGE TILING LOGIC WITH SAFETY CHECKS
# ============================================================================
def split_image_into_tiles(image, tile_size=TILE_SIZE):
    """
    Split image into tiles with safety checks
    
    Args:
        image: OpenCV image
        tile_size: Tile size in pixels
    
    Returns:
        List of tiles with metadata
    """
    height, width = image.shape[:2]
    
    # Calculate expected tile count
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    expected_tiles = tiles_x * tiles_y
    
    if expected_tiles > MAX_TILES:
        raise ValueError(
            f"Image too large: would create {expected_tiles} tiles (max: {MAX_TILES}). "
            f"Reduce image size or increase tile size."
        )
    
    logger.info(f"Splitting image {width}x{height} into {tile_size}x{tile_size} tiles (expected: {expected_tiles})")
    
    tiles = []
    tile_id = 0
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Skip tiny tiles
            if (x_end - x) < 16 or (y_end - y) < 16:
                logger.warning(f"Skipping tiny tile at ({x},{y})")
                continue
            
            tile = image[y:y_end, x:x_end].copy()
            
            tile_info = {
                'tile_id': tile_id,
                'x': x,
                'y': y,
                'width': x_end - x,
                'height': y_end - y,
                'tile_data': tile
            }
            
            tiles.append(tile_info)
            logger.debug(f"Tile {tile_id}: position ({x},{y}), size {tile_info['width']}x{tile_info['height']}")
            tile_id += 1
    
    logger.info(f"Total tiles created: {len(tiles)}")
    return tiles

def encode_image(image):
    """Encode OpenCV image to base64"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')

def decode_image(base64_string):
    """Decode base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def reconstruct_image_from_tiles(tiles_data, original_width, original_height):
    """Reconstruct final image from processed tiles"""
    logger.info(f"Reconstructing image {original_width}x{original_height} from {len(tiles_data)} tiles")
    
    reconstructed = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    
    for tile_info in tiles_data:
        try:
            tile_image = decode_image(tile_info['processed_tile'])
            x = tile_info['x']
            y = tile_info['y']
            width = tile_info['width']
            height = tile_info['height']
            
            reconstructed[y:y+height, x:x+width] = tile_image
        except Exception as e:
            logger.error(f"Error reconstructing tile {tile_info.get('tile_id')}: {e}")
    
    logger.info("Image reconstruction completed")
    return reconstructed

# ============================================================================
# KAFKA TASK PUBLISHING WITH BATCHING - PERFORMANCE OPTIMIZATION
# ============================================================================
def delivery_callback(err, msg, tile_id, job_id):
    """Enhanced delivery callback with error tracking"""
    if err is not None:
        logger.error(f"Task delivery FAILED - Job: {job_id}, Tile: {tile_id}, Error: {err}")
        try:
            redis_client.incr(f"failed_tasks:{job_id}")
        except Exception as e:
            logger.error(f"Failed to track delivery error: {e}")
    else:
        logger.debug(f"Task delivered - Tile: {tile_id} -> Partition: {msg.partition()}")

def publish_tile_tasks(job_id, tiles, operation):
    """Publish tile tasks using shared producer with BATCHING for performance"""
    logger.info(f"Publishing {len(tiles)} tile tasks for job {job_id} (with batching)")
    
    producer = producer_pool.get_producer()
    published_count = 0
    failed_count = 0
    batch_count = 0
    
    try:
        for i, tile in enumerate(tiles):
            try:
                tile_base64 = encode_image(tile['tile_data'])
                
                task = {
                    'job_id': job_id,
                    'tile_id': tile['tile_id'],
                    'operation': operation,
                    'tile_image': tile_base64,
                    'x': tile['x'],
                    'y': tile['y'],
                    'width': tile['width'],
                    'height': tile['height'],
                    'timestamp': time.time()
                }
                
                partition = tile['tile_id'] % 2  # 2 partitions as per requirements
                
                producer.produce(
                    TASK_TOPIC,
                    key=str(tile['tile_id']),
                    value=json.dumps(task).encode('utf-8'),
                    partition=partition,
                    callback=lambda err, msg, tid=tile['tile_id']: 
                        delivery_callback(err, msg, tid, job_id)
                )
                
                # CRITICAL: Poll to trigger callbacks
                producer.poll(0)
                published_count += 1
                
                # Batch flush - flush every BATCH_SIZE tasks for better throughput
                if (i + 1) % BATCH_SIZE == 0:
                    producer.flush(timeout=1)
                    batch_count += 1
                    logger.debug(f"Batch {batch_count} flushed ({BATCH_SIZE} tasks)")
                
            except BufferError:
                logger.warning(f"Producer queue full at tile {tile['tile_id']}, flushing...")
                producer.flush(timeout=5)
                failed_count += 1
            except Exception as e:
                logger.error(f"Error publishing tile {tile['tile_id']}: {e}")
                failed_count += 1
        
        # Final flush for remaining tasks
        producer.flush(timeout=10)
        
        logger.info(f"Task publishing complete - Published: {published_count}, Failed: {failed_count}, Batches: {batch_count}")
        return published_count > 0
        
    except Exception as e:
        logger.error(f"Critical error in publish_tile_tasks: {str(e)}")
        return False

# ============================================================================
# RESULT LISTENER - CONSOLIDATED STORAGE
# ============================================================================
def listen_for_results():
    """Background thread: Listen for processed results - CONSOLIDATED STORAGE"""
    logger.info("Result listener started with CONSOLIDATED storage")
    
    consumer_config = consumer_conf.copy()
    consumer_config['group.id'] = f'master-result-consumer-{INSTANCE_ID}'
    
    consumer = Consumer(consumer_config)
    consumer.subscribe([RESULT_TOPIC])
    
    try:
        while True:
            try:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Result consumer error: {msg.error()}")
                        continue
                
                result = json.loads(msg.value().decode('utf-8'))
                job_id = result['job_id']
                tile_id = result['tile_id']
                short_job_id = job_id[:8]
                
                logger.info(f"Processing result for job {short_job_id}, tile {tile_id}")
                
                # CONSOLIDATED APPROACH - Update tile tracking in ONE key
                tiles_key = f"job:{short_job_id}:tiles"
                tiles_data_str = redis_client.get(tiles_key)
                
                if tiles_data_str:
                    tiles_data = json.loads(tiles_data_str)
                    
                    # Store tile result in consolidated structure
                    tiles_data['received_tiles'][str(tile_id)] = {
                        'processed_tile': result['processed_tile'],
                        'worker_id': result['worker_id'],
                        'processing_time': result.get('processing_time', 0),
                        'timestamp': time.time(),
                        'x': result.get('x', 0),
                        'y': result.get('y', 0),
                        'width': result.get('width', 0),
                        'height': result.get('height', 0)
                    }
                    
                    received_count = len(tiles_data['received_tiles'])
                    expected_count = tiles_data['expected_tiles']
                    
                    # Update job metadata
                    job_key = f"job:{short_job_id}"
                    job_data_str = redis_client.get(job_key)
                    if job_data_str:
                        job_data = json.loads(job_data_str)
                        job_data['results_count'] = received_count
                        job_data['tiles_received'] = list(tiles_data['received_tiles'].keys())
                        
                        # Check if complete
                        if received_count >= expected_count:
                            job_data['status'] = 'ready_for_reconstruction'
                            tiles_data['status'] = 'complete'
                            logger.info(f"Job {short_job_id} complete - all {received_count} tiles received")
                        
                        # Update both keys
                        redis_client.setex(job_key, 86400, json.dumps(job_data))
                        redis_client.setex(tiles_key, 86400, json.dumps(tiles_data))
                        
                        # Update system counter
                        redis_client.set('tiles_processed', str(received_count))
                
                logger.info(f"Result stored - Job: {short_job_id}, Tile: {tile_id}, Total received: {received_count}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in result: {e}")
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
    
    except KeyboardInterrupt:
        logger.info("Result listener shutting down")
    finally:
        consumer.close()

# ============================================================================
# RATE LIMITING
# ============================================================================
request_counts = defaultdict(list)

def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        now = time.time()
        
        # Clean old requests
        request_counts[ip] = [t for t in request_counts[ip] if now - t < RATE_WINDOW]
        
        if len(request_counts[ip]) >= RATE_LIMIT:
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        
        request_counts[ip].append(now)
        return f(*args, **kwargs)
    
    return decorated_function

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', operations=PROCESSING_OPERATIONS)

@app.route('/upload', methods=['POST'])
@rate_limit
def upload_image():
    """Handle image upload and create processing tasks - CONSOLIDATED STORAGE"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    operation = request.form.get('operation', 'grayscale')
    
    if operation not in PROCESSING_OPERATIONS:
        return jsonify({'error': f'Invalid operation. Valid options: {list(PROCESSING_OPERATIONS.keys())}'}), 400
    
    if not file or file.filename == '':
        return jsonify({'error': 'Invalid file'}), 400
    
    if '.' not in file.filename:
        return jsonify({'error': 'File must have an extension'}), 400
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Check active workers
        active_workers = heartbeat_tracker.get_active_workers()
        if len(active_workers) == 0:
            return jsonify({'error': 'No active workers available. Please start workers first.'}), 503
        
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        
        file.save(filepath)
        
        # Validate image
        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to load image or corrupted file'}), 400
        
        height, width = image.shape[:2]
        
        # Size validation
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            os.remove(filepath)
            return jsonify({
                'error': f'Image too small. Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}, got: {width}x{height}'
            }), 400
        
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            os.remove(filepath)
            return jsonify({
                'error': f'Image too large. Maximum dimension: {MAX_IMAGE_DIMENSION}px'
            }), 400
        
        # Create tiles
        try:
            tiles = split_image_into_tiles(image, TILE_SIZE)
        except ValueError as e:
            os.remove(filepath)
            return jsonify({'error': str(e)}), 400
        
        expected_tiles = len(tiles)
        short_job_id = job_id[:8]
        
        # CONSOLIDATED JOB STORAGE - Store everything in job metadata
        job_metadata = {
            'job_id': job_id,
            'original_filename': filename,
            'operation': operation,
            'original_width': width,
            'original_height': height,
            'timestamp': time.time(),
            'status': 'processing',
            'tiles_count': expected_tiles,
            'active_workers': len(active_workers),
            'expected_tiles': expected_tiles,
            'results_count': 0,
            'failed_tasks': 0,
            'tiles_received': [],  # Track which tiles are received
            'processing_start': time.time()
        }
        
        # Store job metadata in ONE key (TTL 24 hours)
        redis_client.setex(f"job:{short_job_id}", 86400, json.dumps(job_metadata))
        
        # Initialize consolidated tile tracking in ONE key
        tile_tracking = {
            'expected_tiles': expected_tiles,
            'received_tiles': {},  # Will store tile results here
            'status': 'processing',
            'created_at': time.time()
        }
        redis_client.setex(f"job:{short_job_id}:tiles", 86400, json.dumps(tile_tracking))
        
        # Update system counters only
        redis_client.set('tiles_total', str(expected_tiles))
        redis_client.set('tiles_processed', '0')
        
        # Publish tasks
        success = publish_tile_tasks(job_id, tiles, operation)
        
        if not success:
            return jsonify({'error': 'Failed to publish tasks to workers'}), 500
        
        logger.info(f"Job {short_job_id} created with CONSOLIDATED storage: {expected_tiles} tiles")
        
        return jsonify({
            'message': 'Image uploaded and processing started',
            'job_id': short_job_id,  # Return short ID
            'tiles_count': expected_tiles,
            'operation': operation,
            'active_workers': len(active_workers),
            'image_size': f"{width}x{height}"
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check job processing status - CONSOLIDATED VERSION"""
    try:
        short_job_id = job_id[:8] if len(job_id) > 8 else job_id
        
        # Get job data from consolidated storage
        job_data_str = redis_client.get(f"job:{short_job_id}")
        if not job_data_str:
            return jsonify({'error': 'Job not found'}), 404
        
        job_data = json.loads(job_data_str)
        
        # Calculate progress
        results_count = job_data.get('results_count', 0)
        expected_tiles = job_data.get('expected_tiles', 1)
        progress = int((results_count / expected_tiles) * 100)
        
        return jsonify({
            'job_id': short_job_id,
            'status': job_data.get('status', 'unknown'),
            'progress': progress,
            'received_tiles': results_count,
            'expected_tiles': expected_tiles,
            'failed_tasks': job_data.get('failed_tasks', 0),
            'job_data': job_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Retrieve processed image"""
    try:
        short_job_id = job_id[:8] if len(job_id) > 8 else job_id
        
        # Get job data
        job_data_str = redis_client.get(f"job:{short_job_id}")
        if not job_data_str:
            return jsonify({'error': 'Job not found'}), 404
        
        job_data = json.loads(job_data_str)
        
        if job_data.get('status') != 'completed':
            return jsonify({'error': f'Job not completed yet. Current status: {job_data.get("status")}'}), 400
        
        result_path = job_data.get('result_path')
        
        if not result_path or not os.path.exists(result_path):
            return jsonify({'error': 'Result file not found'}), 404
        
        return send_file(result_path, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error retrieving result: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reconstruct/<job_id>', methods=['POST'])
def reconstruct_job(job_id):
    """Reconstruct image from consolidated tile storage"""
    try:
        short_job_id = job_id[:8] if len(job_id) > 8 else job_id
        
        # Get job and tile data
        job_data_str = redis_client.get(f"job:{short_job_id}")
        tiles_data_str = redis_client.get(f"job:{short_job_id}:tiles")
        
        if not job_data_str or not tiles_data_str:
            return jsonify({'error': 'Job or tile data not found'}), 404
        
        job_data = json.loads(job_data_str)
        tiles_data = json.loads(tiles_data_str)
        
        received_tiles = tiles_data['received_tiles']
        expected_count = tiles_data['expected_tiles']
        
        if len(received_tiles) < expected_count:
            return jsonify({
                'error': f'Not all tiles received. Expected: {expected_count}, Got: {len(received_tiles)}'
            }), 400
        
        # Convert tile data for reconstruction
        tiles_for_reconstruction = []
        for tile_id, tile_info in received_tiles.items():
            tiles_for_reconstruction.append({
                'tile_id': int(tile_id),
                'processed_tile': tile_info['processed_tile'],
                'x': tile_info['x'],
                'y': tile_info['y'],
                'width': tile_info['width'],
                'height': tile_info['height']
            })
        
        # Sort by tile ID
        tiles_for_reconstruction.sort(key=lambda x: x['tile_id'])
        
        # Reconstruct image
        final_image = reconstruct_image_from_tiles(
            tiles_for_reconstruction,
            job_data['original_width'],
            job_data['original_height']
        )
        
        # Save result
        result_filename = f"{short_job_id}_processed.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, final_image)
        
        # Update job status
        job_data['status'] = 'completed'
        job_data['result_path'] = result_path
        job_data['completion_time'] = time.time()
        redis_client.setex(f"job:{short_job_id}", 86400, json.dumps(job_data))
        
        # Clean up tile data after successful reconstruction
        redis_client.delete(f"job:{short_job_id}:tiles")
        
        logger.info(f"Job {short_job_id} reconstruction completed and tile data cleaned")
        
        return jsonify({
            'message': 'Image reconstructed successfully',
            'job_id': short_job_id,
            'result_path': result_path,
            'tiles_processed': len(tiles_for_reconstruction)
        }), 200
        
    except Exception as e:
        logger.error(f"Error reconstructing image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Render dashboard page with worker health status - IMPROVED UI"""
    return render_template('dashboard.html')

@app.route('/dashboard/data', methods=['GET'])
def dashboard_data():
    """Get dashboard data with worker health status and heartbeat details"""
    try:
        active_workers = heartbeat_tracker.get_active_workers()
        all_heartbeats = heartbeat_tracker.get_all_heartbeats()
        
        worker_details = []
        current_time = time.time()
        
        for wid in active_workers:
            hb = all_heartbeats.get(wid, {})
            last_seen = hb.get('last_seen', 0)
            last_seen_ago = current_time - last_seen
            
            worker_details.append({
                'worker_id': wid,
                'last_seen': last_seen,
                'last_seen_ago': last_seen_ago,
                'status': hb.get('status', 'unknown'),
                'heartbeat_timestamp': datetime.fromtimestamp(last_seen).strftime('%H:%M:%S') if last_seen > 0 else 'Never',
                'is_healthy': last_seen_ago < HEARTBEAT_TIMEOUT,
                'response_time_ms': round(last_seen_ago * 1000, 0)  # Convert to milliseconds
            })
        
        # Sort by most recent heartbeat first
        worker_details.sort(key=lambda x: x['last_seen'], reverse=True)
        
        # Calculate system health score
        total_workers = len(worker_details)
        healthy_workers = len([w for w in worker_details if w['is_healthy']])
        health_percentage = (healthy_workers / total_workers * 100) if total_workers > 0 else 100
        
        return jsonify({
            'active_workers': len(active_workers),
            'worker_list': worker_details,
            'timestamp': current_time,
            'kafka_broker': KAFKA_BROKER,
            'redis_connected': redis_client.ping(),
            'operations_available': len(PROCESSING_OPERATIONS),
            'heartbeat_stats': {
                'total_workers': total_workers,
                'healthy_workers': healthy_workers,
                'health_percentage': health_percentage,
                'heartbeat_timeout': HEARTBEAT_TIMEOUT,
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        active_workers = heartbeat_tracker.get_active_workers()
        redis_healthy = redis_client.ping()
        
        # Test Kafka connectivity
        kafka_healthy = False
        try:
            producer = producer_pool.get_producer()
            if producer:
                kafka_healthy = True
        except Exception:
            pass
        
        overall_status = 'healthy' if (redis_healthy and kafka_healthy) else 'degraded'
        
        return jsonify({
            'status': overall_status,
            'kafka_broker': KAFKA_BROKER,
            'kafka_api': 'Confluent Kafka',
            'kafka_healthy': kafka_healthy,
            'redis_host': REDIS_HOST,
            'redis_healthy': redis_healthy,
            'active_workers': len(active_workers),
            'operations': len(PROCESSING_OPERATIONS),
            'partitions': 2,
            'timestamp': time.time()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List recent jobs (for debugging)"""
    try:
        # This is a simple implementation - in production you'd want proper job tracking
        return jsonify({
            'message': 'Job listing not implemented. Use /status/<job_id> to check specific jobs.',
            'active_workers': len(heartbeat_tracker.get_active_workers())
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metadata', methods=['GET'])
def get_metadata():
    """Get comprehensive metadata about jobs and Redis storage - CONSOLIDATED VERSION"""
    try:
        # Get all Redis keys
        all_keys = []
        job_keys = []           # job:abc12345
        tile_tracking_keys = [] # job:abc12345:tiles  
        counter_keys = []       # active_workers, tiles_total, etc.
        orphaned_keys = []      # Keys that don't belong to existing jobs
        
        try:
            # Get all keys
            cursor = '0'
            while cursor != 0:
                cursor, keys = redis_client.client.scan(cursor=cursor, count=1000)
                all_keys.extend(keys)
            
            # Get list of valid job IDs first
            valid_job_ids = set()
            for key in all_keys:
                if key.startswith('job:') and ':tiles' not in key:
                    job_id = key[4:]  # Remove 'job:' prefix
                    valid_job_ids.add(job_id)
            
            logger.info(f"Found {len(valid_job_ids)} valid job IDs: {list(valid_job_ids)[:5]}...")
            
            # Categorize keys in CONSOLIDATED approach
            for key in all_keys:
                if key.startswith('job:') and ':tiles' not in key:
                    job_keys.append(key)
                elif key.startswith('job:') and ':tiles' in key:
                    # Check if parent job exists
                    parent_job_id = key.split(':')[1]
                    if parent_job_id in valid_job_ids:
                        tile_tracking_keys.append(key)
                    else:
                        orphaned_keys.append(key)
                        logger.debug(f"Orphaned tile tracking key: {key}")
                elif key in ['tiles_total', 'tiles_processed', 'active_workers_count', 'active_workers']:
                    counter_keys.append(key)
                else:
                    # All other keys are orphaned in consolidated storage
                    orphaned_keys.append(key)
                    logger.debug(f"Orphaned key: {key}")
                    
        except Exception as e:
            logger.error(f"Error scanning Redis keys: {e}")
        
        logger.info(f"CONSOLIDATED Key categorization: Jobs={len(job_keys)}, Tile tracking={len(tile_tracking_keys)}, Counters={len(counter_keys)}, Orphaned={len(orphaned_keys)}")
        
        # Get detailed job information
        jobs = []
        for job_key in job_keys:
            try:
                job_data_str = redis_client.get(job_key)
                if job_data_str:
                    job_data = json.loads(job_data_str)
                    job_id = job_data.get('job_id', job_key.replace('job:', ''))
                    short_job_id = job_id[:8]
                    
                    # Calculate processing time
                    created_at = job_data.get('timestamp', time.time())
                    completion_time = job_data.get('completion_time')
                    processing_time = None
                    if completion_time:
                        try:
                            processing_time = float(completion_time) - created_at
                        except:
                            processing_time = None
                    
                    # Calculate grid dimensions
                    tiles_count = int(job_data.get('expected_tiles', 0))
                    rows = int((tiles_count ** 0.5)) or 1
                    cols = (tiles_count + rows - 1) // rows
                    
                    job_info = {
                        'job_id': short_job_id,
                        'full_job_id': job_id,
                        'operation': job_data.get('operation', 'unknown'),
                        'original_filename': job_data.get('original_filename', 'unknown'),
                        'status': job_data.get('status', 'unknown'),
                        'total_tiles': str(job_data.get('expected_tiles', 0)),
                        'tiles_received': str(job_data.get('results_count', 0)),
                        'failed_tasks': str(job_data.get('failed_tasks', 0)),
                        'rows': str(rows),
                        'cols': str(cols),
                        'created_at': str(int(created_at)),
                        'created_at_readable': datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S'),
                        'image_size': f"{job_data.get('original_width', 0)}x{job_data.get('original_height', 0)}",
                        'active_workers': str(job_data.get('active_workers', 0)),
                        'processing_time': f"{processing_time:.2f}s" if processing_time else 'N/A',
                        'completion_time': completion_time,
                        'result_available': bool(job_data.get('result_path')),
                        'progress': int((job_data.get('results_count', 0) / max(job_data.get('expected_tiles', 1), 1)) * 100)
                    }
                    jobs.append(job_info)
            except Exception as e:
                logger.error(f"Error processing job {job_key}: {e}")
        
        # Sort jobs by creation time (newest first)
        jobs.sort(key=lambda x: float(x['created_at']), reverse=True)
        
        # Get current counters
        current_counters = {}
        for key in counter_keys:
            try:
                value = redis_client.get(key)
                current_counters[key] = value or '0'
            except:
                current_counters[key] = '0'
        
        # Get active workers info
        active_workers = heartbeat_tracker.get_active_workers()
        
        # Prepare response with CONSOLIDATED cleanup information
        metadata = {
            'total_jobs': len(jobs),
            'jobs': jobs,
            'redis_summary': {
                'total_keys': len(all_keys),
                'job_keys': len(job_keys),
                'tile_keys': len(tile_tracking_keys),  # Consolidated tile tracking
                'status_keys': 0,  # Not used in consolidated storage
                'result_keys': 0,  # Not used in consolidated storage
                'counter_keys': len(counter_keys),
                'orphaned_keys': len(orphaned_keys)
            },
            'current_counters': current_counters,
            'active_workers': {
                'count': len(active_workers),
                'workers': active_workers
            },
            'cleanup_info': {
                'orphaned_keys': orphaned_keys[:20],  # Show first 20 orphaned keys
                'total_orphaned': len(orphaned_keys),
                'cleanup_recommended': len(orphaned_keys) > 5,
                'storage_type': 'CONSOLIDATED'
            },
            'storage_type': 'Redis (Consolidated Key-Value Storage)',
            'all_keys': sorted(all_keys)[:50],
            'timestamp': time.time(),
            'timestamp_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        return jsonify(metadata), 200
        
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metadata/nuclear-cleanup', methods=['POST'])
def nuclear_cleanup():
    """NUCLEAR OPTION: Delete ALL non-essential keys and rebuild from jobs"""
    try:
        data = request.get_json() or {}
        if not data.get('confirm_nuclear', False):
            return jsonify({
                'error': 'Nuclear cleanup requires confirmation',
                'message': 'Add "confirm_nuclear": true to proceed',
                'warning': 'This will delete ALL Redis data except job definitions'
            }), 400
        
        logger.warning("NUCLEAR CLEANUP INITIATED!")
        
        # Step 1: Backup all job data
        job_backup = {}
        cursor = '0'
        while cursor != 0:
            cursor, keys = redis_client.client.scan(cursor=cursor, match='job:*', count=100)
            for key in keys:
                if ':tiles' not in key:  # Only backup job metadata, not tile data
                    try:
                        data = redis_client.get(key)
                        if data:
                            job_backup[key] = data
                            logger.info(f"Backed up: {key}")
                    except Exception as e:
                        logger.error(f"Failed to backup {key}: {e}")
        
        logger.info(f"Backed up {len(job_backup)} job records")
        
        # Step 2: FLUSH ALL REDIS DATA
        redis_client.client.flushdb()
        logger.warning("ALL REDIS DATA DELETED!")
        
        # Step 3: Restore job data
        restored = 0
        for key, data in job_backup.items():
            try:
                redis_client.set(key, data)
                restored += 1
            except Exception as e:
                logger.error(f"Failed to restore {key}: {e}")
        
        # Step 4: Rebuild essential counters
        redis_client.set('active_workers_count', '0')
        redis_client.set('active_workers', '[]')
        redis_client.set('tiles_total', '0')
        redis_client.set('tiles_processed', '0')
        
        logger.warning(f"NUCLEAR CLEANUP COMPLETE: Restored {restored} jobs, rebuilt counters")
        
        return jsonify({
            'message': 'NUCLEAR cleanup completed!',
            'jobs_restored': restored,
            'redis_keys_after': len(job_backup) + 4,  # jobs + 4 counters
            'warning': 'All processing history and results have been deleted',
            'storage_type': 'CONSOLIDATED'
        }), 200
        
    except Exception as e:
        logger.error(f"NUCLEAR cleanup failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/redis-test', methods=['GET'])
def debug_redis_test():
    """Test Redis connectivity and operations"""
    try:
        # Test basic connectivity
        ping_result = redis_client.ping()
        
        # Test write operation
        test_key = f"test_key_{int(time.time())}"
        redis_client.set(test_key, "test_value", ex=60)  # Expires in 60 seconds
        
        # Test read operation  
        read_result = redis_client.get(test_key)
        
        # Get database info
        db_size = redis_client.client.dbsize()
        all_keys = redis_client.client.keys('*')
        
        return jsonify({
            'redis_ping': ping_result,
            'test_write': test_key,
            'test_read': read_result,
            'database_size': db_size,
            'all_keys': all_keys[:20],  # First 20 keys
            'redis_config': {
                'host': REDIS_HOST,
                'port': REDIS_PORT,
                'connection_pool': str(redis_client.client.connection_pool.connection_kwargs)
            }
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'redis_config': {
                'host': REDIS_HOST,
                'port': REDIS_PORT
            }
        }), 500

@app.route('/metadata/clear', methods=['POST'])
def clear_old_jobs():
    """Clear old completed jobs from Redis - CONSOLIDATED VERSION"""
    try:
        data = request.get_json() or {}
        older_than_hours = data.get('older_than_hours', 24)  # Default: 24 hours
        
        cutoff_time = time.time() - (older_than_hours * 3600)
        cleared_jobs = []
        
        # Scan for job keys (consolidated storage)
        cursor = '0'
        job_keys = []
        while cursor != 0:
            cursor, keys = redis_client.client.scan(cursor=cursor, match='job:*', count=100)
            for key in keys:
                if ':tiles' not in key:  # Only main job keys
                    job_keys.append(key)
        
        logger.info(f"Found {len(job_keys)} job keys to check for cleanup")
        
        for job_key in job_keys:
            try:
                job_data_str = redis_client.get(job_key)
                if job_data_str:
                    job_data = json.loads(job_data_str)
                    created_at = job_data.get('timestamp', time.time())
                    
                    if created_at < cutoff_time:
                        job_id = job_data.get('job_id')
                        if job_id:
                            short_job_id = job_id[:8]
                            
                            # Delete job metadata and tile tracking (consolidated)
                            keys_to_delete = [
                                job_key,
                                f"job:{short_job_id}:tiles"
                            ]
                            
                            # Delete keys
                            for key in keys_to_delete:
                                redis_client.delete(key)
                            
                            cleared_jobs.append(short_job_id)
                            logger.info(f"Cleared old job: {short_job_id}")
                            
            except Exception as e:
                logger.error(f"Error clearing job {job_key}: {e}")
        
        logger.info(f"Cleanup completed. Cleared {len(cleared_jobs)} jobs")
        
        return jsonify({
            'message': f'Cleared {len(cleared_jobs)} old jobs',
            'cleared_jobs': cleared_jobs,
            'older_than_hours': older_than_hours,
            'storage_type': 'CONSOLIDATED'
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing jobs: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()
    
    # Cleanup producer
    try:
        producer_pool.cleanup()
    except Exception as e:
        logger.error(f"Error during producer cleanup: {e}")
    
    logger.info("Shutdown complete")
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration before starting"""
    logger.info("Validating configuration...")
    errors = []
    
    # Check Kafka
    if not KAFKA_BROKER:
        errors.append("KAFKA_BROKER not configured")
    else:
        try:
            test_producer = Producer({'bootstrap.servers': KAFKA_BROKER})
            test_producer.flush(timeout=5)
            logger.info("✓ Kafka connectivity test passed")
        except Exception as e:
            errors.append(f"Cannot connect to Kafka at {KAFKA_BROKER}: {str(e)}")
    
    # Check Redis
    if not REDIS_HOST:
        errors.append("REDIS_HOST not configured")
    else:
        try:
            if redis_client.ping():
                logger.info("✓ Redis connectivity test passed")
            else:
                errors.append("Redis ping failed")
        except Exception as e:
            errors.append(f"Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}: {str(e)}")
    
    # Check directories
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
        logger.info("✓ Upload and result directories ready")
    except Exception as e:
        errors.append(f"Cannot create directories: {str(e)}")
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        return False
    
    logger.info("✓ All configuration checks passed")
    return True

# ============================================================================
# CREATE DASHBOARD TEMPLATE - IMPROVED UI
# ============================================================================
# [Keep all the previous code until create_dashboard_template function, then replace with this:]

def create_dashboard_template():
    """Create dashboard.html template with improved UI"""
    dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Worker Dashboard - Distributed Image Processing</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            color: #888;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #999;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .workers-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .section-title {
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .worker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .worker-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .worker-id {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4ade80;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .worker-info {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .info-label {
            opacity: 0.8;
        }
        
        .info-value {
            font-weight: bold;
        }
        
        .no-workers {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        
        .no-workers-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .back-button {
            display: inline-block;
            background: white;
            color: #667eea;
            padding: 12px 30px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .back-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .system-status {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            justify-content: center;
        }
        
        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .status-healthy {
            background: #4ade80;
            color: white;
        }
        
        .status-warning {
            background: #fbbf24;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🖥 Worker Dashboard</h1>
            <p>Real-time monitoring of distributed image processing workers</p>
            <div class="system-status" id="systemStatus"></div>
            <div style="margin-top: 20px;">
                <a href="/" class="back-button">← Back to Upload</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Active Workers</h3>
                <div class="stat-value" id="activeWorkers">-</div>
                <div class="stat-label">Currently processing</div>
            </div>
            
            <div class="stat-card">
                <h3>Kafka Broker</h3>
                <div class="stat-value" style="font-size: 1.5em;" id="kafkaBroker">-</div>
                <div class="stat-label">Message broker</div>
            </div>
            
            <div class="stat-card">
                <h3>Operations</h3>
                <div class="stat-value" id="operations">-</div>
                <div class="stat-label">Available filters</div>
            </div>
            
            <div class="stat-card">
                <h3>System Status</h3>
                <div class="stat-value" style="font-size: 1.5em;" id="systemHealth">-</div>
                <div class="stat-label">Overall health</div>
            </div>
        </div>
        
        <div class="workers-section">
            <h2 class="section-title">👷 Worker Nodes</h2>
            <div class="worker-grid" id="workerGrid">
                <div class="no-workers">
                    <div class="no-workers-icon">⏳</div>
                    <p>Loading worker information...</p>
                </div>
            </div>
            <div class="refresh-info">
                🔄 Auto-refreshing every 3 seconds
            </div>
        </div>
    </div>
    
    <script>
        function formatTime(seconds) {
            if (seconds < 60) return Math.floor(seconds) + 's ago';
            if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
            return Math.floor(seconds / 3600) + 'h ago';
        }
        
        async function updateDashboard() {
            try {
                const response = await fetch('/dashboard/data');
                const data = await response.json();
                
                // Update stats
                document.getElementById('activeWorkers').textContent = data.active_workers;
                document.getElementById('kafkaBroker').textContent = data.kafka_broker.split(':')[0];
                document.getElementById('operations').textContent = data.operations_available;
                
                // Update system status
                const healthBadge = data.redis_connected ? 
                    '<span class="status-badge status-healthy">✓ Redis Online</span>' :
                    '<span class="status-badge status-warning">⚠ Redis Offline</span>';
                document.getElementById('systemStatus').innerHTML = healthBadge;
                document.getElementById('systemHealth').textContent = data.redis_connected ? '✓ Healthy' : '⚠ Degraded';
                document.getElementById('systemHealth').style.color = data.redis_connected ? '#4ade80' : '#fbbf24';
                
                // Update worker grid
                const workerGrid = document.getElementById('workerGrid');
                
                if (data.worker_list.length === 0) {
                    workerGrid.innerHTML = `
                        <div class="no-workers">
                            <div class="no-workers-icon">💤</div>
                            <p style="font-size: 1.2em; margin-bottom: 10px;">No active workers</p>
                            <p>Start worker nodes to begin processing</p>
                        </div>
                    `;
                } else {
                    workerGrid.innerHTML = data.worker_list.map(worker => `
                        <div class="worker-card">
                            <div class="worker-id">
                                <span class="status-indicator"></span>
                                ${worker.worker_id}
                            </div>
                            <div class="worker-info">
                                <div class="info-row">
                                    <span class="info-label">Status</span>
                                    <span class="info-value">${worker.status.toUpperCase()}</span>
                                </div>
                                <div class="info-row">
                                    <span class="info-label">Last Heartbeat</span>
                                    <span class="info-value">${formatTime(worker.last_seen_ago)}</span>
                                </div>
                                <div class="info-row">
                                    <span class="info-label">Timestamp</span>
                                    <span class="info-value">${new Date(worker.last_seen * 1000).toLocaleTimeString()}</span>
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error updating dashboard:', error);
                document.getElementById('systemHealth').textContent = '✗ Error';
                document.getElementById('systemHealth').style.color = '#ef4444';
            }
        }
        
        // Initial load
        updateDashboard();
        
        // Auto-refresh every 3 seconds
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>'''
    
    dashboard_path = os.path.join(TEMPLATES_FOLDER, 'dashboard.html')
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    logger.info(f"Dashboard template created at {dashboard_path}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("MASTER NODE - Distributed Image Processing Pipeline")
    logger.info(f"Instance ID: {INSTANCE_ID}")
    logger.info(f"Kafka API: Confluent Kafka")
    logger.info(f"Operations: {len(PROCESSING_OPERATIONS)}")
    logger.info(f"Kafka Broker: {KAFKA_BROKER}")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Partitions: 2 (as per requirements)")
    logger.info(f"Tile Size: {TILE_SIZE}x{TILE_SIZE}")
    logger.info(f"Max Tiles: {MAX_TILES}")
    logger.info(f"Batching: Enabled (batch size: {BATCH_SIZE})")
    logger.info("=" * 80)
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        exit(1)
    
    # Create dashboard template
    create_dashboard_template()
    
    # Initialize producer pool
    try:
        producer_pool.initialize()
        logger.info("Producer pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize producer pool: {e}")
        exit(1)
    
    # Start background threads
    logger.info("Starting background threads...")
    
    result_thread = Thread(target=listen_for_results, daemon=False, name="ResultListener")
    result_thread.start()
    logger.info("✓ Result listener thread started")
    
    heartbeat_monitor_thread = Thread(target=monitor_heartbeats, daemon=False, name="HeartbeatMonitor")
    heartbeat_monitor_thread.start()
    logger.info("✓ Heartbeat monitor thread started")
    
    heartbeat_checker_thread = Thread(target=heartbeat_checker, daemon=False, name="HeartbeatChecker")
    heartbeat_checker_thread.start()
    logger.info("✓ Heartbeat checker thread started")
    
    logger.info("=" * 80)
    logger.info("Master Node is ready!")
    logger.info("Starting Flask web server on 0.0.0.0:5000")
    logger.info("=" * 80)
    
    try:
        # In production, use Gunicorn or uWSGI instead
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        signal_handler(signal.SIGINT, None)
