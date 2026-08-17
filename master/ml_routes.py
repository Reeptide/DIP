"""ML detection endpoints - separate from the OpenCV pipeline's routes in
master/app.py, mirroring its upload/status pattern but dispatching to
ml_tasks/ml_results instead of tasks/results (see app/config.py).
"""
import logging
import os
import uuid

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.config import (
    ALLOWED_EXTENSIONS, TILE_SIZE, MIN_IMAGE_SIZE, MAX_IMAGE_DIMENSION, BATCH_SIZE,
    DETECT_CONF_THRESHOLD, DETECT_IOU_THRESHOLD,
)
from app.imaging import split_image_into_tiles, probe_image_dimensions
from app.jobstore import create_job, get_job, get_tile_results, get_tile_latencies_ms
from app.metrics import jobs_created_total
from master.heartbeat import heartbeat_tracker
from master.ml_publisher import publish_ml_tasks
from master.rate_limit import rate_limit

logger = logging.getLogger(__name__)

ml_bp = Blueprint('ml', __name__)


def _cross_tile_nms(detections):
    """Suppress duplicate boxes across tile boundaries, per class - grouped
    by class_name because cv2.dnn.NMSBoxes only takes boxes+scores, so
    running it over all classes mixed together would also suppress a
    genuinely different object of another class overlapping the same
    region."""
    if not detections:
        return detections

    kept = []
    by_class = {}
    for det in detections:
        by_class.setdefault(det['class_name'], []).append(det)

    for dets in by_class.values():
        if len(dets) == 1:
            kept.append(dets[0])
            continue
        boxes_xywh = [
            [d['box'][0], d['box'][1], d['box'][2] - d['box'][0], d['box'][3] - d['box'][1]]
            for d in dets
        ]
        confidences = [d['confidence'] for d in dets]
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences, DETECT_CONF_THRESHOLD, DETECT_IOU_THRESHOLD
        )
        for idx in np.array(indices).flatten():
            kept.append(dets[idx])

    return kept


@ml_bp.route('/detect', methods=['POST'])
@rate_limit
def detect_image():
    """Upload an image for object detection + classification. Same
    validation and tiling as /upload; the difference is entirely in which
    topic tiles are dispatched to and what a 'result' means."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'Invalid file'}), 400

    if '.' not in file.filename:
        return jsonify({'error': 'File must have an extension'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    try:
        # heartbeat_tracker only reflects the OpenCV worker pool (only
        # worker/main.py sends heartbeats) - inference/main.py doesn't yet,
        # so there's no accurate liveness signal for ML workers to gate on
        # here. Reusing heartbeat_tracker anyway would just be wrong, not
        # merely approximate: an idle-but-alive OpenCV fleet with zero
        # inference workers would pass a check that means nothing for this
        # endpoint. Left as a known gap rather than a misleading check.
        active_workers = heartbeat_tracker.get_active_workers()

        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(filepath)

        # See master/app.py's /upload for why this runs before imread().
        with open(filepath, 'rb') as f:
            header = f.read(65536)
        probed = probe_image_dimensions(header)
        if probed and (probed[0] > MAX_IMAGE_DIMENSION or probed[1] > MAX_IMAGE_DIMENSION):
            os.remove(filepath)
            return jsonify({'error': f'Image too large. Maximum dimension: {MAX_IMAGE_DIMENSION}px'}), 400

        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to load image or corrupted file'}), 400

        height, width = image.shape[:2]

        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            os.remove(filepath)
            return jsonify({
                'error': f'Image too small. Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}, got: {width}x{height}'
            }), 400

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            os.remove(filepath)
            return jsonify({'error': f'Image too large. Maximum dimension: {MAX_IMAGE_DIMENSION}px'}), 400

        try:
            tiles = split_image_into_tiles(image, TILE_SIZE)
        except ValueError as e:
            os.remove(filepath)
            return jsonify({'error': str(e)}), 400

        expected_tiles = len(tiles)

        # image is already decoded into memory (line 63) and tiled from
        # there - the on-disk upload is never read again on this path.
        os.remove(filepath)

        create_job(
            job_id,
            original_filename=filename,
            operation='object_detection',
            original_width=width,
            original_height=height,
            expected_tiles=expected_tiles,
            active_workers=len(active_workers),
            tile_size=TILE_SIZE,
        )

        success = publish_ml_tasks(job_id, tiles, BATCH_SIZE)
        if not success:
            return jsonify({'error': 'Failed to publish ML tasks'}), 500

        jobs_created_total.labels(pipeline='ml').inc()

        logger.info(f"Detection job {job_id[:8]} created: {expected_tiles} tiles")

        return jsonify({
            'message': 'Image uploaded for detection',
            'job_id': job_id,
            'tiles_count': expected_tiles,
            'image_size': f"{width}x{height}"
        }), 200

    except Exception as e:
        logger.error(f"Error uploading image for detection: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@ml_bp.route('/detect/status/<job_id>', methods=['GET'])
def detect_status(job_id):
    """Progress while running; the full aggregated detection list, with
    boxes already in original-image coordinates, once complete."""
    try:
        job_data = get_job(job_id)
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        results_count = job_data['results_count']
        expected_tiles = job_data['expected_tiles'] or 1
        progress = int((results_count / expected_tiles) * 100)

        response = {
            'job_id': job_id,
            'status': job_data['status'],
            'progress': progress,
            'received_tiles': results_count,
            'expected_tiles': expected_tiles,
            'tile_latencies_s': [ms / 1000 for ms in get_tile_latencies_ms(job_id)],
        }

        if job_data['status'] in ('completed', 'ready_for_reconstruction', 'degraded'):
            tile_results = get_tile_results(job_id)
            all_detections = []
            for tile_info in tile_results.values():
                # Detection boxes are already offset into original-image
                # coordinates by inference/engine.py's letterbox undo -
                # add the tile's own (x, y) so a box is meaningful against
                # the WHOLE uploaded image, not just its tile.
                for det in tile_info.get('detections', []):
                    x1, y1, x2, y2 = det['box']
                    all_detections.append({
                        'class_name': det['class_name'],
                        'confidence': det['confidence'],
                        'classify_label': det.get('classify_label'),
                        'classify_confidence': det.get('classify_confidence'),
                        'box': [
                            round(x1 + tile_info['x'], 1), round(y1 + tile_info['y'], 1),
                            round(x2 + tile_info['x'], 1), round(y2 + tile_info['y'], 1),
                        ],
                    })

            # An object straddling a tile boundary produces one partial
            # detection per tile it overlaps (2-4 for a corner) - each tile
            # is inferred independently in inference/engine.py, with no
            # visibility into neighboring tiles. inference/engine.py's own
            # NMS only dedups boxes *within* one tile's output, so those
            # partial duplicates survive to here. Run the same NMS pass
            # again, now that every box is in shared original-image
            # coordinates, to collapse them into one.
            all_detections = _cross_tile_nms(all_detections)

            counts = {}
            for det in all_detections:
                counts[det['class_name']] = counts.get(det['class_name'], 0) + 1

            response['detections'] = all_detections
            response['detection_counts'] = counts
            response['total_detections'] = len(all_detections)

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error checking detection status: {str(e)}")
        return jsonify({'error': str(e)}), 500
