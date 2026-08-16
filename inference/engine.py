"""ONNX Runtime wrappers for the two-stage detect -> classify pipeline.

Both engines request CUDAExecutionProvider first and fall back to
CPUExecutionProvider automatically if it isn't available (works with the
plain `onnxruntime` CPU package as-is), gated on INFERENCE_USE_GPU.

HISTORY - the GPU hang, and what actually caused it: an earlier attempt at
GPU execution loaded CUDAExecutionProvider fine (both models reported
['CUDAExecutionProvider', 'CPUExecutionProvider'] and nvidia-smi showed
real utilization), but under real pipeline load - not a synthetic
single-call test - the inference thread intermittently hung: 100%+ CPU
across many threads, zero forward progress, no exception, for an
unpredictable duration, then it cleared on its own. That was blamed on the
hand-assembled CUDA/cuDNN/cuBLAS pins in requirements.txt. The real cause
was narrower: this host's GPU is Blackwell (sm_120), and onnxruntime-gpu
1.19.2 is built with CMAKE_CUDA_ARCHITECTURES="52;60;61;70;75;80", so the
newest GPU code in that wheel is compute_80 PTX. Every CUDA kernel
therefore had to be JIT-compiled by the driver at runtime - CPU-bound,
multi-threaded, silent, and eventually successful, which is exactly the
observed signature. requirements.txt now pins onnxruntime-gpu 1.28.0 (the
CUDA 13 build, which compiles native 120-real cubins) plus the exact
CUDA/cuDNN versions its own [cuda,cudnn] extras declare, so no JIT is
involved. See the comment block in requirements.txt for the full reasoning.
"""
import logging
import os

import cv2
import numpy as np
import onnxruntime as ort

# The CUDA/cuDNN shared objects come from the nvidia-* pip packages, which
# install under site-packages/nvidia/... rather than onto the system linker
# path. nvidia-container-toolkit injects only the driver (libcuda.so), not
# these, so without help onnxruntime cannot dlopen libcublasLt/libcudnn.
# preload_dlls() is onnxruntime's own supported way to resolve them out of
# the nvidia site-packages - preferred over hand-maintaining an
# LD_LIBRARY_PATH list, which silently rots whenever a CUDA major version
# renames or relocates its package directories, as CUDA 12 -> 13 did.
if hasattr(ort, 'preload_dlls'):
    try:
        ort.preload_dlls()
    except Exception as _e:  # never let library probing break a CPU-only run
        logging.getLogger(__name__).warning(f"ort.preload_dlls() failed: {_e}")

from app.config import (
    DETECT_INPUT_SIZE, CLASSIFY_INPUT_SIZE,
    DETECT_CONF_THRESHOLD, DETECT_IOU_THRESHOLD, INFERENCE_USE_GPU,
)
from inference.coco_classes import COCO_CLASSES
from inference.imagenet_classes import IMAGENET_CLASSES

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ONNX Runtime's CPUExecutionProvider defaults intra_op_num_threads to the
# host's full core count, PER SESSION - and this process creates two
# sessions (detect + classify). Horizontally scaling `inference` replicas
# (docker compose up --scale inference=N) then oversubscribes the same
# physical cores N times over: measured via bench/run_inference_replica_sweep.py
# BEFORE this fix, throughput stayed flat (~7 tiles/s) while p50 latency rose
# 120ms -> 903ms and p99 hit ~2s going from 1 to 8 replicas - scaling out made
# it strictly worse, the same oversubscription signature as the pre-fix
# OpenCV worker pool (cv2.setNumThreads(1), see worker/main.py). Capped to 1
# here for the same reason: scale-out should come from running more
# containers, not from each one also fanning a single tile across every
# host core.
INFERENCE_INTRA_OP_THREADS = int(os.getenv('INFERENCE_INTRA_OP_THREADS', 1))


def make_session(model_path: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    want_cuda = INFERENCE_USE_GPU and 'CUDAExecutionProvider' in available
    providers = (['CUDAExecutionProvider'] if want_cuda else []) + ['CPUExecutionProvider']

    options = ort.SessionOptions()
    options.intra_op_num_threads = INFERENCE_INTRA_OP_THREADS

    try:
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    except Exception as e:
        logger.warning(f"Failed to load {model_path} with {providers}, retrying CPU-only: {e}")
        session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
    logger.info(f"Loaded {model_path} - active providers: {session.get_providers()}, "
                f"intra_op_num_threads={INFERENCE_INTRA_OP_THREADS}")
    return session


def _letterbox(image, target_size=DETECT_INPUT_SIZE):
    """Resize preserving aspect ratio, pad to a square with grey (114).
    Returns the padded image plus the scale and (left, top) padding needed
    to map detections back into the original image's coordinate space."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = target_size - new_h, target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top


class DetectionEngine:
    """YOLOv8 ONNX export: single-head output (1, 4+num_classes, num_anchors)
    with box coords first (cx, cy, w, h in the letterboxed 640x640 space)
    then one confidence row per class - no NMS baked into this export, so
    postprocessing does thresholding + NMS itself."""

    def __init__(self, model_path):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self._batched_ok = self._probe_batch_support()

    def _probe_batch_support(self) -> bool:
        """The stock Ultralytics ONNX export (no --dynamic at export time)
        declares batch=1 in its I/O shapes AND bakes that into a Reshape
        node's literal target shape inside the detection head - patching
        the declared input/output shape to a symbolic dim does NOT fix it
        (verified: still throws mid-graph). models/yolov8n.onnx is now a
        re-export from the .pt weights with dynamic=True (ultralytics +
        torch), which fixes this at the source and genuinely batches - kept
        as a startup probe rather than a hardcoded assumption so a
        model swap (e.g. reverting to models/yolov8n_static_backup.onnx)
        degrades to the per-tile fallback automatically instead of
        crashing."""
        try:
            probe = np.zeros((2, 3, DETECT_INPUT_SIZE, DETECT_INPUT_SIZE), dtype=np.float32)
            self.session.run(None, {self.input_name: probe})
            return True
        except Exception:
            logger.warning(
                "Detection model does not support batch>1 (stock Ultralytics "
                "export has a hardcoded batch=1 Reshape in its head) - "
                "batching tiles into one forward pass, falling back to one "
                "call per tile within each accumulated batch."
            )
            return False

    def infer_batch(self, images: list) -> list:
        """images: list of BGR np.ndarray (any size, any dtype uint8).
        Returns one list of detections per image: [{class_name, confidence,
        box: [x1, y1, x2, y2]}] in that image's own original pixel coords.

        Dynamic batching (inference/main.py) still accumulates tiles by
        size-or-timeout and hands them here as one call - that scheduling
        benefit (fewer round trips, bounded latency) holds regardless of
        _batched_ok. Only the actual forward pass falls back to per-tile
        when the loaded model can't do better.
        """
        tensors, meta = [], []
        for image in images:
            padded, scale, left, top = _letterbox(image)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensors.append(rgb.transpose(2, 0, 1))
            meta.append((scale, left, top))

        if self._batched_ok:
            batch = np.stack(tensors)
            outputs = self.session.run(None, {self.input_name: batch})[0]  # (B, 84, N)
        else:
            outputs = np.stack([
                self.session.run(None, {self.input_name: t[np.newaxis, ...]})[0][0]
                for t in tensors
            ])

        return [
            self._postprocess(outputs[i], *meta[i])
            for i in range(len(images))
        ]

    def _postprocess(self, output, scale, left, top):
        # output: (4 + num_classes, num_anchors) -> (num_anchors, 4 + num_classes)
        preds = output.T
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confidences >= DETECT_CONF_THRESHOLD
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # (cx, cy, w, h) -> (x, y, w, h) for cv2.dnn.NMSBoxes
        boxes_xywh_cv = np.column_stack([
            boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
            boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
            boxes_xywh[:, 2],
            boxes_xywh[:, 3],
        ])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh_cv.tolist(), confidences.tolist(),
            DETECT_CONF_THRESHOLD, DETECT_IOU_THRESHOLD
        )
        if len(indices) == 0:
            return []

        detections = []
        for idx in np.array(indices).flatten():
            x, y, w, h = boxes_xywh_cv[idx]
            # undo letterbox: subtract padding, divide by the resize scale
            x1 = (x - left) / scale
            y1 = (y - top) / scale
            x2 = (x + w - left) / scale
            y2 = (y + h - top) / scale
            detections.append({
                'class_name': COCO_CLASSES[class_ids[idx]],
                'confidence': float(confidences[idx]),
                'box': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
        return detections


class ClassificationEngine:
    """MobileNetV2 (ONNX model zoo export), standard ImageNet preprocessing."""

    def __init__(self, model_path):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def infer_batch(self, crops: list) -> list:
        """crops: list of BGR np.ndarray (arbitrary size - typically a
        detection box crop). Returns [(label, confidence), ...], one per
        crop, in the same order."""
        valid = [(i, c) for i, c in enumerate(crops) if c.size > 0 and c.shape[0] > 0 and c.shape[1] > 0]
        if not valid:
            return [('unknown', 0.0)] * len(crops)

        batch = np.empty((len(valid), 3, CLASSIFY_INPUT_SIZE, CLASSIFY_INPUT_SIZE), dtype=np.float32)
        for batch_i, (_, crop) in enumerate(valid):
            resized = cv2.resize(crop, (CLASSIFY_INPUT_SIZE, CLASSIFY_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            batch[batch_i] = normalized.transpose(2, 0, 1)

        outputs = self.session.run(None, {self.input_name: batch})[0]  # (len(valid), 1000)

        results = [('unknown', 0.0)] * len(crops)
        for batch_i, (orig_i, _) in enumerate(valid):
            logits = outputs[batch_i]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            top = int(np.argmax(probs))
            results[orig_i] = (IMAGENET_CLASSES[top], float(probs[top]))
        return results
