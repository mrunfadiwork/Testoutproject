"""
face_utils.py – Multi-backend face detection + embedding

Backends auto-selected based on what is installed:
  1. InsightFace (buffalo_sc)   → 512-d ArcFace  [best quality]
  2. face_recognition (dlib)   → 128-d           [good, CPU-only]
  3. OpenCV Haar Cascade       → detection only  [always available, no recognition]

Works on Windows, macOS (Intel + Apple Silicon), Linux.
Install the preferred backend with:
  pip install insightface onnxruntime
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import threading
import logging
import os
import sys

from config import (
    BLUR_THRESHOLD, FACE_SIZE_MIN_RATIO, FACE_MARGIN_RATIO,
    CAMERA_INDEX,
    BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, ENABLE_PREPROCESSING,
)

logger = logging.getLogger(__name__)

# ─── Frozen-app model path override ─────────────────────────────────────────
# When packaged by PyInstaller, sys._MEIPASS is the temp extraction directory.
# We tell InsightFace to look for models there instead of ~/.insightface so the
# bundled buffalo_sc models are used without any network access.
if getattr(sys, "frozen", False):
    _bundled_insightface = os.path.join(sys._MEIPASS, "insightface_models")
    os.environ.setdefault("INSIGHTFACE_HOME", _bundled_insightface)
    logger.info(f"Frozen mode: INSIGHTFACE_HOME → {_bundled_insightface}")



# ─── Unified Face Result ─────────────────────────────────────────────────────

class FaceResult:
    """
    Unified wrapper returned by all backends.
    Mimics InsightFace's face object interface (.bbox, .embedding)
    so all existing code works without changes.
    """
    __slots__ = ("bbox", "embedding")

    def __init__(self, bbox, embedding: Optional[np.ndarray]):
        self.bbox      = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.embedding = embedding                           # np.ndarray or None


# ─── Backend Selection ───────────────────────────────────────────────────────

_BACKEND: Optional[str] = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> str:
    """
    Auto-detect and cache the best available backend.
    Priority: insightface > face_recognition > opencv
    """
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                try:
                    from insightface.app import FaceAnalysis  # noqa: F401
                    _BACKEND = "insightface"
                    logger.info("Face backend: InsightFace (512-d ArcFace)")
                except ImportError:
                    try:
                        import face_recognition  # noqa: F401
                        _BACKEND = "face_recognition"
                        logger.warning(
                            "InsightFace not found – falling back to face_recognition (128-d dlib).\n"
                            "For best results: pip install insightface onnxruntime"
                        )
                    except ImportError:
                        _BACKEND = "opencv"
                        logger.warning(
                            "InsightFace and face_recognition not found.\n"
                            "Falling back to OpenCV Haar (detection only – recognition disabled).\n"
                            "Install: pip install insightface onnxruntime"
                        )
    return _BACKEND


# ─── InsightFace Singleton ───────────────────────────────────────────────────

_insight_app  = None
_insight_lock = threading.Lock()


def _get_insight():
    global _insight_app
    if _insight_app is None:
        with _insight_lock:
            if _insight_app is None:
                from insightface.app import FaceAnalysis
                # Auto-select best ONNX execution provider for this device
                try:
                    import onnxruntime as ort
                    available = ort.get_available_providers()
                    if "CUDAExecutionProvider" in available:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    elif "CoreMLExecutionProvider" in available:       # Apple Silicon
                        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                    else:
                        providers = ["CPUExecutionProvider"]
                except Exception:
                    providers = ["CPUExecutionProvider"]

                logger.info(f"InsightFace ONNX providers: {providers}")
                app = FaceAnalysis(name="buffalo_sc", providers=providers)
                app.prepare(ctx_id=0, det_size=(640, 640))
                _insight_app = app
                logger.info("InsightFace model loaded.")
    return _insight_app


# ─── OpenCV Haar Cascade Singleton ──────────────────────────────────────────

_haar_cascade = None


def _get_haar():
    global _haar_cascade
    if _haar_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(path)
    return _haar_cascade


# ─── Camera ──────────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    """
    Open the camera. Cross-platform (Windows, macOS, Linux).
    Raises RuntimeError if no camera can be opened.
    """
    # macOS: set env var before VideoCapture so OpenCV does not try to request
    # camera permission from within a background thread (which fails on macOS).
    if sys.platform == "darwin":
        os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        for alt_idx in [1, 2]:
            logger.warning(f"Camera index {CAMERA_INDEX} failed – trying index {alt_idx}...")
            cap = cv2.VideoCapture(alt_idx)
            if cap.isOpened():
                logger.info(f"Camera opened on fallback index {alt_idx}.")
                break
        else:
            raise RuntimeError(
                f"Kamera index {CAMERA_INDEX} tidak bisa dibuka.\n"
                "Pastikan webcam tidak dipakai aplikasi lain (Zoom, Teams, dll)."
            )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm-up: discard initial dark / blank frames common on many devices
    logger.info("Camera warm-up: discarding initial frames...")
    for _ in range(10):
        cap.read()
    logger.info("Camera ready.")
    return cap


# ─── Quality Filters ─────────────────────────────────────────────────────────

def is_blurry(frame_gray: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    return cv2.Laplacian(frame_gray, cv2.CV_64F).var() < threshold


def compute_blur_score(frame_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())


def is_face_cropped(bbox_xyxy: Tuple, frame_shape: Tuple,
                    margin: float = FACE_MARGIN_RATIO) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = frame_shape[:2]
    mx, my = int(w * margin), int(h * margin)
    return x1 < mx or y1 < my or x2 > w - mx or y2 > h - my


def is_face_too_small(bbox_xyxy: Tuple, frame_shape: Tuple,
                      min_ratio: float = FACE_SIZE_MIN_RATIO) -> bool:
    x1, _, x2, _ = [int(v) for v in bbox_xyxy]
    return ((x2 - x1) / frame_shape[1]) < min_ratio


# ─── Frame Utilities ─────────────────────────────────────────────────────────

def average_frames(frames: list) -> np.ndarray:
    """
    Average a list of BGR frames to reduce camera sensor noise.
    Returns a uint8 BGR image with the same shape as the input frames.
    """
    if not frames:
        raise ValueError("average_frames: empty frame list")
    if len(frames) == 1:
        return frames[0].copy()
    stacked = np.stack(frames, axis=0).astype(np.float32)
    averaged = stacked.mean(axis=0)
    return np.clip(averaged, 0, 255).astype(np.uint8)


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Optional preprocessing pipeline for noisy / low-light cameras:
      1. Bilateral filter  – denoising while preserving face edges
      2. CLAHE             – adaptive contrast normalisation per channel
    Controlled by ENABLE_PREPROCESSING in config.py.
    """
    if not ENABLE_PREPROCESSING:
        return frame_bgr

    # Bilateral denoising (applied in BGR space)
    denoised = cv2.bilateralFilter(
        frame_bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    # CLAHE per channel in LAB colour space (avoids colour shift)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─── Per-backend Detectors ───────────────────────────────────────────────────

def _faces_insightface(frame_bgr: np.ndarray) -> list:
    """InsightFace backend – returns native InsightFace face objects."""
    try:
        app   = _get_insight()
        faces = app.get(frame_bgr)
        return faces if faces else []
    except Exception as e:
        logger.error(f"InsightFace detection error: {e}", exc_info=True)
        return []


def _faces_face_recognition(frame_bgr: np.ndarray) -> list:
    """face_recognition (dlib HOG) backend – returns FaceResult objects."""
    try:
        import face_recognition
        rgb       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)
        results   = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            results.append(FaceResult(
                bbox=[left, top, right, bottom],
                embedding=enc.astype(np.float32)
            ))
        return results
    except Exception as e:
        logger.error(f"face_recognition error: {e}", exc_info=True)
        return []


def _faces_opencv(frame_bgr: np.ndarray) -> list:
    """OpenCV Haar cascade backend – detection only, no embeddings."""
    try:
        cascade    = _get_haar()
        gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces_rect = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        results = []
        if len(faces_rect):
            for (x, y, w, h) in faces_rect:
                results.append(FaceResult(
                    bbox=[x, y, x + w, y + h],
                    embedding=None   # no model → no embedding
                ))
        return results
    except Exception as e:
        logger.error(f"OpenCV Haar error: {e}", exc_info=True)
        return []


# ─── Public Detection API ────────────────────────────────────────────────────

def get_all_faces(frame_bgr: np.ndarray) -> list:
    """
    Detect all faces using the best available backend.
    Returns a list of face objects, each with .bbox [x1,y1,x2,y2]
    and .embedding (np.ndarray or None).
    """
    backend = get_backend()
    if backend == "insightface":
        return _faces_insightface(frame_bgr)
    elif backend == "face_recognition":
        return _faces_face_recognition(frame_bgr)
    else:
        return _faces_opencv(frame_bgr)


def get_largest_face(faces: list):
    """Select the largest face (closest to camera)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def get_face_embedding(face) -> Optional[np.ndarray]:
    """Extract embedding from a face object (works for all backends)."""
    if face is None or face.embedding is None:
        return None
    return face.embedding.astype(np.float32)


def compute_representative_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean embedding, L2-normalized."""
    stack    = np.array(embeddings, dtype=np.float32)
    mean_emb = stack.mean(axis=0)
    norm     = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 0 else mean_emb


# ─── Similarity & Matching ───────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_best_match(
    query_embedding: np.ndarray,
    database: dict,
    threshold: float
) -> Tuple[Optional[int], str, float]:
    best_id, best_name, best_sim = None, "Unknown", -1.0
    for uid, data in database.items():
        stored = data["embedding"]
        # Skip embeddings from a different backend (different dimensions)
        if stored.shape != query_embedding.shape:
            logger.warning(
                f"Embedding dimension mismatch for '{data['name']}': "
                f"stored={stored.shape}, query={query_embedding.shape}. "
                "Re-enroll this user on this device."
            )
            continue
        sim = cosine_similarity(query_embedding, stored)
        if sim > best_sim:
            best_sim, best_id, best_name = sim, uid, data["name"]
    if best_sim >= threshold:
        return best_id, best_name, best_sim
    return None, "Unknown", best_sim


# ─── Drawing ─────────────────────────────────────────────────────────────────

def draw_face_box_xyxy(
    frame: np.ndarray,
    bbox,                    # iterable [x1,y1,x2,y2]
    label: str,
    similarity: float,
    is_unknown: bool = False
):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (34, 197, 94) if not is_unknown else (68, 68, 239)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_text = f"{label} ({similarity:.2f})" if not is_unknown else label
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y2 - th - 12), (x1 + tw + 10, y2), color, -1)
    cv2.putText(frame, label_text, (x1 + 5, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
