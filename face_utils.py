"""
face_utils.py  – InsightFace ONLY (deteksi + embedding dalam satu pipeline)
Kamera: langsung pakai index dari config.py (sudah diverifikasi = index 0)
InsightFace buffalo_sc sudah include:
  - SCRFD face detector
  - ArcFace 512-d embedding
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import threading

from config import (
    BLUR_THRESHOLD, FACE_SIZE_MIN_RATIO, FACE_MARGIN_RATIO,
    CAMERA_INDEX
)

# ─── Singleton InsightFace ────────────────────────────────────────────────────
_insight_app = None
_lock        = threading.Lock()


def _get_insight():
    global _insight_app
    if _insight_app is None:
        with _lock:
            if _insight_app is None:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CPUExecutionProvider"]
                )
                app.prepare(ctx_id=0, det_size=(320, 320))
                _insight_app = app
    return _insight_app


# ─── Camera ──────────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    """
    Buka kamera di index yang sudah diverifikasi (CAMERA_INDEX dari config).
    Return VideoCapture object. Caller wajib panggil .release() setelahnya.
    Raise RuntimeError jika kamera tidak bisa dibuka.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Kamera index {CAMERA_INDEX} tidak bisa dibuka.\n"
            "Pastikan webcam tidak dipakai aplikasi lain (Zoom, Teams, dll)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
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


# ─── Detection + Embedding via InsightFace ───────────────────────────────────

def get_all_faces(frame_bgr: np.ndarray) -> list:
    """
    Jalankan InsightFace pada frame BGR.
    Return list face objects, masing-masing punya:
      .bbox  → np.array [x1,y1,x2,y2]
      .embedding → np.array (512,) atau None
    """
    try:
        app   = _get_insight()
        faces = app.get(frame_bgr)
        return faces if faces else []
    except Exception:
        return []


def get_largest_face(faces: list):
    """Pilih wajah terbesar (paling dekat kamera)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))


def get_face_embedding(face) -> Optional[np.ndarray]:
    """Ambil embedding dari face object InsightFace."""
    if face is None or face.embedding is None:
        return None
    emb = face.embedding.astype(np.float32)
    return emb


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
        sim = cosine_similarity(query_embedding, data["embedding"])
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
