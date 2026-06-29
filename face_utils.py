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
    CAMERA_INDEX,
    BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, ENABLE_PREPROCESSING
)

# ─── Singleton InsightFace ────────────────────────────────────────────────────
_insight_app = None
_clahe       = None    # CLAHE singleton
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


def _get_clahe():
    """Lazy-init CLAHE untuk normalisasi kontras adaptif."""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID
        )
    return _clahe


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


# ─── Preprocessing Pipeline (Noise Reduction) ────────────────────────────────

def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline 3-langkah untuk mengurangi noise kamera laptop:
      1. Bilateral Filter  → denoising sambil preserve tepi wajah
      2. CLAHE (Lab space)  → normalisasi kontras adaptif per-tile
      3. Unsharp Mask       → recovery detail halus setelah denoising
    Return frame BGR yang sudah bersih.
    """
    if not ENABLE_PREPROCESSING:
        return frame_bgr

    # 1. Bilateral filter — noise hilang, tepi wajah tetap tajam
    denoised = cv2.bilateralFilter(
        frame_bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE
    )

    # 2. CLAHE hanya pada kanal L (Lightness) di ruang warna Lab
    #    → mengatasi wajah terlalu gelap/terang akibat backlight
    lab     = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l_eq    = _get_clahe().apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_Lab2BGR)

    # 3. Unsharp Mask — pulihkan detail yang sedikit hilang setelah denoising
    blurred   = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened


def average_frames(frames: list) -> np.ndarray:
    """
    Rata-rata N frame berurutan untuk kurangi random temporal noise.
    Noise acak saling meniadakan saat di-average → gambar lebih bersih.
    frames: list of BGR np.ndarray
    """
    if not frames:
        raise ValueError("frames list kosong")
    if len(frames) == 1:
        return frames[0]
    stacked = np.stack(frames, axis=0).astype(np.float32)
    return np.clip(stacked.mean(axis=0), 0, 255).astype(np.uint8)


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
    Preprocessing + InsightFace detection + embedding.
    Frame diproses melalui pipeline noise reduction sebelum dianalisis.
    Return list face objects.
    """
    try:
        processed = preprocess_frame(frame_bgr)
        app       = _get_insight()
        faces     = app.get(processed)
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
