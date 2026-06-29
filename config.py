"""
config.py
Konfigurasi global untuk Sistem Absensi Berbasis Pengenalan Wajah
"""

import os

# ─── Path ───────────────────────────────────────────────────────────────────
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
DATA_DIR              = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR        = os.path.join(DATA_DIR, "embeddings")
ATTENDANCE_PHOTOS_DIR = os.path.join(DATA_DIR, "attendance_photos")
DB_PATH               = os.path.join(DATA_DIR, "face_db.sqlite")

# Pastikan direktori ada
for _dir in [DATA_DIR, EMBEDDINGS_DIR, ATTENDANCE_PHOTOS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─── Enrollment ───────────────────────────────────────────────────────────────
ENROLL_VIDEO_DURATION     = 10       # detik maksimum perekaman
ENROLL_FRAME_SKIP         = 5        # ambil frame setiap N frame
ENROLL_MIN_FRAMES         = 5        # minimal frame valid untuk enrollment
ENROLL_MAX_FRAMES         = 30       # maksimal frame untuk diproses
BLUR_THRESHOLD            = 100.0    # Laplacian variance; di bawah ini = blur
FACE_SIZE_MIN_RATIO       = 0.08     # wajah harus ≥ 8% lebar frame
FACE_MARGIN_RATIO         = 0.10     # margin toleransi wajah terpotong (10%)
FACE_MODEL                = "hog"    # "hog" (CPU cepat) atau "cnn" (GPU akurat)
ENCODING_NUM_JITTERS      = 1        # jitter augmentasi embedding (1=cepat)

# ─── Absensi ────────────────────────────────────────────────────────────────
COSINE_SIMILARITY_THRESHOLD = 0.55   # similarity ≥ ini → dianggap match
ATTENDANCE_COOLDOWN_HOURS   = 8      # jam cooldown per user per hari
RECOGNITION_SMOOTHING       = 5      # jumlah frame untuk stabilisasi prediksi
DISPLAY_FPS                 = 30     # target FPS tampilan webcam

# ─── Kamera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0             # Index 0 = webcam laptop (sudah diverifikasi)
CAMERA_BACKEND = None        # None = auto, atau cv2.CAP_DSHOW untuk Windows

# ─── GUI ────────────────────────────────────────────────────────────────────
APP_TITLE   = "Sistem Absensi Pengenalan Wajah"
APP_WIDTH   = 1100
APP_HEIGHT  = 700
THEME_BG    = "#0F1117"
THEME_PANEL = "#1A1D2E"
THEME_CARD  = "#252840"
THEME_ACCENT= "#4F6EF7"
THEME_GREEN = "#22C55E"
THEME_RED   = "#EF4444"
THEME_AMBER = "#F59E0B"
THEME_TEXT  = "#E2E8F0"
THEME_MUTED = "#64748B"
