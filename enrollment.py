"""
enrollment.py  – Video-Based Face Enrollment (InsightFace only)
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import Callable, Optional

import face_utils as fu
import database as db
from config import (
    ENROLL_FRAME_SKIP, ENROLL_MIN_FRAMES, ENROLL_MAX_FRAMES,
    ENROLL_VIDEO_DURATION, EMBEDDINGS_DIR, BLUR_THRESHOLD
)


class EnrollmentSession:
    def __init__(
        self,
        user_name: str,
        on_frame_callback:    Optional[Callable] = None,
        on_status_callback:   Optional[Callable] = None,
        on_complete_callback: Optional[Callable] = None,
    ):
        self.user_name    = user_name.strip()
        self.on_frame     = on_frame_callback
        self.on_status    = on_status_callback
        self.on_complete  = on_complete_callback
        self.running      = False
        self.good_embeddings: list = []
        self.blur_scores:     list = []

    def start(self):
        self._log(f"Memulai enrollment: {self.user_name}")
        self._log("Memuat model AI... (pertama kali ~30 detik)")

        # ── Buka kamera ──
        try:
            cap = fu.open_camera()
        except RuntimeError as e:
            self._complete(False, f"❌ {e}")
            return
        self._log("Kamera berhasil dibuka.")

        self.running = True
        frame_idx    = 0
        start_time   = time.time()

        try:
            while self.running:
                elapsed   = time.time() - start_time
                remaining = max(0, int(ENROLL_VIDEO_DURATION - elapsed))

                if elapsed >= ENROLL_VIDEO_DURATION:
                    self._log("Batas waktu tercapai.")
                    break
                if len(self.good_embeddings) >= ENROLL_MAX_FRAMES:
                    self._log("Frame maksimum terkumpul.")
                    break

                ret, frame_bgr = cap.read()
                if not ret:
                    continue

                frame_idx += 1
                gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                # ── Deteksi + embedding via InsightFace ──
                faces = fu.get_all_faces(frame_bgr)
                face  = fu.get_largest_face(faces)

                if face is None:
                    self._emit_frame(frame_bgr, len(self.good_embeddings),
                                     f"⏳ Arahkan wajah ke kamera — {remaining}s")
                    continue

                bbox = face.bbox   # [x1,y1,x2,y2]

                # ── Quality checks ──
                blur_ok = not fu.is_blurry(gray, BLUR_THRESHOLD)
                crop_ok = not fu.is_face_cropped(bbox, frame_bgr.shape)
                size_ok = not fu.is_face_too_small(bbox, frame_bgr.shape)

                annotated = frame_bgr.copy()

                if not (blur_ok and crop_ok and size_ok):
                    reasons = []
                    if not blur_ok: reasons.append("blur")
                    if not crop_ok: reasons.append("terpotong")
                    if not size_ok: reasons.append("terlalu kecil")
                    fu.draw_face_box_xyxy(annotated, bbox, ", ".join(reasons),
                                          0.0, is_unknown=True)
                    self._emit_frame(annotated, len(self.good_embeddings),
                                     f"⚠️ Skip: {', '.join(reasons)} — {remaining}s")
                    continue

                # ── Ambil setiap N frame ──
                if frame_idx % ENROLL_FRAME_SKIP != 0:
                    fu.draw_face_box_xyxy(annotated, bbox, "OK", 1.0)
                    self._emit_frame(annotated, len(self.good_embeddings),
                                     f"✅ {len(self.good_embeddings)} frame — {remaining}s")
                    continue

                # ── Ambil embedding ──
                embedding = fu.get_face_embedding(face)
                if embedding is None:
                    continue

                blur_score = fu.compute_blur_score(gray)
                self.good_embeddings.append(embedding)
                self.blur_scores.append(blur_score)

                fu.draw_face_box_xyxy(annotated, bbox,
                                      f"Frame {len(self.good_embeddings)}", 1.0)
                self._emit_frame(annotated, len(self.good_embeddings),
                                 f"✅ {len(self.good_embeddings)} frame valid — {remaining}s")

        except Exception as e:
            self._complete(False, f"❌ Error tidak terduga: {e}")
            return
        finally:
            cap.release()

        if len(self.good_embeddings) < ENROLL_MIN_FRAMES:
            self._complete(
                False,
                f"❌ Hanya {len(self.good_embeddings)} frame valid "
                f"(minimal {ENROLL_MIN_FRAMES}).\n"
                "Tips: pastikan pencahayaan cukup dan wajah tampak jelas."
            )
            return

        self._process_and_save()

    def stop(self):
        self.running = False

    def _select_best_frames(self):
        if not self.blur_scores:
            return self.good_embeddings
        pairs = sorted(zip(self.blur_scores, self.good_embeddings),
                       key=lambda x: x[0], reverse=True)
        top_n = max(ENROLL_MIN_FRAMES, len(pairs) // 2)
        return [e for _, e in pairs[:top_n]]

    def _process_and_save(self):
        self._log("Menghitung embedding representatif...")
        best   = self._select_best_frames()
        repr_e = fu.compute_representative_embedding(best)

        safe  = "".join(c if c.isalnum() else "_" for c in self.user_name)
        fname = f"{safe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.npy"
        fpath = os.path.join(EMBEDDINGS_DIR, fname)
        np.save(fpath, repr_e)

        uid = db.add_user(self.user_name, repr_e, fpath, len(self.good_embeddings))
        self._complete(
            True,
            f"✅ Enrollment berhasil!\n"
            f"Nama         : {self.user_name}\n"
            f"Frame valid  : {len(self.good_embeddings)}\n"
            f"Frame dipakai: {len(best)}\n"
            f"User ID      : {uid}"
        )

    def _emit_frame(self, frame, good, status):
        if self.on_frame:
            self.on_frame(frame, good, status)

    def _log(self, msg):
        if self.on_status:
            self.on_status(msg)

    def _complete(self, ok, msg):
        self.running = False
        if self.on_complete:
            self.on_complete(ok, msg)
