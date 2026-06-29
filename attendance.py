"""
attendance.py  – Absensi Real-Time (InsightFace only)
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Callable, Optional
from collections import deque, Counter

import face_utils as fu
import database as db
from config import (
    COSINE_SIMILARITY_THRESHOLD, ATTENDANCE_COOLDOWN_HOURS,
    RECOGNITION_SMOOTHING, ATTENDANCE_PHOTOS_DIR
)


class AttendanceSession:
    def __init__(
        self,
        on_frame_callback:      Optional[Callable] = None,
        on_attendance_callback: Optional[Callable] = None,
        on_status_callback:     Optional[Callable] = None,
        on_stop_callback:       Optional[Callable] = None,
    ):
        self.on_frame      = on_frame_callback
        self.on_attendance = on_attendance_callback
        self.on_status     = on_status_callback
        self.on_stop       = on_stop_callback
        self.running       = False
        self.face_db       = {}
        self._pred_buffer  = deque(maxlen=RECOGNITION_SMOOTHING)

    def start(self):
        self._log("Memuat database wajah...")
        self.face_db = db.load_all_embeddings()
        count = len(self.face_db)
        if count == 0:
            self._log("⚠️ Database kosong. Lakukan enrollment terlebih dahulu.")
        else:
            self._log(f"Database: {count} wajah terdaftar.")

        self._log("Memuat model AI...")

        # ── Buka kamera ──
        try:
            cap = fu.open_camera()
        except RuntimeError as e:
            self._log(f"❌ {e}")
            if self.on_stop:
                self.on_stop()
            return
        self._log("Kamera berhasil dibuka.")

        self.running    = True
        process_every   = 3
        frame_count     = 0
        last_detections = []

        try:
            while self.running:
                ret, frame_bgr = cap.read()
                if not ret:
                    continue

                frame_count += 1
                annotated    = frame_bgr.copy()

                if frame_count % process_every == 0:
                    faces           = fu.get_all_faces(frame_bgr)
                    last_detections = []

                    for face in faces:
                        bbox      = face.bbox
                        embedding = fu.get_face_embedding(face)

                        if embedding is None:
                            last_detections.append((bbox, "?", 0.0, True))
                            continue

                        uid, name, sim = fu.find_best_match(
                            embedding, self.face_db, COSINE_SIMILARITY_THRESHOLD
                        )
                        is_unknown = (uid is None)

                        self._pred_buffer.append((uid, name, sim))
                        s_uid, s_name, s_sim = self._get_smoothed()

                        last_detections.append((bbox, s_name, s_sim, is_unknown))

                        if not is_unknown and s_uid is not None:
                            if not db.check_already_attended(
                                    s_uid, ATTENDANCE_COOLDOWN_HOURS):
                                self._record_attendance(
                                    s_uid, s_name, s_sim, frame_bgr)

                for bbox, name, sim, is_unknown in last_detections:
                    fu.draw_face_box_xyxy(annotated, bbox, name, sim,
                                          is_unknown=is_unknown)

                self._draw_overlay(annotated, len(last_detections))

                if self.on_frame:
                    self.on_frame(annotated)

        except Exception as e:
            self._log(f"❌ Error: {e}")
        finally:
            cap.release()
            if self.on_stop:
                self.on_stop()

    def stop(self):
        self.running = False

    def reload_database(self):
        self.face_db = db.load_all_embeddings()
        self._log(f"Database dimuat ulang: {len(self.face_db)} wajah.")

    def _get_smoothed(self):
        if not self._pred_buffer:
            return None, "Unknown", 0.0
        uid_votes = [x[0] for x in self._pred_buffer if x[0] is not None]
        if not uid_votes:
            return self._pred_buffer[-1]
        mcu     = Counter(uid_votes).most_common(1)[0][0]
        matches = [(u, n, s) for u, n, s in self._pred_buffer if u == mcu]
        if matches:
            uid, name, _ = matches[-1]
            avg_sim = float(np.mean([s for _, _, s in matches]))
            return uid, name, avg_sim
        return self._pred_buffer[-1]

    def _record_attendance(self, uid, name, sim, frame_bgr):
        ts_str     = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe       = "".join(c if c.isalnum() else "_" for c in name)
        photo_path = os.path.join(ATTENDANCE_PHOTOS_DIR, f"{safe}_{ts_str}.jpg")
        cv2.imwrite(photo_path, frame_bgr)
        db.add_attendance(uid, name, sim, photo_path)
        self._log(f"✅ HADIR: {name} | Sim: {sim:.3f} | "
                  f"{datetime.now().strftime('%H:%M:%S')}")
        if self.on_attendance:
            self.on_attendance(name, sim, datetime.now())

    def _draw_overlay(self, frame, face_count):
        now_str  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        db_count = len(self.face_db)
        cv2.rectangle(frame, (0, 0), (340, 22), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"DB: {db_count} | Detected: {face_count} | {now_str}",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def _log(self, msg):
        if self.on_status:
            self.on_status(msg)
