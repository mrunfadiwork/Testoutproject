"""
database.py
Handler SQLite untuk menyimpan data user, embedding, dan absensi.
"""

import sqlite3
import numpy as np
import json
import os
from datetime import datetime
from config import DB_PATH


def get_conn():
    """Buat koneksi SQLite dengan row_factory untuk akses kolom by-name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Inisialisasi tabel database jika belum ada."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                registered_at TEXT NOT NULL,
                embedding_path TEXT,
                frame_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                user_name   TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                similarity  REAL NOT NULL,
                photo_path  TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        conn.commit()


# ─── User CRUD ───────────────────────────────────────────────────────────────

def add_user(name: str, embedding: np.ndarray, embedding_path: str, frame_count: int) -> int:
    """Tambah user baru dan simpan embedding. Return user_id."""
    now = datetime.now().isoformat()
    with get_conn() as conn:
        # Jika sudah ada, update embedding-nya
        existing = conn.execute("SELECT id FROM users WHERE name=?", (name,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE users SET registered_at=?, embedding_path=?, frame_count=? WHERE name=?",
                (now, embedding_path, frame_count, name)
            )
            conn.commit()
            return existing["id"]
        else:
            cur = conn.execute(
                "INSERT INTO users (name, registered_at, embedding_path, frame_count) VALUES (?,?,?,?)",
                (name, now, embedding_path, frame_count)
            )
            conn.commit()
            return cur.lastrowid


def get_all_users():
    """Ambil semua user terdaftar."""
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users ORDER BY name").fetchall()


def delete_user(user_id: int):
    """Hapus user dan riwayat absensinya."""
    with get_conn() as conn:
        conn.execute("DELETE FROM attendance WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()


def load_all_embeddings() -> dict:
    """
    Muat semua embedding dari file .npy.
    Return dict: {user_id: {"name": str, "embedding": np.ndarray}}
    """
    result = {}
    with get_conn() as conn:
        users = conn.execute("SELECT id, name, embedding_path FROM users").fetchall()
    for user in users:
        path = user["embedding_path"]
        if path and os.path.exists(path):
            emb = np.load(path)
            result[user["id"]] = {
                "name": user["name"],
                "embedding": emb
            }
    return result


# ─── Attendance ───────────────────────────────────────────────────────────────

def add_attendance(user_id: int, user_name: str, similarity: float, photo_path: str) -> int:
    """Catat absensi baru."""
    now = datetime.now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO attendance (user_id, user_name, timestamp, date, similarity, photo_path) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, user_name, now.isoformat(), now.date().isoformat(), similarity, photo_path)
        )
        conn.commit()
        return cur.lastrowid


def check_already_attended(user_id: int, cooldown_hours: int = 8) -> bool:
    """
    Cek apakah user sudah absen dalam X jam terakhir.
    Return True jika sudah (cooldown aktif).
    """
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(hours=cooldown_hours)).isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM attendance WHERE user_id=? AND timestamp > ? LIMIT 1",
            (user_id, cutoff)
        ).fetchone()
    return row is not None


def get_attendance_today():
    """Ambil semua absensi hari ini."""
    today = datetime.now().date().isoformat()
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM attendance WHERE date=? ORDER BY timestamp DESC",
            (today,)
        ).fetchall()


def get_attendance_all():
    """Ambil semua riwayat absensi."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM attendance ORDER BY timestamp DESC"
        ).fetchall()


def get_attendance_stats():
    """Statistik ringkasan absensi."""
    today = datetime.now().date().isoformat()
    with get_conn() as conn:
        total_users   = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        today_count   = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
        total_records = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    return {
        "total_users": total_users,
        "today_count": today_count,
        "total_records": total_records
    }
