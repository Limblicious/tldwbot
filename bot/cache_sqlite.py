# bot/cache_sqlite.py

import os
import sqlite3
import threading
import time

_lock = threading.Lock()
_conn = None


def _get_conn(db_path: str):
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS captions (
                video_id TEXT PRIMARY KEY,
                text     TEXT NOT NULL,
                source   TEXT NOT NULL,
                fetched_at INTEGER NOT NULL
            )
        """)
        _conn.commit()
    return _conn


def get(video_id: str, ttl: int, db_path: str) -> tuple[str, str] | None:
    now = int(time.time())
    with _lock:
        c = _get_conn(db_path).execute("SELECT text, source, fetched_at FROM captions WHERE video_id = ?", (video_id,))
        row = c.fetchone()
        if not row:
            return None
        text, source, fetched_at = row
        if now - fetched_at > ttl:
            # stale -> treat as miss (do not delete; can be refreshed)
            return None
        return text, source


def set(video_id: str, text: str, source: str, db_path: str) -> None:
    now = int(time.time())
    with _lock:
        _get_conn(db_path).execute(
            "INSERT INTO captions(video_id, text, source, fetched_at) VALUES(?,?,?,?) "
            "ON CONFLICT(video_id) DO UPDATE SET text=excluded.text, source=excluded.source, fetched_at=excluded.fetched_at",
            (video_id, text, source, now))
        _get_conn(db_path).commit()
