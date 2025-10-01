"""SQLite storage utilities for caching transcripts and summaries."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TranscriptRecord:
    video_id: str
    source: str
    text: str
    created_at: datetime


@dataclass
class SummaryRecord:
    video_id: str
    prompt_hash: str
    model: str
    summary: str
    created_at: datetime


class Storage:
    """Thread-safe SQLite wrapper for caching transcripts and summaries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    video_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    video_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (video_id, prompt_hash, model)
                )
                """
            )

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_transcript(self, video_id: str, source: str, text: str) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO transcripts (video_id, source, text, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    source=excluded.source,
                    text=excluded.text,
                    created_at=excluded.created_at
                """,
                (video_id, source, text, datetime.utcnow().isoformat()),
            )

    def get_transcript(self, video_id: str) -> Optional[TranscriptRecord]:
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT video_id, source, text, created_at FROM transcripts WHERE video_id=?",
                (video_id,),
            ).fetchone()
            if row:
                return TranscriptRecord(
                    video_id=row["video_id"],
                    source=row["source"],
                    text=row["text"],
                    created_at=datetime.fromisoformat(row["created_at"])
                    if isinstance(row["created_at"], str)
                    else row["created_at"],
                )
            return None

    def upsert_summary(self, video_id: str, prompt_hash: str, model: str, summary: str) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO summaries (video_id, prompt_hash, model, summary, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id, prompt_hash, model) DO UPDATE SET
                    summary=excluded.summary,
                    created_at=excluded.created_at
                """,
                (video_id, prompt_hash, model, summary, datetime.utcnow().isoformat()),
            )

    def get_summary(self, video_id: str, prompt_hash: str, model: str) -> Optional[SummaryRecord]:
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT video_id, prompt_hash, model, summary, created_at
                FROM summaries
                WHERE video_id=? AND prompt_hash=? AND model=?
                """,
                (video_id, prompt_hash, model),
            ).fetchone()
            if row:
                return SummaryRecord(
                    video_id=row["video_id"],
                    prompt_hash=row["prompt_hash"],
                    model=row["model"],
                    summary=row["summary"],
                    created_at=datetime.fromisoformat(row["created_at"])
                    if isinstance(row["created_at"], str)
                    else row["created_at"],
                )
            return None

