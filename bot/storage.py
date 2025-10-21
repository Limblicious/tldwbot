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


@dataclass
class PerformanceMetric:
    video_id: str
    transcript_tokens: int
    num_chunks: int
    processing_time_seconds: float
    strategy: str  # "one_shot" or "hierarchical"
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    transcript_tokens INTEGER NOT NULL,
                    num_chunks INTEGER NOT NULL,
                    processing_time_seconds REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            # Index for faster lookups when calculating estimates
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_perf_strategy_tokens
                ON performance_metrics(strategy, transcript_tokens)
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

    def record_performance(
        self,
        video_id: str,
        transcript_tokens: int,
        num_chunks: int,
        processing_time_seconds: float,
        strategy: str,
    ) -> None:
        """Record performance metrics for adaptive estimation."""
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO performance_metrics
                (video_id, transcript_tokens, num_chunks, processing_time_seconds, strategy, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (video_id, transcript_tokens, num_chunks, processing_time_seconds, strategy, datetime.utcnow().isoformat()),
            )

    def get_adaptive_estimate(
        self,
        transcript_tokens: int,
        strategy: str,
        concurrency: int = 4,
    ) -> Optional[float]:
        """
        Get adaptive time estimate based on historical performance.

        Uses recent historical data from similar videos to estimate processing time.
        Falls back to None if insufficient data.
        """
        with self._lock, self._get_conn() as conn:
            # Get recent metrics for similar token counts (±30% range)
            token_min = int(transcript_tokens * 0.7)
            token_max = int(transcript_tokens * 1.3)

            rows = conn.execute(
                """
                SELECT processing_time_seconds, transcript_tokens, num_chunks
                FROM performance_metrics
                WHERE strategy = ?
                  AND transcript_tokens BETWEEN ? AND ?
                ORDER BY created_at DESC
                LIMIT 10
                """,
                (strategy, token_min, token_max),
            ).fetchall()

            if not rows or len(rows) < 3:
                # Not enough data for reliable estimate
                return None

            # Calculate average time per token from recent runs
            total_time = sum(row["processing_time_seconds"] for row in rows)
            total_tokens = sum(row["transcript_tokens"] for row in rows)

            if total_tokens == 0:
                return None

            # Time per token (average)
            time_per_token = total_time / total_tokens

            # Estimate for current transcript
            estimated_time = time_per_token * transcript_tokens

            # Add 20% buffer for variance
            return estimated_time * 1.2

