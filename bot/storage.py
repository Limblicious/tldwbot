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


@dataclass
class JobRecord:
    video_id: str
    user_id: int
    channel_id: int
    status: str  # "queued", "processing", "completed", "failed"
    progress: str  # Human-readable progress message
    estimated_time_sec: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
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
            # Job tracking table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    video_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    progress TEXT NOT NULL,
                    estimated_time_sec REAL NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_user_status
                ON jobs(user_id, status)
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

    def create_job(
        self,
        video_id: str,
        user_id: int,
        channel_id: int,
        estimated_time_sec: float,
        progress: str = "Queued"
    ) -> None:
        """Create a new job record."""
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (video_id, user_id, channel_id, status, progress, estimated_time_sec, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    status='queued',
                    progress=excluded.progress,
                    estimated_time_sec=excluded.estimated_time_sec,
                    created_at=excluded.created_at
                """,
                (video_id, user_id, channel_id, "queued", progress, estimated_time_sec, datetime.utcnow().isoformat()),
            )

    def update_job_status(
        self,
        video_id: str,
        status: str,
        progress: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> None:
        """Update job status and progress."""
        with self._lock, self._get_conn() as conn:
            if started_at and completed_at:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status=?, progress=?, started_at=?, completed_at=?
                    WHERE video_id=?
                    """,
                    (status, progress, started_at.isoformat(), completed_at.isoformat(), video_id),
                )
            elif started_at:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status=?, progress=?, started_at=?
                    WHERE video_id=?
                    """,
                    (status, progress, started_at.isoformat(), video_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status=?, progress=?
                    WHERE video_id=?
                    """,
                    (status, progress, video_id),
                )

    def get_user_jobs(self, user_id: int, limit: int = 5) -> list[JobRecord]:
        """Get recent jobs for a user."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT video_id, user_id, channel_id, status, progress, estimated_time_sec,
                       started_at, completed_at, created_at
                FROM jobs
                WHERE user_id=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

            return [
                JobRecord(
                    video_id=row["video_id"],
                    user_id=row["user_id"],
                    channel_id=row["channel_id"],
                    status=row["status"],
                    progress=row["progress"],
                    estimated_time_sec=row["estimated_time_sec"],
                    started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
                )
                for row in rows
            ]

    def cleanup_old_jobs(self, days: int = 7) -> None:
        """Clean up job records older than N days."""
        with self._lock, self._get_conn() as conn:
            cutoff = datetime.utcnow().timestamp() - (days * 86400)
            conn.execute(
                """
                DELETE FROM jobs
                WHERE created_at < datetime(?, 'unixepoch')
                """,
                (cutoff,),
            )

