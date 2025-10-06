"""Transcript acquisition pipeline for YouTube videos."""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from .storage import Storage

logger = logging.getLogger(__name__)


YOUTUBE_ID_REGEX = re.compile(
    r"(?:v=|\/)([0-9A-Za-z_-]{11})",
    re.IGNORECASE,
)


@dataclass
class TranscriptResult:
    text: str
    source: str
    language: Optional[str] = None


class TranscriptError(RuntimeError):
    """Raised when no transcript could be acquired."""



def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract the YouTube video ID from a URL or return the ID as-is."""
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", url_or_id):
        return url_or_id
    match = YOUTUBE_ID_REGEX.search(url_or_id)
    if match:
        return match.group(1)
    return None


def get_cached_transcript(
    storage: Storage, video_id: str, force_refresh: bool
) -> Optional[TranscriptResult]:
    if force_refresh:
        return None
    record = storage.get_transcript(video_id)
    if record:
        logger.info("Transcript cache hit for %s via %s", video_id, record.source)
        return TranscriptResult(text=record.text, source=record.source)
    return None


def parse_vtt(vtt_content: str) -> str:
    """Parse VTT subtitle format and extract plain text."""
    lines = []
    for line in vtt_content.split('\n'):
        line = line.strip()
        # Skip WEBVTT header, timestamp lines, and empty lines
        if not line or line.startswith('WEBVTT') or '-->' in line or line.isdigit():
            continue
        # Skip metadata lines
        if line.startswith('Kind:') or line.startswith('Language:'):
            continue
        lines.append(line)
    return '\n'.join(lines)


def fetch_transcript(url: str) -> Optional[str]:
    """
    Fetch transcript using yt-dlp subprocess.

    First tries manual subtitles, then auto-generated subtitles.
    Returns plain text or None if no subtitles available.
    """
    # Try manual subtitles first
    cmd = [
        'yt-dlp',
        '--skip-download',
        '--write-subs',
        '--sub-lang', 'en',
        '--sub-format', 'vtt',
        '-o', '-',
        url
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout:
            text = parse_vtt(result.stdout)
            if text.strip():
                logger.info("Fetched manual subtitles for %s", url)
                return text

        # Check if stderr indicates no subtitles
        if 'no subtitles' in result.stderr.lower():
            logger.info("No manual subtitles, trying auto-generated for %s", url)
    except subprocess.TimeoutExpired:
        logger.error("Timeout fetching manual subtitles for %s", url)
    except Exception as exc:
        logger.error("Error fetching manual subtitles for %s: %s", url, exc)

    # Try auto-generated subtitles
    cmd_auto = [
        'yt-dlp',
        '--skip-download',
        '--write-auto-sub',
        '--sub-lang', 'en',
        '--sub-format', 'vtt',
        '-o', '-',
        url
    ]

    try:
        result = subprocess.run(
            cmd_auto,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout:
            text = parse_vtt(result.stdout)
            if text.strip():
                logger.info("Fetched auto-generated subtitles for %s", url)
                return text

    except subprocess.TimeoutExpired:
        logger.error("Timeout fetching auto-generated subtitles for %s", url)
    except Exception as exc:
        logger.error("Error fetching auto-generated subtitles for %s: %s", url, exc)

    return None


def fetch_captions(video_id: str) -> Optional[TranscriptResult]:
    """Fetch captions using yt-dlp subprocess."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    text = fetch_transcript(url)

    if text:
        return TranscriptResult(text=text, source="yt-dlp", language="en")

    logger.error("No captions available for video %s", video_id)
    return None


def obtain_transcript(
    storage: Storage,
    video_id: str,
    *,
    force_refresh: bool,
    use_local_whisper: bool = False,
    whisper_model_size: str = "",
    openai_client: Optional[object] = None,
) -> TranscriptResult:
    """
    Obtain transcript for a YouTube video.

    Only fetches captions via yt-dlp subprocess. Does not fall back to Whisper.
    Raises TranscriptError if captions are unavailable.
    """
    cached = get_cached_transcript(storage, video_id, force_refresh)
    if cached:
        return cached

    # Fetch captions only - no Whisper fallback
    captions = fetch_captions(video_id)
    if captions:
        storage.upsert_transcript(video_id, captions.source, captions.text)
        return captions

    # No captions available
    logger.error("No captions available for video %s", video_id)
    raise TranscriptError("No captions available for this video.")
