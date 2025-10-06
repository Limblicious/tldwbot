"""Transcript acquisition pipeline for YouTube videos."""
from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from .storage import Storage

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency during CI
    WhisperModel = None  # type: ignore

import yt_dlp
from openai import OpenAI

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


def fetch_captions(video_id: str) -> Optional[TranscriptResult]:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            pass
        if transcript is None:
            for candidate in transcript_list:
                transcript = candidate
                if getattr(candidate, "language_code", "").startswith("en"):
                    break
        if transcript is None:
            raise NoTranscriptFound("No transcripts available")
        data = transcript.fetch()
        text = "\n".join(item.get("text", "").strip() for item in data if item.get("text"))
        language = getattr(transcript, "language", None) or getattr(transcript, "language_code", None)
        logger.info("Fetched captions for %s", video_id)
        return TranscriptResult(text=text, source="captions", language=language)
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        logger.warning("Captions unavailable for %s: %s", video_id, exc)
        return None
    except Exception as exc:  # pragma: no cover - upstream errors
        logger.error("Failed to download captions for %s: %s", video_id, exc)
        return None


_LOCAL_WHISPER_CACHE: dict[str, WhisperModel] = {}


def _load_whisper_model(size: str) -> WhisperModel:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")
    if size not in _LOCAL_WHISPER_CACHE:
        logger.info("Loading faster-whisper model %s", size)
        _LOCAL_WHISPER_CACHE[size] = WhisperModel(size)
    return _LOCAL_WHISPER_CACHE[size]


def transcribe_with_local_whisper(video_id: str, model_size: str) -> Optional[TranscriptResult]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "audio.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "quiet": True,
                "noplaylist": True,
                "nocheckcertificate": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                downloaded_path = ydl.prepare_filename(info)
            audio_path = _ensure_audio_extension(downloaded_path)
            model = _load_whisper_model(model_size)
            segments, _ = model.transcribe(audio_path)
            lines = []
            for segment in segments:
                if segment.text:
                    lines.append(segment.text.strip())
            text = "\n".join(lines)
            logger.info("Local whisper transcription complete for %s", video_id)
            return TranscriptResult(text=text, source="local_whisper")
    except Exception as exc:
        logger.error("Local whisper transcription failed for %s: %s", video_id, exc)
        return None


def _ensure_audio_extension(path: str) -> str:
    # If yt-dlp downloads a video container, convert to wav via ffmpeg
    ext = Path(path).suffix.lower()
    if ext in {".wav", ".mp3", ".m4a", ".opus"}:
        return path
    converted = f"{Path(path).with_suffix('.wav')}"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        converted,
    ]
    os.system(" ".join(command))
    return converted if Path(converted).exists() else path


def transcribe_with_openai_whisper(
    video_id: str, client: OpenAI
) -> Optional[TranscriptResult]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "audio.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "quiet": True,
                "noplaylist": True,
                "nocheckcertificate": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                downloaded_path = ydl.prepare_filename(info)
            audio_path = _ensure_audio_extension(downloaded_path)
            with open(audio_path, "rb") as file_stream:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file_stream,
                )
            text = response.text
            logger.info("Cloud whisper transcription complete for %s", video_id)
            return TranscriptResult(text=text, source="cloud_whisper")
    except Exception as exc:
        logger.error("OpenAI whisper transcription failed for %s: %s", video_id, exc)
        return None


def obtain_transcript(
    storage: Storage,
    video_id: str,
    *,
    force_refresh: bool,
    use_local_whisper: bool,
    whisper_model_size: str,
    openai_client: Optional[OpenAI],
) -> TranscriptResult:
    cached = get_cached_transcript(storage, video_id, force_refresh)
    if cached:
        return cached

    # Captions first
    captions = fetch_captions(video_id)
    if captions:
        storage.upsert_transcript(video_id, captions.source, captions.text)
        return captions

    # Local whisper fallback
    if use_local_whisper:
        whisper_result = transcribe_with_local_whisper(video_id, whisper_model_size)
        if whisper_result:
            storage.upsert_transcript(video_id, whisper_result.source, whisper_result.text)
            return whisper_result

    # Cloud whisper fallback
    if openai_client is None:
        raise TranscriptError("OpenAI client unavailable for whisper fallback")
    cloud_result = transcribe_with_openai_whisper(video_id, openai_client)
    if cloud_result:
        storage.upsert_transcript(video_id, cloud_result.source, cloud_result.text)
        return cloud_result

    raise TranscriptError("Unable to retrieve transcript")

