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
    Try to fetch subtitles via yt-dlp, preferring manual then auto-generated.
    Accept English variants (en, en-US, en-GB, ...). Return plain text or None.
    """
    import subprocess
    import logging

    logger = logging.getLogger(__name__)

    # Accept English variants; yt-dlp expects --sub-langs for lists/patterns.
    lang_expr = "en.*,en"

    def _run_and_parse(cmd: list[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
            )
        except subprocess.TimeoutExpired:
            return None

        if result.returncode == 0 and result.stdout:
            text = parse_vtt(result.stdout)
            return text.strip() or None
        return None

    def _direct_url_fallback(auto: bool) -> Optional[str]:
        """
        Some yt-dlp builds won't stream subs to stdout with -o -.
        As a fallback, print the subtitle data structure and parse for VTT URL.
        """
        try:
            import json
            import requests

            # Get the full subtitle structure as JSON
            if auto:
                dict_key = "automatic_captions"
            else:
                dict_key = "subtitles"

            cmd = [
                "yt-dlp",
                "--skip-download",
                "--no-playlist",
                "--print", f"%({dict_key})j",
                url,
            ]
            pr = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if pr.returncode != 0 or not pr.stdout.strip():
                return None

            # Parse JSON and find English VTT URL
            subs_data = json.loads(pr.stdout.strip())

            # Try different English language codes
            for lang_key in ["en", "en-US", "en-GB", "en-AU", "en-CA"]:
                if lang_key in subs_data:
                    formats = subs_data[lang_key]
                    # Find VTT format
                    for fmt in formats:
                        if fmt.get("ext") == "vtt" and "url" in fmt:
                            vtt_url = fmt["url"]
                            # Sometimes YouTube returns HLS M3U8 playlists instead of direct VTT
                            # If URL contains "timedtext", we can force vtt format
                            if "/api/timedtext" in vtt_url and "fmt=" not in vtt_url:
                                vtt_url = vtt_url + "&fmt=vtt"
                            r = requests.get(vtt_url, timeout=30)
                            r.raise_for_status()
                            content = r.text
                            # If we got M3U8 playlist, try to force VTT format
                            if content.startswith("#EXTM3U"):
                                # Extract base timedtext URL from M3U8 and force vtt
                                for line in content.split("\n"):
                                    if "/api/timedtext?" in line:
                                        # Add or replace fmt parameter
                                        if "fmt=" in line:
                                            vtt_url = line.split("&fmt=")[0] + "&fmt=vtt"
                                        else:
                                            vtt_url = line.strip() + "&fmt=vtt"
                                        r = requests.get(vtt_url, timeout=30)
                                        r.raise_for_status()
                                        content = r.text
                                        break
                            text = parse_vtt(content).strip()
                            if text:
                                return text
            return None
        except Exception:
            return None

    # Try MANUAL subs first
    cmd_manual = [
        "yt-dlp",
        "--skip-download",
        "--no-playlist",
        "--write-subs",                 # manual subs
        "--sub-langs", lang_expr,       # NOTE: plural 'sub-langs'
        "--sub-format", "vtt",
        "-o", "-",                      # stream to stdout if supported
        url,
    ]
    txt = _run_and_parse(cmd_manual)
    if txt:
        logger.info("Fetched manual subtitles for %s", url)
        return txt

    # If we didn't get it via stdout, try direct-URL fallback for manual subs
    txt = _direct_url_fallback(auto=False)
    if txt:
        logger.info("Fetched manual subtitles via direct URL for %s", url)
        return txt

    logger.info("No manual subtitles, trying auto-generated for %s", url)

    # Try AUTO-GENERATED subs
    cmd_auto = [
        "yt-dlp",
        "--skip-download",
        "--no-playlist",
        "--write-auto-subs",            # auto-generated subs
        "--sub-langs", lang_expr,       # accept en variants
        "--sub-format", "vtt",
        "-o", "-",
        url,
    ]
    txt = _run_and_parse(cmd_auto)
    if txt:
        logger.info("Fetched auto-generated subtitles for %s", url)
        return txt

    # Direct URL fallback for auto
    txt = _direct_url_fallback(auto=True)
    if txt:
        logger.info("Fetched auto-generated subtitles via direct URL for %s", url)
        return txt

    return None


def fetch_captions(video_id: str) -> Optional[TranscriptResult]:
    """
    Primary path: youtube-transcript-api (manual → auto → translate to en)
    Secondary path: yt-dlp subprocess with English variants
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    # --- Primary: youtube-transcript-api ---
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        # If you keep a cookies.txt for age/region restrictions, set env YT_COOKIES
        import os
        cookies_path = os.getenv("YT_COOKIES") or None

        tl = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path)

        # Prefer manually-created English first
        for finder in (tl.find_manually_created_transcript, tl.find_generated_transcript):
            try:
                t = finder(["en", "en-US", "en-GB", "en-AU", "en-CA"])
                parts = t.fetch()
                text = "\n".join(
                    s["text"].replace("\n", " ").strip()
                    for s in parts
                    if s.get("text", "").strip()
                ).strip()
                if text:
                    src = "yt-api-manual" if finder is tl.find_manually_created_transcript else "yt-api-auto"
                    return TranscriptResult(text=text, source=src, language=getattr(t, "language_code", "en"))
            except Exception:
                pass

        # Translate any available track to English if needed
        for t in tl:
            try:
                t_en = t.translate("en")
                parts = t_en.fetch()
                text = "\n".join(
                    s["text"].replace("\n", " ").strip()
                    for s in parts
                    if s.get("text", "").strip()
                ).strip()
                if text:
                    return TranscriptResult(text=text, source="yt-api-translated", language="en")
            except Exception:
                continue

    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        pass
    except Exception:
        # Network, cookie, or other transient issues – fall through to yt-dlp path
        pass

    # --- Secondary: yt-dlp subprocess path (fixed flags) ---
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
