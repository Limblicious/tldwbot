"""Transcript acquisition pipeline for YouTube videos."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .storage import Storage
from .cache_sqlite import get as cache_get_persist, set as cache_set_persist
from .limiters import TokenBucket, SlidingWindowCounter, CircuitBreaker429, full_jitter_sleep

logger = logging.getLogger(__name__)

# Environment configuration for YouTube API hardening
YT_FORCE_IPV4 = os.getenv("YT_FORCE_IPV4", "1") in ("1", "true", "True")
YT_COOKIES = os.getenv("YT_COOKIES")  # optional path to cookies.txt
YT_UA = os.getenv("YT_UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")
YT_REQ_SLEEP = float(os.getenv("YT_REQ_SLEEP", "0"))  # seconds between external requests
YT_CACHE_TTL = int(os.getenv("YT_CACHE_TTL", "7200"))  # 2h

# Rate control
RATE_RPS = float(os.getenv("RATE_RPS", "1.0"))  # tokens per second
RATE_BURST = int(os.getenv("RATE_BURST", "2"))  # burst size for token bucket
RETRY_MAX = int(os.getenv("RETRY_MAX", "3"))  # per video fetch
RETRY_BASE_SEC = float(os.getenv("RETRY_BASE_SEC", "1.0"))  # backoff base

# Circuit breaker for 429
CB_429_THRESHOLD = int(os.getenv("CB_429_THRESHOLD", "3"))  # consecutive 429s to open
CB_OPEN_SECS = int(os.getenv("CB_OPEN_SECS", "1800"))  # 30 min open
CB_HALF_PROBE_SECS = int(os.getenv("CB_HALF_PROBE_SECS", "120"))  # next probe delay

# Quotas
USER_QUOTA_MAX = int(os.getenv("USER_QUOTA_MAX", "5"))  # 5 videos / window
USER_QUOTA_WINDOW = int(os.getenv("USER_QUOTA_WINDOW", "600"))
CHAN_QUOTA_MAX = int(os.getenv("CHAN_QUOTA_MAX", "20"))  # 20 videos / window
CHAN_QUOTA_WINDOW = int(os.getenv("CHAN_QUOTA_WINDOW", "600"))

# Persistent cache
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "/app/cache.db")
PERSIST_TTL_SECS = int(os.getenv("PERSIST_TTL_SECS", os.getenv("YT_CACHE_TTL", "86400")))  # default 1d

# Browser-like headers for direct requests
BROWSER_HEADERS = {
    "User-Agent": YT_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.youtube.com",
    "Referer": "https://www.youtube.com/",
    "DNT": "1",
}

# Requests session with retry policy
_session = requests.Session()
retries = Retry(total=3, backoff_factor=1.2, status_forcelist=(429, 500, 502, 503, 504))
_session.mount("https://", HTTPAdapter(max_retries=retries))
_session.mount("http://", HTTPAdapter(max_retries=retries))

# Single-flight guard + TTL cache to avoid duplicate hits
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, str]] = {}  # video_id -> (expires_at, text)
_inflight: dict[str, threading.Event] = {}  # video_id -> event

# Rate limiters and circuit breaker
_BUCKET = TokenBucket(RATE_RPS, RATE_BURST)
_USERQ = SlidingWindowCounter(USER_QUOTA_WINDOW)
_CHANQ = SlidingWindowCounter(CHAN_QUOTA_WINDOW)
_CB429 = CircuitBreaker429(CB_429_THRESHOLD, CB_OPEN_SECS, CB_HALF_PROBE_SECS)


def _cache_get(video_id: str) -> Optional[str]:
    with _cache_lock:
        x = _cache.get(video_id)
        if not x:
            return None
        exp, txt = x
        if time.time() > exp:
            _cache.pop(video_id, None)
            return None
        return txt


def _cache_set(video_id: str, text: str) -> None:
    with _cache_lock:
        _cache[video_id] = (time.time() + YT_CACHE_TTL, text)


def _single_flight(video_id: str):
    # returns (event, is_leader)
    with _cache_lock:
        ev = _inflight.get(video_id)
        if ev is None:
            ev = threading.Event()
            _inflight[video_id] = ev
            return ev, True
        return ev, False


def _single_flight_done(video_id: str):
    with _cache_lock:
        ev = _inflight.pop(video_id, None)
        if ev:
            ev.set()


def _before_network_call(user_id: str | int | None, channel_id: str | int | None):
    # circuit breaker
    _CB429.before()
    # quotas
    if user_id is not None and _USERQ.count(str(user_id)) >= USER_QUOTA_MAX:
        raise TranscriptError("User quota exceeded; try later.")
    if channel_id is not None and _CHANQ.count(str(channel_id)) >= CHAN_QUOTA_MAX:
        raise TranscriptError("Channel quota exceeded; try later.")
    # token bucket pacing
    wait = _BUCKET.take()
    if wait > 0:
        time.sleep(wait)


def _record_result(ok: bool, rate_limited: bool, user_id: str | int | None, channel_id: str | int | None):
    if user_id is not None:
        _USERQ.add(str(user_id))
    if channel_id is not None:
        _CHANQ.add(str(channel_id))
    if ok:
        _CB429.on_success()
    elif rate_limited:
        _CB429.on_429()
    else:
        _CB429.on_other_error()


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
    rate_limited = False

    # Accept English variants; yt-dlp expects --sub-langs for lists/patterns.
    lang_expr = "en.*,en"

    def _run_and_parse(cmd: list[str]) -> Optional[str]:
        nonlocal rate_limited
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
            )
        except subprocess.TimeoutExpired:
            return None

        # success path
        if result.returncode == 0 and result.stdout:
            text = parse_vtt(result.stdout)
            return text.strip() or None

        # detect rate limiting in stderr
        err = (result.stderr or "")
        if " 429" in err or "Too Many Requests" in err:
            rate_limited = True
            logger.warning("yt-dlp returned 429 / Too Many Requests for %s; will try fallbacks", url)
        return None

    def _build_yt_dlp_cmd(base_cmd: list[str]) -> list[str]:
        """Add hardening options to yt-dlp command."""
        cmd = base_cmd.copy()
        # Insert options after 'yt-dlp' (position 1)
        insert_pos = 1
        if YT_COOKIES:
            cmd[insert_pos:insert_pos] = ["--cookies", YT_COOKIES]
            insert_pos += 2
        if YT_FORCE_IPV4:
            cmd[insert_pos:insert_pos] = ["--force-ipv4"]
            insert_pos += 1
        if YT_UA:
            cmd[insert_pos:insert_pos] = ["--user-agent", YT_UA]
            insert_pos += 2
        if YT_REQ_SLEEP:
            cmd[insert_pos:insert_pos] = ["--sleep-requests", str(int(YT_REQ_SLEEP))]
            insert_pos += 2
        return cmd

    def _direct_url_fallback(auto: bool) -> Optional[str]:
        """
        Some yt-dlp builds won't stream subs to stdout with -o -.
        As a fallback, print the subtitle data structure and parse for VTT URL.
        """
        nonlocal rate_limited
        try:
            import json

            # Get the full subtitle structure as JSON
            if auto:
                dict_key = "automatic_captions"
            else:
                dict_key = "subtitles"

            cmd = _build_yt_dlp_cmd([
                "yt-dlp",
                "--skip-download",
                "--no-playlist",
                "--print", f"%({dict_key})j",
                url,
            ])
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
                            if YT_REQ_SLEEP:
                                time.sleep(YT_REQ_SLEEP)
                            r = _session.get(vtt_url, headers=BROWSER_HEADERS, timeout=30)

                            # Check for rate limiting
                            if r.status_code == 429:
                                rate_limited = True
                                logger.warning(
                                    "HTTP 429 when fetching subtitles for %s (direct URL); will try other fallbacks",
                                    url,
                                )
                                return None

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
                                        if YT_REQ_SLEEP:
                                            time.sleep(YT_REQ_SLEEP)
                                        r = _session.get(vtt_url, headers=BROWSER_HEADERS, timeout=30)

                                        # Check for rate limiting again
                                        if r.status_code == 429:
                                            rate_limited = True
                                            logger.warning(
                                                "HTTP 429 when fetching subtitles for %s (direct URL); will try other fallbacks",
                                                url,
                                            )
                                            return None

                                        r.raise_for_status()
                                        content = r.text
                                        break
                            text = parse_vtt(content).strip()
                            if text:
                                return text
            return None
        except TranscriptError:
            # Re-raise TranscriptError so it propagates up
            raise
        except Exception as e:
            logger.debug("Direct URL fallback failed: %s", str(e)[:100])
            return None

    # Try MANUAL subs first
    cmd_manual = _build_yt_dlp_cmd([
        "yt-dlp",
        "--skip-download",
        "--no-playlist",
        "--write-subs",                 # manual subs
        "--sub-langs", lang_expr,       # NOTE: plural 'sub-langs'
        "--sub-format", "vtt",
        "-o", "-",                      # stream to stdout if supported
        url,
    ])
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
    cmd_auto = _build_yt_dlp_cmd([
        "yt-dlp",
        "--skip-download",
        "--no-playlist",
        "--write-auto-subs",            # auto-generated subs
        "--sub-langs", lang_expr,       # accept en variants
        "--sub-format", "vtt",
        "-o", "-",
        url,
    ])
    txt = _run_and_parse(cmd_auto)
    if txt:
        logger.info("Fetched auto-generated subtitles for %s", url)
        return txt

    # Direct URL fallback for auto
    txt = _direct_url_fallback(auto=True)
    if txt:
        logger.info("Fetched auto-generated subtitles via direct URL for %s", url)
        return txt

    if rate_limited:
        raise TranscriptError("YouTube is temporarily rate limiting requests. Try again later or provide cookies (YT_COOKIES).")
    return None


def fetch_captions(video_id: str) -> Optional[TranscriptResult]:
    """
    Primary path: youtube-transcript-api (manual → auto → translate to en)
    Secondary path: yt-dlp subprocess with English variants
    """
    # cache hit fast-path
    cached = _cache_get(video_id)
    if cached:
        return TranscriptResult(text=cached, source="cache", language="en")

    ev, is_leader = _single_flight(video_id)
    if not is_leader:
        # Wait for leader to finish and return cache value (or None)
        ev.wait(timeout=30)
        cached = _cache_get(video_id)
        return TranscriptResult(text=cached, source="cache", language="en") if cached else None

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"

        # --- Primary: youtube-transcript-api ---
        try:
            from youtube_transcript_api import (
                YouTubeTranscriptApi,
                NoTranscriptFound,
                TranscriptsDisabled,
                VideoUnavailable,
            )

            # Use cookies from env for age/region restrictions
            cookies_path = YT_COOKIES or None

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
                        _cache_set(video_id, text)
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
                        _cache_set(video_id, text)
                        return TranscriptResult(text=text, source="yt-api-translated", language="en")
                except Exception:
                    continue

        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            pass
        except Exception as e:
            # Check for rate limiting
            error_str = str(e).lower()
            if "429" in error_str or "too many requests" in error_str:
                logger.error("YouTube rate limit hit for video %s: %s", video_id, str(e)[:200])
                raise TranscriptError("YouTube is temporarily rate limiting requests. Try again later or provide cookies (YT_COOKIES).")
            # Network, cookie, or other transient issues – fall through to yt-dlp path
            logger.debug("youtube-transcript-api failed for %s, trying yt-dlp: %s", video_id, str(e)[:100])

        # --- Secondary: yt-dlp subprocess path (fixed flags) ---
        text = fetch_transcript(url)
        if text:
            _cache_set(video_id, text)
            return TranscriptResult(text=text, source="yt-dlp", language="en")

        logger.error("No captions available for video %s", video_id)
        return None
    finally:
        _single_flight_done(video_id)


def guarded_fetch_captions(video_id: str, user_id: str | int | None = None, channel_id: str | int | None = None) -> Optional[TranscriptResult]:
    """
    Wraps fetch_captions with quotas, token bucket, circuit breaker, and persistent cache.
    """
    # persistent cache (fast path)
    hit = cache_get_persist(video_id, PERSIST_TTL_SECS, CACHE_DB_PATH)
    if hit:
        text, source = hit
        return TranscriptResult(text=text, source=f"cache:{source}", language="en")

    # single-flight is assumed already present in your code; keep it.
    _before_network_call(user_id, channel_id)
    rate_limited = False
    try:
        # call your existing fetch_captions(video_id) implementation (do not duplicate logic)
        res = fetch_captions(video_id)
        if res and res.text:
            cache_set_persist(video_id, res.text, res.source, CACHE_DB_PATH)
            _record_result(ok=True, rate_limited=False, user_id=user_id, channel_id=channel_id)
            return res
        _record_result(ok=False, rate_limited=False, user_id=user_id, channel_id=channel_id)
        return None
    except TranscriptError as e:
        # if message indicates rate limiting, mark it
        msg = str(e).lower()
        rate_limited = ("rate limit" in msg) or ("429" in msg) or ("too many requests" in msg)
        _record_result(ok=False, rate_limited=rate_limited, user_id=user_id, channel_id=channel_id)
        raise
    except RuntimeError as e:
        # circuit-open
        if "circuit-open" in str(e):
            raise TranscriptError("Temporarily cooling down due to YouTube limits; please try later.")
        raise


def obtain_transcript(
    storage: Storage,
    video_id: str,
) -> TranscriptResult:
    """
    Obtain transcript for a YouTube video.

    Primary: youtube-transcript-api (manual -> auto -> translate->en)
    Secondary: yt-dlp (stdout or direct-URL fallback). No Whisper fallback.
    Raises TranscriptError if captions are unavailable or if YouTube rate limits (429).

    NOTE: This function is deprecated. Use guarded_fetch_captions() instead.
    """
    # Check storage cache
    record = storage.get_transcript(video_id)
    if record:
        logger.info("Transcript cache hit for %s via %s", video_id, record.source)
        return TranscriptResult(text=record.text, source=record.source)

    # Fetch captions only - no Whisper fallback
    captions = fetch_captions(video_id)
    if captions:
        storage.upsert_transcript(video_id, captions.source, captions.text)
        return captions

    # No captions available
    logger.error(
        "No captions available for video %s after trying youtube-transcript-api, yt-dlp stdout, and direct-URL fallback",
        video_id,
    )
    raise TranscriptError(
        "No captions available right now. (Not a rate limit; no usable caption track was found.)"
    )
