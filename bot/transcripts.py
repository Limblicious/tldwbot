"""Transcript acquisition pipeline for YouTube videos."""
from __future__ import annotations

import json
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

# Additional pacing / cooldown
NEG_CACHE_TTL = int(os.getenv("NEG_CACHE_TTL", "3600"))  # per-video cooldown after 429 (1h)
JITTER_MAX_MS = int(os.getenv("JITTER_MAX_MS", "800"))   # stable jitter cap (0–800ms)

# VTT file cleanup
KEEP_VTT = os.getenv("KEEP_VTT", "0") in ("1", "true", "True")

# Subprocess timeout protection
YTDLP_TIMEOUT_SEC = int(os.getenv("YTDLP_TIMEOUT_SEC", "90"))  # 90s timeout for yt-dlp calls

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
retries = Retry(
    total=3,
    backoff_factor=1.5,
    status_forcelist=(429, 500, 502, 503, 504),
    respect_retry_after_header=True,
)
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

# Simple metrics counters
_metrics = {
    "fetch_attempts": 0,
    "fetch_429": 0,
    "neg_cache_hits": 0,
    "direct_url_first_hits": 0,
    "stdout_hits": 0,
}

# Negative cache for per-video cooldown after 429
_neg_cache_lock = threading.Lock()
_neg_cache: dict[str, float] = {}  # video_id -> expires_at


def _purge_leftover_vtts():
    """Delete any leftover .vtt files from previous runs (unless KEEP_VTT=1)."""
    if KEEP_VTT:
        return
    try:
        from pathlib import Path
        for p in Path.cwd().glob("*.vtt"):
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


# Purge leftover VTT files on module load
_purge_leftover_vtts()


def _neg_cache_set(video_id: str):
    with _neg_cache_lock:
        _neg_cache[video_id] = time.time() + NEG_CACHE_TTL


def _neg_cache_get(video_id: str) -> bool:
    with _neg_cache_lock:
        exp = _neg_cache.get(video_id)
        if not exp:
            return False
        if time.time() > exp:
            _neg_cache.pop(video_id, None)
            return False
        return True


def _stable_jitter(video_id: str):
    """Deterministic jitter (0..JITTER_MAX_MS ms) based on video_id."""
    h = abs(hash(video_id)) % (JITTER_MAX_MS + 1)
    if h:
        time.sleep(h / 1000.0)


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
    """Parse VTT subtitle format and extract text with timestamps."""
    lines = []
    current_timestamp = None

    for line in vtt_content.split('\n'):
        line = line.strip()

        # Skip WEBVTT header, empty lines, and metadata
        if not line or line.startswith('WEBVTT') or line.isdigit():
            continue
        if line.startswith('Kind:') or line.startswith('Language:'):
            continue

        # Extract timestamp from lines like "00:00:05.000 --> 00:00:08.000"
        if '-->' in line:
            # Extract start timestamp (before the -->)
            timestamp_part = line.split('-->')[0].strip()
            # Convert to [MM:SS] or [HH:MM:SS] format
            # VTT format is typically HH:MM:SS.mmm or MM:SS.mmm
            time_parts = timestamp_part.split(':')
            if len(time_parts) == 3:
                # HH:MM:SS.mmm format
                hh, mm, ss = time_parts
                ss = ss.split('.')[0]  # Remove milliseconds
                if hh == '00':
                    current_timestamp = f"[{mm}:{ss}]"
                else:
                    current_timestamp = f"[{hh}:{mm}:{ss}]"
            elif len(time_parts) == 2:
                # MM:SS.mmm format
                mm, ss = time_parts
                ss = ss.split('.')[0]
                current_timestamp = f"[{mm}:{ss}]"
            continue

        # Append text lines with timestamp prefix
        if current_timestamp and line:
            lines.append(f"{current_timestamp} {line}")
            current_timestamp = None  # Reset after using
        elif line:
            lines.append(line)

    return '\n'.join(lines)


def fetch_transcript(url: str, tl=None) -> Optional[str]:
    """
    Try to fetch subtitles via yt-dlp, preferring manual then auto-generated.
    Accept English variants (en, en-US, en-GB, ...). Return plain text or None.
    """
    import subprocess
    import logging

    logger = logging.getLogger(__name__)
    rate_limited = False

    def _build_yt_dlp_cmd(base_cmd: list[str]) -> list[str]:
        """Add hardening options to yt-dlp command."""
        cmd = base_cmd.copy()
        # Insert options after 'yt-dlp' (position 1)
        insert_pos = 1
        # Reliability knobs
        cmd[insert_pos:insert_pos] = ["--retries", "3", "--retry-sleep", "1", "--socket-timeout", "15"]
        insert_pos += 6
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

    def _print_and_fetch(is_auto: bool) -> Optional[str]:
        """
        Low-cost direct-URL discovery via JSON print.
        Prefer this because it's cheap and avoids subtitle-file writes.
        """
        nonlocal rate_limited
        # Get the subtitle structure as JSON
        print_tpl = "automatic_captions%()j" if is_auto else "subtitles%()j"
        cmd = ["yt-dlp", "--skip-download", "--no-playlist", "--print", print_tpl, url]
        cmd = _build_yt_dlp_cmd(cmd)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=YTDLP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0:
            if r.stderr and (" 429" in r.stderr or "Too Many Requests" in r.stderr):
                rate_limited = True
            return None
        try:
            data = r.stdout.strip()
            if not data:
                return None
            payload = json.loads(data)
            # payload maps language -> [ {url: ... , ext: ...}, ... ]
            for lang, entries in (payload or {}).items():
                if not lang or not lang.lower().startswith("en"):  # accept en variants
                    continue
                for ent in entries or []:
                    if ent.get("ext") != "vtt":  # prefer vtt
                        continue
                    vtt_url = ent.get("url")
                    if not vtt_url:
                        continue
                    if YT_REQ_SLEEP:
                        time.sleep(YT_REQ_SLEEP)
                    resp = _session.get(vtt_url, headers=BROWSER_HEADERS, timeout=30)
                    if resp.status_code == 429:
                        rate_limited = True
                        return None
                    resp.raise_for_status()
                    text = parse_vtt(resp.text).strip()
                    if text:
                        _metrics["direct_url_first_hits"] += 1
                        return text
        except Exception:
            return None
        return None

    def _direct_url_fallback(auto: bool) -> Optional[str]:
        """
        Legacy fallback: print the subtitle data structure and parse for VTT URL.
        """
        nonlocal rate_limited
        try:
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
            pr = subprocess.run(cmd, capture_output=True, text=True, timeout=YTDLP_TIMEOUT_SEC)
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

    # Bump fetch attempts metric
    _metrics["fetch_attempts"] += 1

    # REORDERED: Prefer low-cost direct-URL JSON print first

    # Try MANUAL subs via direct JSON path (cheap)
    if tl:
        with tl.span("ytdlp_print_manual"):
            txt = _print_and_fetch(is_auto=False)
    else:
        txt = _print_and_fetch(is_auto=False)
    if txt:
        logger.info("Fetched manual subtitles via direct-URL for %s", url)
        return txt

    # Try AUTO subs via direct JSON path (cheap)
    if tl:
        with tl.span("ytdlp_print_auto"):
            txt = _print_and_fetch(is_auto=True)
    else:
        txt = _print_and_fetch(is_auto=True)
    if txt:
        logger.info("Fetched auto-generated subtitles via direct-URL for %s", url)
        return txt

    logger.info("No manual subtitles via direct JSON, trying legacy direct-URL for %s", url)

    # Try the legacy direct-URL fallback for manual subs
    txt = _direct_url_fallback(auto=False)
    if txt:
        logger.info("Fetched manual subtitles via legacy direct-URL for %s", url)
        return txt

    logger.info("No manual subtitles, trying auto-generated legacy for %s", url)

    # Try the legacy direct-URL fallback for auto subs
    txt = _direct_url_fallback(auto=True)
    if txt:
        logger.info("Fetched auto-generated subtitles via legacy direct-URL for %s", url)
        return txt

    if rate_limited:
        raise TranscriptError("YouTube is temporarily rate limiting requests. Try again later or provide cookies (YT_COOKIES).")
    return None


def fetch_captions(video_id: str, tl=None) -> Optional[TranscriptResult]:
    """
    Primary path: youtube-transcript-api (manual → auto → translate to en)
    Secondary path: yt-dlp subprocess with English variants
    """
    # Check negative cache (per-video cooldown after 429)
    if _neg_cache_get(video_id):
        _metrics["neg_cache_hits"] += 1
        raise TranscriptError("Temporarily cooling down on this video due to prior rate limiting.")

    # cache hit fast-path
    if tl:
        with tl.span("memory_cache_get"):
            cached = _cache_get(video_id)
    else:
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

        # Apply stable jitter based on video_id
        if tl:
            with tl.span("stable_jitter"):
                _stable_jitter(video_id)
        else:
            _stable_jitter(video_id)

        # --- Primary: youtube-transcript-api ---
        # TEMPORARILY DISABLED: youtube-transcript-api has XML parsing issues with some videos
        # Skip directly to yt-dlp fallback which is more reliable
        skip_ytapi = True
        if not skip_ytapi:
            try:
                from youtube_transcript_api import (
                    YouTubeTranscriptApi,
                    NoTranscriptFound,
                    TranscriptsDisabled,
                    VideoUnavailable,
                )

                # Use cookies from env for age/region restrictions
                cookies_path = YT_COOKIES or None

                if tl:
                    with tl.span("yt_transcript_api"):
                        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path)
                else:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path)

                # Prefer manually-created English first
                for finder in (transcript_list.find_manually_created_transcript, transcript_list.find_generated_transcript):
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
                            src = "yt-api-manual" if finder is transcript_list.find_manually_created_transcript else "yt-api-auto"
                            return TranscriptResult(text=text, source=src, language=getattr(t, "language_code", "en"))
                    except Exception:
                        pass

                # Translate any available track to English if needed
                for t in transcript_list:
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
                    _neg_cache_set(video_id)
                    _metrics["fetch_429"] += 1
                    raise TranscriptError("YouTube is temporarily rate limiting requests. Try again later or provide cookies (YT_COOKIES).")
                # Network, cookie, or other transient issues – fall through to yt-dlp path
                logger.debug("youtube-transcript-api failed for %s, trying yt-dlp: %s", video_id, str(e)[:100])

        # --- Secondary: yt-dlp subprocess path (fixed flags) ---
        try:
            if tl:
                with tl.span("yt_dlp_subprocess"):
                    text = fetch_transcript(url, tl=tl)
            else:
                text = fetch_transcript(url)

            if text:
                _cache_set(video_id, text)
                return TranscriptResult(text=text, source="yt-dlp", language="en")
        except TranscriptError as e:
            # Check if it's a rate limit error from fetch_transcript
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg or "too many requests" in msg:
                _neg_cache_set(video_id)
                _metrics["fetch_429"] += 1
            raise

        logger.error("No captions available for video %s", video_id)
        return None
    finally:
        _single_flight_done(video_id)


def guarded_fetch_captions(video_id: str, user_id: str | int | None = None, channel_id: str | int | None = None, tl=None) -> Optional[TranscriptResult]:
    """
    Wraps fetch_captions with quotas, token bucket, circuit breaker, and persistent cache.
    """
    try:
        # persistent cache (fast path)
        if tl:
            with tl.span("persistent_cache_get"):
                hit = cache_get_persist(video_id, PERSIST_TTL_SECS, CACHE_DB_PATH)
        else:
            hit = cache_get_persist(video_id, PERSIST_TTL_SECS, CACHE_DB_PATH)

        if hit:
            text, source = hit
            return TranscriptResult(text=text, source=f"cache:{source}", language="en")

        # single-flight is assumed already present in your code; keep it.
        if tl:
            with tl.span("before_network_call"):
                _before_network_call(user_id, channel_id)
        else:
            _before_network_call(user_id, channel_id)

        rate_limited = False
        try:
            # call your existing fetch_captions(video_id) implementation (do not duplicate logic)
            if tl:
                with tl.span("fetch_captions"):
                    res = fetch_captions(video_id, tl=tl)
            else:
                res = fetch_captions(video_id)

            if res and res.text:
                if tl:
                    with tl.span("persistent_cache_set"):
                        cache_set_persist(video_id, res.text, res.source, CACHE_DB_PATH)
                else:
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
    finally:
        # Log metrics for monitoring
        logger.debug("yt-metrics attempts=%s 429=%s negcache=%s directFirst=%s stdout=%s",
                     _metrics["fetch_attempts"], _metrics["fetch_429"],
                     _metrics["neg_cache_hits"], _metrics["direct_url_first_hits"],
                     _metrics["stdout_hits"])


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
