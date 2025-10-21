"""Discord bot entrypoint."""
from __future__ import annotations

import certifi
import os
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import asyncio
import io
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

from .storage import Storage
from .summarize import (
    compute_summary_cache_key,
    summarize_transcript,
    summarize_transcript_async,
    DEFAULT_PROMPT,
    CONTEXT_TOKENS,
    MICRO_TOKENS,
    CONCURRENCY,
    ONE_SHOT_ENABLED,
    OUTPUT_TOKENS,
    BUDGET,
)
from .transcripts import TranscriptError, extract_video_id, obtain_transcript, guarded_fetch_captions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tldwbot")

# Deduplication: track recent requests to avoid duplicate processing
_recent_requests: dict[tuple, float] = {}
_recent_lock = asyncio.Lock()
_DEDUP_TTL = 600  # 10 minutes

# Two-queue concurrency limiting
# Short videos: complete within 10 minutes (600s), use interaction responses
# Long videos: may take longer, use channel.send() after user confirmation
SHORT_QUEUE_LIMIT = int(os.getenv("SHORT_QUEUE_LIMIT", "3"))
LONG_QUEUE_LIMIT = int(os.getenv("LONG_QUEUE_LIMIT", "1"))
_short_queue_semaphore = asyncio.Semaphore(SHORT_QUEUE_LIMIT)
_long_queue_semaphore = asyncio.Semaphore(LONG_QUEUE_LIMIT)

# Queue tracking for status display
_short_queue_count = 0
_long_queue_count = 0
_queue_lock = asyncio.Lock()

# Pending long video confirmations (video_id -> task info)
_pending_confirmations: dict[str, dict] = {}
_confirmation_lock = asyncio.Lock()


class LongVideoConfirmView(discord.ui.View):
    """View with Yes/No buttons for long video confirmation."""

    def __init__(self, video_id: str, estimated_minutes: int):
        super().__init__(timeout=300)  # 5 minute timeout for user response
        self.video_id = video_id
        self.estimated_minutes = estimated_minutes
        self.confirmed = None

    @discord.ui.button(label="Yes, Process It", style=discord.ButtonStyle.green)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = True
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="No, Cancel", style=discord.ButtonStyle.red)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = False
        self.stop()
        await interaction.response.defer()


def get_queue_status() -> str:
    """Get formatted queue status message."""
    short_processing = SHORT_QUEUE_LIMIT - _short_queue_semaphore._value
    short_waiting = _short_queue_count - short_processing
    long_processing = LONG_QUEUE_LIMIT - _long_queue_semaphore._value
    long_waiting = _long_queue_count - long_processing

    status_parts = []
    if short_processing > 0 or short_waiting > 0:
        status_parts.append(f"Short queue: {short_processing} processing, {short_waiting} waiting")
    if long_processing > 0 or long_waiting > 0:
        status_parts.append(f"Long queue: {long_processing} processing, {long_waiting} waiting")

    if status_parts:
        return " | ".join(status_parts)
    return "All queues empty"


class SummaryBot(commands.Bot):
    def __init__(
        self,
        *,
        discord_token: str,
        openai_api_key: str,
        summary_model: str,
        db_path: str,
        max_chars_per_chunk: int,
        max_discord_chars: int,
    ) -> None:
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)
        self.discord_token = discord_token
        self.summary_model = summary_model
        self.storage = Storage(db_path)
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_discord_chars = max_discord_chars
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    async def setup_hook(self) -> None:
        await self.tree.sync()

    def run_bot(self) -> None:
        super().run(self.discord_token)


async def summarize_command(
    interaction: discord.Interaction,
    url: str,
    bot: SummaryBot,
) -> None:
    await interaction.response.defer(thinking=True, ephemeral=False)
    video_id = extract_video_id(url)
    if not video_id:
        await interaction.edit_original_response(
            content="❌ Unable to determine YouTube video ID. Please provide a valid URL or ID."
        )
        return

    # Deduplication check
    user_id = interaction.user.id
    channel_id = interaction.channel_id if interaction.channel_id else 0
    dedup_key = (channel_id, interaction.id, video_id)

    async with _recent_lock:
        now = asyncio.get_event_loop().time()
        # Clean old entries
        to_remove = [k for k, ts in _recent_requests.items() if now - ts > _DEDUP_TTL]
        for k in to_remove:
            del _recent_requests[k]

        if dedup_key in _recent_requests:
            queue_status = get_queue_status()
            await interaction.edit_original_response(
                content=f"⏭️ This video is already being processed.\n\n📊 Queue Status: {queue_status}"
            )
            return
        _recent_requests[dedup_key] = now

    # Initial queue status display
    queue_status = get_queue_status()
    await interaction.edit_original_response(
        content=f"⏳ Fetching captions...\n\n📊 Queue Status: {queue_status}"
    )

    # Determine if video is short or long (need to fetch transcript first)
    try:
        await _process_video_request(interaction, url, video_id, bot, user_id, channel_id)
    finally:
        # Clean up from recent requests after processing completes
        async with _recent_lock:
            _recent_requests.pop(dedup_key, None)


async def _process_video_request(
    interaction: discord.Interaction,
    url: str,
    video_id: str,
    bot: SummaryBot,
    user_id: int,
    channel_id: int,
) -> None:
    """Process video request, routing to appropriate queue based on estimated time."""
    global _short_queue_count, _long_queue_count

    # Check cache first (before queueing)
    prompt_to_use = DEFAULT_PROMPT.strip()
    prompt_hash = compute_summary_cache_key(
        video_id=video_id, prompt=prompt_to_use, model=bot.summary_model
    )

    summary_record = bot.storage.get_summary(video_id, prompt_hash, bot.summary_model)

    if summary_record:
        # Cache hit - return immediately
        summary_text = summary_record.summary
        transcript_result = bot.storage.get_transcript(video_id)
        transcript_source = transcript_result.source if transcript_result else "unknown"
        logger.info("Summary cache hit for %s", video_id)

        queue_status = get_queue_status()
        await send_summary_response(
            interaction,
            video_id=video_id,
            transcript_source=transcript_source,
            summary_text=summary_text,
            max_discord_chars=bot.max_discord_chars,
            queue_status=queue_status,
        )
        return

    # Fetch transcript to estimate processing time
    try:
        transcript_result = await asyncio.to_thread(
            guarded_fetch_captions,
            video_id,
            user_id=user_id,
            channel_id=channel_id,
        )
        if not transcript_result:
            await interaction.edit_original_response(
                content="❌ No captions available for this video."
            )
            return
    except TranscriptError as exc:
        msg = str(exc)
        if "quota exceeded" in msg.lower() or "cooling down" in msg.lower():
            await interaction.edit_original_response(
                content=f"⏸️ {exc}"
            )
        else:
            await interaction.edit_original_response(
                content=f"❌ Failed to obtain transcript: {exc}"
            )
        return

    # Estimate processing time
    from bot.tokens import count_tokens
    from bot.summarize import BUDGET, ONE_SHOT_ENABLED, OUTPUT_TOKENS, MICRO_TOKENS, CONCURRENCY

    transcript_tokens = count_tokens(bot.summary_model, transcript_result.text)
    logger.info(f"Transcript for {video_id}: {transcript_tokens} tokens")

    # New limit: 10 minutes for short queue
    max_processing_seconds = int(os.getenv("MAX_PROCESSING_SECONDS", "600"))
    tokens_per_second = float(os.getenv("TOKENS_PER_SECOND", "30"))
    api_overhead_per_call = float(os.getenv("API_OVERHEAD_SEC", "2.0"))

    estimated_time_sec = _estimate_processing_time(
        transcript_tokens,
        bot,
        tokens_per_second,
        api_overhead_per_call
    )

    logger.info(f"Estimated processing time for {video_id}: {estimated_time_sec:.1f}s (short queue limit: {max_processing_seconds}s)")

    # Route to appropriate queue
    is_long_video = estimated_time_sec > max_processing_seconds

    if is_long_video:
        # Long video - requires user confirmation
        await _handle_long_video(
            interaction,
            video_id,
            bot,
            transcript_result,
            transcript_tokens,
            estimated_time_sec,
            user_id,
            channel_id
        )
    else:
        # Short video - process immediately in short queue
        await _handle_short_video(
            interaction,
            video_id,
            bot,
            transcript_result,
            transcript_tokens,
            estimated_time_sec,
            max_processing_seconds
        )


def _estimate_processing_time(
    transcript_tokens: int,
    bot: SummaryBot,
    tokens_per_second: float,
    api_overhead_per_call: float
) -> float:
    """Estimate processing time for a transcript."""
    from bot.summarize import CONTEXT_TOKENS, ONE_SHOT_ENABLED, OUTPUT_TOKENS, MICRO_TOKENS, CONCURRENCY, BUDGET
    import math

    one_shot_threshold = CONTEXT_TOKENS - 800
    strategy = "one_shot" if (ONE_SHOT_ENABLED and transcript_tokens <= one_shot_threshold) else "hierarchical"

    # Try adaptive estimate first
    adaptive_estimate = bot.storage.get_adaptive_estimate(
        transcript_tokens=transcript_tokens,
        strategy=strategy,
        concurrency=CONCURRENCY
    )

    if adaptive_estimate is not None:
        return adaptive_estimate
    elif ONE_SHOT_ENABLED and transcript_tokens <= BUDGET:
        # One-shot estimate
        return (OUTPUT_TOKENS / tokens_per_second) + api_overhead_per_call
    else:
        # Hierarchical estimate
        num_chunks = max(1, (transcript_tokens + MICRO_TOKENS - 1) // MICRO_TOKENS)
        chunk_output_tokens = num_chunks * 256
        chunk_time = (chunk_output_tokens / tokens_per_second) + (num_chunks * api_overhead_per_call / CONCURRENCY)

        merge_levels = max(0, math.ceil(math.log2(num_chunks))) if num_chunks > 1 else 0
        total_merge_calls = 0
        remaining = num_chunks
        for level in range(merge_levels):
            group_size = max(1, 12000 // 1000)
            groups_at_level = max(1, (remaining + group_size - 1) // group_size)
            total_merge_calls += groups_at_level
            remaining = groups_at_level

        merge_output_tokens = total_merge_calls * 600
        merge_time = (merge_output_tokens / tokens_per_second) + (total_merge_calls * api_overhead_per_call / CONCURRENCY)

        return chunk_time + merge_time


async def _handle_short_video(
    interaction: discord.Interaction,
    video_id: str,
    bot: SummaryBot,
    transcript_result,
    transcript_tokens: int,
    estimated_time_sec: float,
    max_processing_seconds: int
) -> None:
    """Handle short video processing in the short queue."""
    global _short_queue_count

    # Increment queue count
    async with _queue_lock:
        _short_queue_count += 1

    try:
        queue_status = get_queue_status()
        await interaction.edit_original_response(
            content=f"⏳ Entering short video queue...\n\n📊 Queue Status: {queue_status}"
        )

        # Wait for slot in short queue
        async with _short_queue_semaphore:
            queue_status = get_queue_status()
            await interaction.edit_original_response(
                content=f"⏳ Processing now (short queue)...\n\n📊 Queue Status: {queue_status}"
            )

            # Process the video
            summary_text = await _summarize_video(
                interaction,
                video_id,
                bot,
                transcript_result,
                transcript_tokens,
                estimated_time_sec,
                is_long_video=False
            )

            # Send response via interaction
            queue_status = get_queue_status()
            await send_summary_response(
                interaction,
                video_id=video_id,
                transcript_source=transcript_result.source,
                summary_text=summary_text,
                max_discord_chars=bot.max_discord_chars,
                queue_status=queue_status,
            )
    finally:
        # Decrement queue count
        async with _queue_lock:
            _short_queue_count -= 1


async def _handle_long_video(
    interaction: discord.Interaction,
    video_id: str,
    bot: SummaryBot,
    transcript_result,
    transcript_tokens: int,
    estimated_time_sec: float,
    user_id: int,
    channel_id: int
) -> None:
    """Handle long video with user confirmation and channel.send()."""
    global _long_queue_count

    # Format time estimate
    estimated_minutes = int(estimated_time_sec / 60)
    if estimated_minutes < 60:
        time_str = f"~{estimated_minutes} minutes"
    else:
        hours = estimated_minutes // 60
        mins = estimated_minutes % 60
        time_str = f"~{hours}h {mins}m"

    # Ask user for confirmation
    view = LongVideoConfirmView(video_id, estimated_minutes)
    queue_status = get_queue_status()

    await interaction.edit_original_response(
        content=(
            f"⚠️ **Long Video Detected**\n\n"
            f"This video will take approximately **{time_str}** to summarize.\n"
            f"The summary will be posted in this channel when complete, and you'll be pinged.\n\n"
            f"📊 Queue Status: {queue_status}\n\n"
            f"Do you want to continue?"
        ),
        view=view
    )

    # Wait for user response (5 minute timeout)
    await view.wait()

    if view.confirmed is None:
        # Timeout
        await interaction.edit_original_response(
            content="⏱️ Confirmation timeout. Request canceled.",
            view=None
        )
        return
    elif view.confirmed is False:
        # User canceled
        await interaction.edit_original_response(
            content="❌ Request canceled by user.",
            view=None
        )
        return

    # User confirmed - proceed with long video processing
    async with _queue_lock:
        _long_queue_count += 1

    # Store channel and user for later ping
    channel = interaction.channel

    try:
        queue_status = get_queue_status()
        await interaction.edit_original_response(
            content=f"✅ Confirmed! Entering long video queue...\n\n📊 Queue Status: {queue_status}",
            view=None
        )

        # Wait for slot in long queue
        async with _long_queue_semaphore:
            queue_status = get_queue_status()
            if channel:
                await channel.send(
                    f"<@{user_id}> 🎬 Now processing your long video: `{video_id}`\n\n📊 Queue Status: {queue_status}"
                )

            # Process the video
            summary_text = await _summarize_video(
                None,  # No interaction updates for long videos
                video_id,
                bot,
                transcript_result,
                transcript_tokens,
                estimated_time_sec,
                is_long_video=True
            )

            # Send result via channel.send() to bypass 15-min interaction limit
            queue_status = get_queue_status()
            if channel:
                await _send_summary_to_channel(
                    channel,
                    user_id,
                    video_id,
                    transcript_result.source,
                    summary_text,
                    queue_status
                )
    finally:
        # Decrement queue count
        async with _queue_lock:
            _long_queue_count -= 1


async def _summarize_video(
    interaction: Optional[discord.Interaction],
    video_id: str,
    bot: SummaryBot,
    transcript_result,
    transcript_tokens: int,
    estimated_time_sec: float,
    is_long_video: bool
) -> str:
    """Core summarization logic, shared by both queues."""
    from bot.summarize import CONTEXT_TOKENS, ONE_SHOT_ENABLED, MICRO_TOKENS
    import time

    # Store transcript
    bot.storage.upsert_transcript(video_id, transcript_result.source, transcript_result.text)

    # Determine strategy
    one_shot_threshold = CONTEXT_TOKENS - 800
    strategy = "one_shot" if (ONE_SHOT_ENABLED and transcript_tokens <= one_shot_threshold) else "hierarchical"

    # Create progress callback (only for short videos with interaction)
    start_time = time.time()
    last_update_time = [0.0]

    async def progress_callback(message: str):
        if interaction is None or is_long_video:
            return  # No progress updates for long videos

        now = time.time()
        if now - last_update_time[0] >= 5.0:
            try:
                elapsed = now - start_time
                remaining = max(0, estimated_time_sec - elapsed)
                if remaining < 60:
                    time_remaining = f"~{int(remaining)}s remaining"
                else:
                    minutes = int(remaining // 60)
                    seconds = int(remaining % 60)
                    time_remaining = f"~{minutes}m {seconds}s remaining"

                queue_status = get_queue_status()
                await interaction.edit_original_response(
                    content=f"⏳ {message}... ({time_remaining})\n\n📊 Queue Status: {queue_status}"
                )
                last_update_time[0] = now
            except Exception:
                pass

    # Summarize
    from bot.summarize import summarize_transcript_async

    summary_text = await summarize_transcript_async(
        transcript_result.text,
        client=bot.openai_client,
        model=bot.summary_model,
        max_chars_per_chunk=bot.max_chars_per_chunk,
        progress_callback=progress_callback,
    )

    # Record performance
    actual_time = time.time() - start_time
    num_chunks = max(1, (transcript_tokens + MICRO_TOKENS - 1) // MICRO_TOKENS) if strategy == "hierarchical" else 1

    bot.storage.record_performance(
        video_id=video_id,
        transcript_tokens=transcript_tokens,
        num_chunks=num_chunks,
        processing_time_seconds=actual_time,
        strategy=strategy,
    )

    # Save summary
    prompt_to_use = DEFAULT_PROMPT.strip()
    prompt_hash = compute_summary_cache_key(
        video_id=video_id, prompt=prompt_to_use, model=bot.summary_model
    )
    bot.storage.upsert_summary(video_id, prompt_hash, bot.summary_model, summary_text)

    return summary_text


async def _send_summary_to_channel(
    channel,
    user_id: int,
    video_id: str,
    transcript_source: str,
    summary_text: str,
    queue_status: str
) -> None:
    """Send summary to channel for long videos (bypasses interaction timeout)."""
    EMBED_TITLE_MAX = 256
    EMBED_DESC_MAX = 4096
    EMBED_TOTAL_MAX = 6000

    title = "Video Summary (Long Video)"[:EMBED_TITLE_MAX]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    header = (
        f"<@{user_id}> ✅ **Summary Complete!**\n"
        f"**Video:** {video_url} · **Transcript source:** {transcript_source}\n\n"
        f"📊 Queue Status: {queue_status}"
    )

    max_desc_by_total = max(0, min(EMBED_DESC_MAX, EMBED_TOTAL_MAX - len(title) - 80))
    desc = summary_text[:max_desc_by_total]

    embed = discord.Embed(title=title, description=desc, color=discord.Color.green())

    await channel.send(content=header, embed=embed)


async def send_summary_response(
    interaction: discord.Interaction,
    *,
    video_id: str,
    transcript_source: str,
    summary_text: str,
    max_discord_chars: int,  # kept for signature compatibility; not used
    queue_status: str,
) -> None:
    """Send a single public message with one embed, no attachments."""
    # Discord limits
    EMBED_TITLE_MAX = 256
    EMBED_DESC_MAX = 4096
    EMBED_TOTAL_MAX = 6000

    title = "Video Summary"[:EMBED_TITLE_MAX]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    header = f"**Video:** {video_url} · **Transcript source:** {transcript_source}\n\n📊 Queue Status: {queue_status}"

    # Single embed; compute safe description cap so total <= 6000
    # leave ~80 chars slack for internals/formatting
    max_desc_by_total = max(0, min(EMBED_DESC_MAX, EMBED_TOTAL_MAX - len(title) - 80))
    desc = summary_text[:max_desc_by_total]

    embed = discord.Embed(title=title, description=desc, color=discord.Color.blurple())

    await interaction.edit_original_response(
        content=header,   # tiny; under 2k
        embeds=[embed],   # exactly one embed
        attachments=[]    # never attach files
    )


def main() -> None:
    load_dotenv()

    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not discord_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    summary_model = os.getenv("OPENAI_SUMMARY_MODEL", "deepseek-r1-distill-qwen-7b")
    db_path = os.getenv("CACHE_DB", "cache.sqlite3")
    max_chars_per_chunk = int(os.getenv("MAX_CHARS_PER_CHUNK", "4000"))
    max_discord_chars = int(os.getenv("MAX_DISCORD_MSG_CHARS", "1900"))

    bot = SummaryBot(
        discord_token=discord_token,
        openai_api_key=openai_api_key,
        summary_model=summary_model,
        db_path=db_path,
        max_chars_per_chunk=max_chars_per_chunk,
        max_discord_chars=max_discord_chars,
    )

    @bot.tree.command(name="summarize", description="Summarize a YouTube video")
    @app_commands.describe(
        url="YouTube video URL or ID",
    )
    async def summarize(  # type: ignore[unused-ignore]
        interaction: discord.Interaction,
        url: str,
    ) -> None:
        await summarize_command(
            interaction,
            url,
            bot,
        )

    bot.run_bot()


if __name__ == "__main__":  # pragma: no cover
    main()

