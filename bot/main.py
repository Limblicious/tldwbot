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
from .summarize import compute_summary_cache_key, summarize_transcript, DEFAULT_PROMPT
from .transcripts import TranscriptError, extract_video_id, obtain_transcript, guarded_fetch_captions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tldwbot")

# Deduplication: track recent requests to avoid duplicate processing
_recent_requests: dict[tuple, float] = {}
_recent_lock = asyncio.Lock()
_DEDUP_TTL = 600  # 10 minutes

# Concurrency limiting
_worker_semaphore = asyncio.Semaphore(2)


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
        self.openai_client = OpenAI(api_key=openai_api_key)

    async def setup_hook(self) -> None:
        await self.tree.sync()

    def run_bot(self) -> None:
        super().run(self.discord_token)


async def summarize_command(
    interaction: discord.Interaction,
    url: str,
    bot: SummaryBot,
) -> None:
    await interaction.response.defer(thinking=True, ephemeral=True)
    video_id = extract_video_id(url)
    if not video_id:
        await interaction.followup.send(
            "❌ Unable to determine YouTube video ID. Please provide a valid URL or ID.",
            ephemeral=True,
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
            await interaction.followup.send(
                "⏭️ This video is already being processed.",
                ephemeral=True,
            )
            return
        _recent_requests[dedup_key] = now

    # Concurrency limiting
    async with _worker_semaphore:
        prompt_to_use = DEFAULT_PROMPT.strip()
        prompt_hash = compute_summary_cache_key(
            video_id=video_id, prompt=prompt_to_use, model=bot.summary_model
        )

        summary_record = bot.storage.get_summary(video_id, prompt_hash, bot.summary_model)
        transcript_result = None

        if summary_record:
            summary_text = summary_record.summary
            transcript_result = bot.storage.get_transcript(video_id)
            transcript_source = transcript_result.source if transcript_result else "unknown"
            logger.info("Summary cache hit for %s", video_id)
        else:
            try:
                transcript_result = await asyncio.to_thread(
                    guarded_fetch_captions,
                    video_id,
                    user_id=user_id,
                    channel_id=channel_id,
                )
                if not transcript_result:
                    await interaction.followup.send(
                        "❌ No captions available for this video.",
                        ephemeral=True,
                    )
                    return
            except TranscriptError as exc:
                msg = str(exc)
                if "quota exceeded" in msg.lower() or "cooling down" in msg.lower():
                    await interaction.followup.send(
                        f"⏸️ {exc}",
                        ephemeral=True,
                    )
                else:
                    await interaction.followup.send(
                        f"❌ Failed to obtain transcript: {exc}",
                        ephemeral=True,
                    )
                return

            # Store transcript in storage
            bot.storage.upsert_transcript(video_id, transcript_result.source, transcript_result.text)

            transcript_source = transcript_result.source
            try:
                summary_text = await asyncio.to_thread(
                    summarize_transcript,
                    transcript_result.text,
                    client=bot.openai_client,
                    model=bot.summary_model,
                    max_chars_per_chunk=bot.max_chars_per_chunk,
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("Summarization failed for %s", video_id)
                await interaction.followup.send(
                    f"❌ Summarization failed: {exc}",
                    ephemeral=True,
                )
                return

            bot.storage.upsert_summary(video_id, prompt_hash, bot.summary_model, summary_text)

        await send_summary_response(
            interaction,
            video_id=video_id,
            transcript_source=transcript_source,
            summary_text=summary_text,
            max_discord_chars=bot.max_discord_chars,
        )


def split_for_discord(text: str, limit: int) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_point = remaining.rfind("\n", 0, limit)
        if split_point == -1:
            split_point = remaining.rfind(" ", 0, limit)
        if split_point == -1:
            split_point = limit
        chunks.append(remaining[:split_point].strip())
        remaining = remaining[split_point:].strip()
    return chunks


async def send_summary_response(
    interaction: discord.Interaction,
    *,
    video_id: str,
    transcript_source: str,
    summary_text: str,
    max_discord_chars: int,
) -> None:
    header = f"**Video ID:** `{video_id}`\n**Transcript source:** {transcript_source}"
    per_message_limit = min(2000, max_discord_chars)
    messages = split_for_discord(summary_text, per_message_limit)
    if not messages:
        messages = ["(empty summary)"]

    files = []
    if len(summary_text) > max_discord_chars:
        file_buffer = io.BytesIO(summary_text.encode("utf-8"))
        file_buffer.seek(0)
        files.append(discord.File(file_buffer, filename=f"{video_id}_summary.txt"))

    first_body = messages[0]
    extra_chunks = messages[1:]
    if len(header) + 2 + len(first_body) > per_message_limit:
        allowed = max(0, per_message_limit - len(header) - 2)
        if allowed > 0:
            overflow = first_body[allowed:].strip()
            first_body = first_body[:allowed].rstrip()
        else:
            overflow = first_body
            first_body = ""
        combined = []
        if overflow:
            combined.append(overflow)
        combined.extend(extra_chunks)
        extra_text = "\n".join(filter(None, combined))
        extra_chunks = split_for_discord(extra_text, per_message_limit) if extra_text else []

    first_content = f"{header}\n\n{first_body}" if first_body else header
    await interaction.followup.send(first_content, files=files if files else None)

    for chunk in extra_chunks:
        await interaction.followup.send(chunk[:per_message_limit])


def main() -> None:
    load_dotenv()

    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not discord_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    summary_model = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini")
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

