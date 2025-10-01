"""Discord bot entrypoint."""
from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

from .storage import Storage
from .summarize import compute_summary_cache_key, summarize_transcript, DEFAULT_PROMPT
from .transcripts import TranscriptError, extract_video_id, obtain_transcript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tldwbot")


def getenv_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class SummaryBot(commands.Bot):
    def __init__(
        self,
        *,
        discord_token: str,
        openai_api_key: str,
        summary_model: str,
        db_path: str,
        use_local_whisper: bool,
        whisper_model_size: str,
        max_chars_per_chunk: int,
        max_discord_chars: int,
    ) -> None:
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)
        self.discord_token = discord_token
        self.summary_model = summary_model
        self.storage = Storage(db_path)
        self.use_local_whisper = use_local_whisper
        self.whisper_model_size = whisper_model_size
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
    force_refresh: bool,
    prompt_override: Optional[str],
    bot: SummaryBot,
) -> None:
    await interaction.response.defer(thinking=True)
    video_id = extract_video_id(url)
    if not video_id:
        await interaction.followup.send(
            "❌ Unable to determine YouTube video ID. Please provide a valid URL or ID.",
            ephemeral=True,
        )
        return

    prompt_to_use = (prompt_override or DEFAULT_PROMPT).strip()
    prompt_hash = compute_summary_cache_key(
        video_id=video_id, prompt=prompt_to_use, model=bot.summary_model
    )

    summary_record = None if force_refresh else bot.storage.get_summary(video_id, prompt_hash, bot.summary_model)
    transcript_result = None

    if summary_record:
        summary_text = summary_record.summary
        transcript_result = bot.storage.get_transcript(video_id)
        transcript_source = transcript_result.source if transcript_result else "unknown"
        logger.info("Summary cache hit for %s", video_id)
    else:
        try:
            transcript_result = await asyncio.to_thread(
                obtain_transcript,
                bot.storage,
                video_id,
                force_refresh=force_refresh,
                use_local_whisper=bot.use_local_whisper,
                whisper_model_size=bot.whisper_model_size,
                openai_client=bot.openai_client,
            )
        except TranscriptError as exc:
            await interaction.followup.send(
                f"❌ Failed to obtain transcript: {exc}",
                ephemeral=True,
            )
            return

        transcript_source = transcript_result.source
        try:
            summary_text = await asyncio.to_thread(
                summarize_transcript,
                transcript_result.text,
                client=bot.openai_client,
                model=bot.summary_model,
                max_chars_per_chunk=bot.max_chars_per_chunk,
                prompt_override=prompt_override,
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
    use_local_whisper = getenv_bool("USE_LOCAL_WHISPER", True)
    whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "medium")
    max_chars_per_chunk = int(os.getenv("MAX_CHARS_PER_CHUNK", "4000"))
    max_discord_chars = int(os.getenv("MAX_DISCORD_MSG_CHARS", "1900"))

    bot = SummaryBot(
        discord_token=discord_token,
        openai_api_key=openai_api_key,
        summary_model=summary_model,
        db_path=db_path,
        use_local_whisper=use_local_whisper,
        whisper_model_size=whisper_model_size,
        max_chars_per_chunk=max_chars_per_chunk,
        max_discord_chars=max_discord_chars,
    )

    @bot.tree.command(name="summarize", description="Summarize a YouTube video")
    @app_commands.describe(
        url="YouTube video URL or ID",
        force_refresh="Bypass cached transcript and summary",
        prompt_override="Override the default summary prompt",
    )
    async def summarize(  # type: ignore[unused-ignore]
        interaction: discord.Interaction,
        url: str,
        force_refresh: bool = False,
        prompt_override: Optional[str] = None,
    ) -> None:
        await summarize_command(
            interaction,
            url,
            force_refresh,
            prompt_override,
            bot,
        )

    bot.run_bot()


if __name__ == "__main__":  # pragma: no cover
    main()

