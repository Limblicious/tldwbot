"""Transcript summarization with adaptive strategy and hierarchical processing."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI

from bot.tokens import count_tokens
from bot.stream_split import stream_split

logger = logging.getLogger("tldwbot")

# Configuration from environment
ONE_SHOT_ENABLED = os.getenv("ONE_SHOT_ENABLED", "1") in ("1", "true", "True")
CONTEXT_TOKENS = int(os.getenv("CONTEXT_TOKENS", "128000"))  # Model context window
PROMPT_TOKENS = int(os.getenv("PROMPT_TOKENS", "2500"))
OUTPUT_TOKENS = int(os.getenv("OUTPUT_TOKENS", "2000"))
SAFETY_TOKENS = int(os.getenv("SAFETY_TOKENS", "1000"))
BUDGET = CONTEXT_TOKENS - PROMPT_TOKENS - OUTPUT_TOKENS - SAFETY_TOKENS

# For small context windows (e.g., 4096), use simpler allocation
if CONTEXT_TOKENS <= 8192:
    # Small context: use ~50% for chunks, ~30% for merging
    MICRO_TOKENS = max(512, int(CONTEXT_TOKENS * 0.5))
    MERGE_TOKENS = max(1000, int(CONTEXT_TOKENS * 0.3))
else:
    # Large context: use original formula
    MICRO_TOKENS = min(4096, BUDGET // 4)
    MERGE_TOKENS = min(12000, max(1000, BUDGET - 5000))  # Ensure minimum 1000 tokens

CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))
PER_CALL_TIMEOUT = int(os.getenv("PER_CALL_TIMEOUT_SEC", "60"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
MAX_TOTAL_RUNTIME_SEC = int(os.getenv("MAX_TOTAL_RUNTIME_SEC", "1800"))

# Hard size caps so one embed always fits
SUMMARY_CHAR_BUDGET = int(os.getenv("SUMMARY_CHAR_BUDGET", "3500"))  # final text budget
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "1100"))  # ~4 chars/token ≈ 4400 chars
MAX_GENERATED_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "600"))

DEFAULT_PROMPT = """
You are an expert analyst producing concise yet information-rich video summaries.
Write the final answer using GitHub-flavored Markdown with the following sections in order:
1. TL;DW — 2-3 sentence overview.
2. Key Points — bulleted list of the most important insights (this should be the longest section).
3. Notable Quotes — bullet list of memorable direct quotes with timestamps in [MM:SS] format.
4. Caveats & Limitations — bullet list of uncertainties, limitations, or missing context.
""".strip()

CHUNK_PROMPT = """
Summarize this portion of a video transcript concisely. Extract key points, quotes, and insights.
Keep the summary brief (max 300 words) while preserving important details.
If you extract any quotes, include their timestamps in [MM:SS] format.
""".strip()

MERGE_PROMPT = """
Combine these partial summaries into a single cohesive summary following the required format:
1. TL;DW — 2-3 sentence overview
2. Key Points — bulleted list
3. Notable Quotes — bullet list with timestamps in [MM:SS] format
4. Caveats & Limitations — bullet list

Maintain all important facts, timestamps, and nuances while eliminating redundancy.
""".strip()


@dataclass
class ChunkSummary:
    index: int
    text: str


def _enforce_char_budget(text: str, budget: int) -> str:
    """Enforce character budget by truncating at paragraph boundary if needed."""
    if len(text) <= budget:
        return text
    # Prefer cutting at a paragraph/bullet boundary
    cut = text.rfind("\n", 0, budget)
    if cut == -1:
        cut = budget
    return text[:cut].rstrip() + "\n\n*(condensed)*"


async def call_openai_with_retry(
    prompt: str,
    client: OpenAI,
    model: str,
    timeout: int = PER_CALL_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    tl=None,
    max_tokens: Optional[int] = None
) -> str:
    """Call OpenAI API with timeout and exponential backoff retry."""
    # Use MAX_GENERATED_TOKENS if max_tokens not explicitly provided
    if max_tokens is None:
        max_tokens = MAX_GENERATED_TOKENS

    for attempt in range(max_retries + 1):
        try:
            payload = dict(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant. Write concise answers only. "
                                "Do NOT include chain-of-thought or hidden reasoning."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            if tl:
                with tl.span("openai_api_call", attempt=attempt):
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.chat.completions.create,
                            **payload
                        ),
                        timeout=timeout
                    )
            else:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.chat.completions.create,
                        **payload
                    ),
                    timeout=timeout
                )
            return response.choices[0].message.content.strip()

        except asyncio.TimeoutError:
            if attempt < max_retries:
                wait = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                await asyncio.sleep(wait)
                continue
            raise RuntimeError(f"OpenAI API call timed out after {max_retries + 1} attempts")

        except Exception as e:
            # Check if it's a context size error - don't retry, fail fast
            error_str = str(e).lower()
            if "context" in error_str and ("exceed" in error_str or "size" in error_str):
                raise RuntimeError(
                    f"Transcript exceeds model context window. "
                    f"This should have been caught earlier - please report this bug."
                ) from e

            if attempt < max_retries:
                wait = (2 ** attempt) + (time.time() % 1)
                await asyncio.sleep(wait)
                continue
            raise


async def summarize_chunk_async(
    chunk_text: str,
    *,
    chunk_index: int,
    total_chunks: int,
    client: OpenAI,
    model: str,
    tl=None,
) -> ChunkSummary:
    """Summarize a single chunk asynchronously."""
    prompt = (
        f"Transcript chunk {chunk_index + 1}.\n"
        "Return ONLY compact bullets (≤5 bullets, ≤12 words each). No TL;DW. "
        "Preserve any timestamps in [MM:SS] format for quotes.\n\n"
        f"Transcript:\n{chunk_text}"
    )
    summary_text = await call_openai_with_retry(
        prompt, client, model, tl=tl,
        max_tokens=max(256, SUMMARY_MAX_TOKENS // 4)
    )
    return ChunkSummary(index=chunk_index, text=summary_text)


async def merge_summaries_async(
    summaries: List[ChunkSummary],
    *,
    client: OpenAI,
    model: str,
    is_final: bool = False,
    tl=None,
) -> str:
    """Merge multiple chunk summaries into one."""
    sorted_chunks = sorted(summaries, key=lambda s: s.index)
    parts = "\n\n".join(f"Part {chunk.index + 1}:\n{chunk.text}" for chunk in sorted_chunks)

    if is_final:
        prompt = (
            f"Combine the partial summaries into one final summary.\n"
            f"HARD OUTPUT LIMIT: <= {SUMMARY_CHAR_BUDGET} characters total. "
            "TL;DW (2 sentences), Key Points (≤10 bullets, ≤15 words each - this should be most of the summary), "
            "Notable Quotes (≤3 bullets with timestamps in [MM:SS] format), Caveats & Limitations (≤3 bullets). "
            "No extra sections like Action Items.\n\n"
            f"Partial summaries:\n{parts}"
        )
        merged = await call_openai_with_retry(prompt, client, model, tl=tl, max_tokens=SUMMARY_MAX_TOKENS)
        return _enforce_char_budget(merged, SUMMARY_CHAR_BUDGET)
    else:
        prompt = f"Combine these summaries concisely:\n{parts}"
        return await call_openai_with_retry(prompt, client, model, tl=tl)


async def hierarchical_summarize(
    transcript: str,
    *,
    client: OpenAI,
    model: str,
    tl=None,
    progress_callback=None
) -> str:
    """
    Hierarchical map-reduce summarization for very long transcripts.

    Strategy:
    1. Split transcript into micro chunks using streaming splitter
    2. Summarize each chunk in parallel (with concurrency limit)
    3. Group partial summaries and merge hierarchically
    4. Final merge to produce structured output
    """
    start_time = time.time()
    logger.info("Starting hierarchical summarization (max runtime: %ds)", MAX_TOTAL_RUNTIME_SEC)

    # Phase 1: Map - summarize micro chunks lazily
    semaphore = asyncio.Semaphore(CONCURRENCY)
    partials: List[ChunkSummary] = []
    tasks = []
    chunk_index = 0
    last_log_time = start_time

    async def summarize_with_limit(idx, chunk):
        async with semaphore:
            if progress_callback:
                await progress_callback(f"Processing chunk {idx + 1}")
            chunk_start = time.time()
            result = await summarize_chunk_async(
                chunk,
                chunk_index=idx,
                total_chunks=-1,  # Unknown until streaming completes
                client=client,
                model=model,
                tl=tl
            )
            chunk_elapsed = time.time() - chunk_start
            logger.debug("Chunk %d processed in %.2fs", idx + 1, chunk_elapsed)
            return result

    # Stream chunks and process them on-the-fly without materializing full list
    for chunk in stream_split(transcript, model, MICRO_TOKENS, overlap_tokens=128):
        # Yield control to event loop to prevent heartbeat blocking
        await asyncio.sleep(0)

        # Create task for this chunk
        task = asyncio.create_task(summarize_with_limit(chunk_index, chunk))
        tasks.append(task)
        chunk_index += 1

        # Process in batches to limit concurrent tasks
        if len(tasks) >= CONCURRENCY:
            batch_results = await asyncio.gather(*tasks)
            partials.extend(batch_results)
            tasks.clear()

            # Log progress every 30 seconds
            now = time.time()
            if now - last_log_time >= 30:
                elapsed = now - start_time
                logger.info("Phase 1 progress: %d chunks processed in %.1fs (%.1f chunks/min)",
                           len(partials), elapsed, len(partials) / (elapsed / 60))
                last_log_time = now

    # Process remaining tasks
    if tasks:
        batch_results = await asyncio.gather(*tasks)
        partials.extend(batch_results)

    total_chunks = chunk_index
    if tl:
        tl.set_metadata("total_chunks_pass1", total_chunks)

    phase1_elapsed = time.time() - start_time
    logger.info("Phase 1 complete: %d chunks processed in %.1fs (avg %.2fs/chunk)",
               total_chunks, phase1_elapsed, phase1_elapsed / total_chunks if total_chunks > 0 else 0)

    # Check runtime
    elapsed = time.time() - start_time
    if elapsed > MAX_TOTAL_RUNTIME_SEC:
        logger.warning("Timeout reached after Phase 1 (%.1fs > %ds), returning partial summary",
                      elapsed, MAX_TOTAL_RUNTIME_SEC)
        # Return best-effort merge
        return await merge_summaries_async(partials[:10], client=client, model=model, is_final=True, tl=tl) + \
               "\n\n*(Partial summary due to time limit)*"

    # Phase 2: Reduce hierarchically
    logger.info("Starting Phase 2: Hierarchical reduction")
    current_level = partials
    level = 0

    while len(current_level) > 1:
        level += 1
        level_start = time.time()
        # Group summaries that fit in MERGE_TOKENS
        # Estimate ~200 tokens per summary (compact bullets), group accordingly
        # Use more aggressive grouping: aim for 6-10 summaries per merge
        estimated_tokens_per_summary = 200
        group_size = max(6, min(10, MERGE_TOKENS // estimated_tokens_per_summary))
        groups = [current_level[i:i+group_size] for i in range(0, len(current_level), group_size)]

        logger.info("Phase 2 Level %d: merging %d summaries into %d groups (group_size=%d)",
                   level, len(current_level), len(groups), group_size)

        if tl:
            tl.set_metadata(f"reduce_level_{level}_groups", len(groups))

        if progress_callback:
            await progress_callback(f"Merging summaries (level {level}, {len(groups)} groups)")

        async def merge_group(group_idx, group):
            async with semaphore:
                merge_start = time.time()
                merged_text = await merge_summaries_async(
                    group,
                    client=client,
                    model=model,
                    is_final=(len(groups) == 1),
                    tl=tl
                )
                merge_elapsed = time.time() - merge_start
                logger.debug("Level %d Group %d merged in %.2fs", level, group_idx + 1, merge_elapsed)
                return ChunkSummary(index=group_idx, text=merged_text)

        if tl:
            with tl.span(f"reduce_phase_level_{level}", groups=len(groups)):
                tasks = [merge_group(i, g) for i, g in enumerate(groups)]
                current_level = await asyncio.gather(*tasks)
        else:
            tasks = [merge_group(i, g) for i, g in enumerate(groups)]
            current_level = await asyncio.gather(*tasks)

        level_elapsed = time.time() - level_start
        logger.info("Phase 2 Level %d complete: %d groups merged in %.1fs (avg %.2fs/group)",
                   level, len(groups), level_elapsed, level_elapsed / len(groups) if len(groups) > 0 else 0)

        # Check runtime again
        elapsed = time.time() - start_time
        if elapsed > MAX_TOTAL_RUNTIME_SEC:
            logger.warning("Timeout reached at Phase 2 Level %d (%.1fs > %ds), returning partial summary",
                          level, elapsed, MAX_TOTAL_RUNTIME_SEC)
            result = current_level[0].text if current_level else "*(Processing interrupted)*"
            return result + "\n\n*(Partial summary due to time limit)*"

    # Final result
    total_elapsed = time.time() - start_time
    logger.info("Hierarchical summarization complete in %.1fs (%.1f minutes)",
               total_elapsed, total_elapsed / 60)
    return current_level[0].text if current_level else ""


def calculate_dynamic_output_tokens(transcript_tokens: int, context_window: int, prompt_overhead: int = 500) -> int:
    """
    Dynamically calculate max output tokens to maximize context usage.

    Formula: max_output = context_window - transcript_tokens - prompt_overhead

    Args:
        transcript_tokens: Number of tokens in the transcript
        context_window: Model's total context window (e.g., 4096)
        prompt_overhead: Estimated tokens for prompts, system messages, formatting (default: 500)

    Returns:
        Maximum output tokens that fits within context window
    """
    # Calculate available space
    available = context_window - transcript_tokens - prompt_overhead

    # Ensure minimum viable output (at least 300 tokens for a summary)
    min_output = 300

    # Cap at reasonable maximum (no need for summaries > 1200 tokens)
    max_output = 1200

    # Return constrained value
    return max(min_output, min(available, max_output))


async def summarize_transcript_async(
    transcript: str,
    *,
    client: OpenAI,
    model: str,
    max_chars_per_chunk: int,  # Legacy parameter, ignored
    tl=None,
    progress_callback=None
) -> str:
    """
    Adaptive summarization strategy:
    - One-shot if transcript fits in context window
    - Hierarchical map-reduce for very long transcripts
    """
    transcript = transcript.strip()

    if not transcript:
        raise ValueError("Transcript is empty")

    # Count tokens
    if tl:
        with tl.span("count_tokens") as node:
            transcript_tokens = count_tokens(model, transcript)
            node.metadata["tokens"] = transcript_tokens
    else:
        transcript_tokens = count_tokens(model, transcript)

    if tl:
        tl.set_metadata("transcript_tokens", transcript_tokens)

    # Dynamic one-shot threshold: can we fit transcript + prompt + output in context?
    # Use generous check: transcript + 500 (prompt) + 300 (min output) < CONTEXT_TOKENS
    one_shot_threshold = CONTEXT_TOKENS - 800  # 4096 - 800 = 3296 tokens max transcript for one-shot

    # Strategy 1: One-shot if it fits
    if ONE_SHOT_ENABLED and transcript_tokens <= one_shot_threshold:
        # Dynamically calculate optimal output tokens
        dynamic_max_tokens = calculate_dynamic_output_tokens(
            transcript_tokens=transcript_tokens,
            context_window=CONTEXT_TOKENS,
            prompt_overhead=500
        )

        one_shot_prompt = (
            f"{DEFAULT_PROMPT}\n\n"
            f"HARD OUTPUT LIMIT: Respond in <= {SUMMARY_CHAR_BUDGET} characters total. "
            f"Keep TL;DW (2 sentences), Key Points (≤10 bullets, ≤15 words each - this should be most of the summary), "
            f"Notable Quotes (≤3 short bullets with timestamps in [MM:SS] format), Caveats & Limitations (≤3 bullets). "
            f"No extra sections like Action Items.\n\n"
            f"Transcript:\n{transcript}"
        )
        if tl:
            with tl.span("one_shot_summarize"):
                result = await call_openai_with_retry(one_shot_prompt, client, model, tl=tl, max_tokens=dynamic_max_tokens)
        else:
            result = await call_openai_with_retry(one_shot_prompt, client, model, tl=tl, max_tokens=dynamic_max_tokens)
        return _enforce_char_budget(result, SUMMARY_CHAR_BUDGET)

    # Strategy 2: Hierarchical map-reduce
    if tl:
        with tl.span("hierarchical_summarize"):
            result = await hierarchical_summarize(
                transcript,
                client=client,
                model=model,
                tl=tl,
                progress_callback=progress_callback
            )
    else:
        result = await hierarchical_summarize(
            transcript,
            client=client,
            model=model,
            tl=tl,
            progress_callback=progress_callback
        )

    return result


# Backward compatibility: sync wrapper
def summarize_transcript(
    transcript: str,
    *,
    client: OpenAI,
    model: str,
    max_chars_per_chunk: int,
    tl=None,
) -> str:
    """Synchronous wrapper for backward compatibility."""
    return asyncio.run(summarize_transcript_async(
        transcript,
        client=client,
        model=model,
        max_chars_per_chunk=max_chars_per_chunk,
        tl=tl
    ))


def compute_summary_cache_key(*, video_id: str, prompt: str, model: str) -> str:
    payload = f"{video_id}:{model}:{prompt.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
