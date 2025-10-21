"""Intelligent transcript sampling for very long videos."""
from __future__ import annotations
import logging
import re
from typing import List, Tuple

logger = logging.getLogger("tldwbot")


def sample_transcript_intelligently(
    transcript: str,
    max_tokens: int,
    tokens_per_char: float = 0.5,  # More realistic: ~2 chars per token
    strategy: str = "balanced"  # "balanced", "dense", or "sparse"
) -> Tuple[str, str]:
    """
    Sample a transcript to fit within max_tokens while preserving key information.

    Strategy Options:
    - "balanced": 20% start, 60% middle (sampled), 20% end
    - "dense": More aggressive sampling, better for very long videos
    - "sparse": Less aggressive, preserves more detail

    Processing:
    1. Take beginning (context setting)
    2. Sample evenly throughout middle
    3. Take end (conclusions)
    4. Preserve timestamps for quotes
    5. Iteratively reduce if still too large

    Args:
        transcript: Full transcript with timestamps
        max_tokens: Maximum tokens to include
        tokens_per_char: Approximate tokens per character ratio (default 0.5)
        strategy: Sampling strategy ("balanced", "dense", or "sparse")

    Returns:
        Tuple of (sampled_transcript, strategy_used)
    """
    logger.info("Sampling transcript: %d chars, target %d tokens, strategy=%s",
               len(transcript), max_tokens, strategy)
    # Start aggressive: aim for 80% of target to leave room for overhead
    max_chars = int(max_tokens / tokens_per_char * 0.8)

    # If already fits, return as-is
    if len(transcript) <= max_chars:
        logger.info("Transcript fits within budget, no sampling needed")
        return transcript, "none"

    # Split into lines (preserving timestamps)
    lines = transcript.split('\n')
    total_lines = len(lines)

    if total_lines < 10:
        # Very short, just truncate
        logger.warning("Very short transcript, truncating to %d chars", max_chars)
        return transcript[:max_chars], "truncate"

    # Strategy-based allocation
    if strategy == "dense":
        # More aggressive: 15% start, 70% middle (heavily sampled), 15% end
        start_chars = int(max_chars * 0.15)
        middle_chars = int(max_chars * 0.70)
        end_chars = int(max_chars * 0.15)
        sample_ratio = 0.3  # Take only 30% of middle
    elif strategy == "sparse":
        # Less aggressive: 25% start, 50% middle (lightly sampled), 25% end
        start_chars = int(max_chars * 0.25)
        middle_chars = int(max_chars * 0.50)
        end_chars = int(max_chars * 0.25)
        sample_ratio = 0.7  # Take 70% of middle
    else:  # balanced
        # Default: 20% start, 60% middle, 20% end
        start_chars = int(max_chars * 0.2)
        middle_chars = int(max_chars * 0.6)
        end_chars = int(max_chars * 0.2)
        sample_ratio = 0.5  # Take 50% of middle

    # Take beginning lines
    start_lines = []
    char_count = 0
    for line in lines:
        if char_count + len(line) > start_chars:
            break
        start_lines.append(line)
        char_count += len(line) + 1  # +1 for newline

    # Take end lines
    end_lines = []
    char_count = 0
    for line in reversed(lines):
        if char_count + len(line) > end_chars:
            break
        end_lines.insert(0, line)
        char_count += len(line) + 1

    # Sample middle evenly
    start_idx = len(start_lines)
    end_idx = total_lines - len(end_lines)
    middle_lines_total = end_idx - start_idx

    if middle_lines_total <= 0:
        # Edge case: start and end overlap
        return '\n'.join(start_lines + end_lines)

    # Calculate sampling rate to fit middle_chars
    middle_available_lines = lines[start_idx:end_idx]

    # Estimate chars per line
    avg_chars_per_line = sum(len(l) for l in middle_available_lines) / len(middle_available_lines)
    target_lines = int(middle_chars / avg_chars_per_line * sample_ratio)

    # Sample evenly
    if target_lines >= len(middle_available_lines):
        middle_lines = middle_available_lines
    else:
        step = len(middle_available_lines) / target_lines
        middle_lines = [
            middle_available_lines[int(i * step)]
            for i in range(target_lines)
        ]

    # Combine with clear markers
    sampled = (
        start_lines +
        ['', f'[... {len(middle_available_lines) - len(middle_lines)} middle lines sampled using {strategy} strategy ...]', ''] +
        middle_lines +
        ['', '[... continuing to end ...]', ''] +
        end_lines
    )

    result = '\n'.join(sampled)

    # Final safety truncation
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[... truncated for length ...]"

    reduction_pct = (1 - len(result) / len(transcript)) * 100
    logger.info("Sampling complete: %d chars -> %d chars (%.1f%% reduction)",
               len(transcript), len(result), reduction_pct)

    return result, strategy


def sample_transcript_content_aware(
    transcript: str,
    max_tokens: int,
    tokens_per_char: float = 0.5
) -> Tuple[str, str]:
    """
    Advanced content-aware sampling that prioritizes information-dense sections.

    Strategy:
    1. Score each line/paragraph by information density (length, unique words, timestamps)
    2. Always include start (15%) and end (15%)
    3. Sample middle (70%) based on content scores
    4. Preserve temporal distribution

    Args:
        transcript: Full transcript with timestamps
        max_tokens: Maximum tokens to include
        tokens_per_char: Approximate tokens per character ratio

    Returns:
        Tuple of (sampled_transcript, "content-aware")
    """
    logger.info("Content-aware sampling: %d chars, target %d tokens", len(transcript), max_tokens)
    max_chars = int(max_tokens / tokens_per_char * 0.8)

    if len(transcript) <= max_chars:
        return transcript, "none"

    lines = transcript.split('\n')
    total_lines = len(lines)

    if total_lines < 20:
        return sample_transcript_intelligently(transcript, max_tokens, tokens_per_char, "balanced")

    # Allocate: 15% start, 70% content-scored middle, 15% end
    start_pct = 0.15
    end_pct = 0.15
    middle_pct = 0.70

    start_line_count = max(1, int(total_lines * start_pct))
    end_line_count = max(1, int(total_lines * end_pct))

    start_lines = lines[:start_line_count]
    end_lines = lines[-end_line_count:]
    middle_lines = lines[start_line_count:-end_line_count]

    # Score middle lines by information density
    scored_lines = []
    for idx, line in enumerate(middle_lines):
        score = 0
        # Longer lines often have more content
        score += len(line) * 0.1
        # Lines with timestamps (quotes) are valuable
        if re.search(r'\[\d+:\d+\]', line):
            score += 50
        # Unique word diversity
        words = set(line.lower().split())
        score += len(words) * 2
        # Penalize very short lines (likely filler)
        if len(line) < 20:
            score *= 0.5

        scored_lines.append((idx, line, score))

    # Sort by score, take top lines
    scored_lines.sort(key=lambda x: x[2], reverse=True)

    # Calculate how many middle lines to keep
    start_chars = sum(len(l) + 1 for l in start_lines)
    end_chars = sum(len(l) + 1 for l in end_lines)
    middle_budget = max_chars - start_chars - end_chars - 200  # Reserve for markers

    selected_middle = []
    current_chars = 0
    for idx, line, score in scored_lines:
        if current_chars + len(line) > middle_budget:
            break
        selected_middle.append((idx, line))
        current_chars += len(line) + 1

    # Re-sort by original index to maintain temporal order
    selected_middle.sort(key=lambda x: x[0])
    middle_lines_final = [line for idx, line in selected_middle]

    # Combine
    sampled = (
        start_lines +
        ['', f'[... {len(middle_lines) - len(middle_lines_final)} middle lines filtered by content score ...]', ''] +
        middle_lines_final +
        ['', '[... end section ...]', ''] +
        end_lines
    )

    result = '\n'.join(sampled)

    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[... truncated ...]"

    reduction_pct = (1 - len(result) / len(transcript)) * 100
    logger.info("Content-aware sampling complete: %d chars -> %d chars (%.1f%% reduction)",
               len(transcript), len(result), reduction_pct)

    return result, "content-aware"


def estimate_processing_time(
    token_count: int,
    tokens_per_second: float = 30.0,
    api_overhead: float = 1.5,
    concurrency: int = 4,
    micro_tokens: int = 699
) -> float:
    """
    Estimate processing time in seconds.

    Args:
        token_count: Number of tokens in transcript
        tokens_per_second: Generation speed
        api_overhead: Seconds per API call overhead
        concurrency: Parallel processing limit
        micro_tokens: Chunk size

    Returns:
        Estimated seconds
    """
    num_chunks = max(1, (token_count + micro_tokens - 1) // micro_tokens)

    # Chunk processing time
    chunk_output_tokens = num_chunks * 256
    chunk_time = (chunk_output_tokens / tokens_per_second) + (num_chunks * api_overhead / concurrency)

    # Merge time (rough estimate)
    import math
    if num_chunks > 1:
        merge_levels = math.ceil(math.log2(num_chunks))
        merge_calls = num_chunks // 4  # Rough estimate
        merge_time = (merge_calls * 600 / tokens_per_second) + (merge_calls * api_overhead / concurrency)
    else:
        merge_time = 0

    return chunk_time + merge_time


def should_sample_transcript(
    token_count: int,
    max_processing_seconds: int = 180,
    tokens_per_second: float = 30.0,
    concurrency: int = 4
) -> bool:
    """
    Determine if transcript should be sampled to meet time budget.

    Args:
        token_count: Number of tokens in transcript
        max_processing_seconds: Hard time limit (default 3 minutes)
        tokens_per_second: Generation speed
        concurrency: Parallel processing limit

    Returns:
        True if sampling is needed
    """
    estimated_time = estimate_processing_time(
        token_count,
        tokens_per_second=tokens_per_second,
        concurrency=concurrency
    )

    # Add 20% buffer
    return estimated_time > (max_processing_seconds * 0.8)


def calculate_sample_budget(
    max_processing_seconds: int = 180,
    tokens_per_second: float = 30.0,
    concurrency: int = 4,
    micro_tokens: int = 699
) -> int:
    """
    Calculate maximum transcript tokens to fit in time budget.

    Returns:
        Maximum tokens to process
    """
    # Work backwards from time budget
    # Assume optimal case: target 80% of time budget
    target_time = max_processing_seconds * 0.8

    # Rough formula: tokens ≈ (target_time * tokens_per_second * concurrency) / 2
    # The /2 accounts for merge overhead
    max_tokens = int((target_time * tokens_per_second * concurrency) / 2.5)

    return max_tokens
