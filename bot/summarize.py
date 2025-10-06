"""Transcript summarization utilities."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI

DEFAULT_PROMPT = """
You are an expert analyst producing concise yet information-rich video summaries.
Write the final answer using GitHub-flavored Markdown with the following sections in order:
1. TL;DR — 2-3 sentence overview.
2. Key Points — bulleted list of the most important insights.
3. Timestamped Outline — ordered list of notable moments with timestamps where available.
4. Notable Quotes — bullet list of memorable direct quotes.
5. Action Items — bullet list of recommended follow-up actions.
6. Caveats & Limitations — bullet list of uncertainties, limitations, or missing context.
Each heading should be formatted as `## Heading`.
If a section has no content, write `- None reported.` under that heading.
""".strip()


@dataclass
class ChunkSummary:
    index: int
    text: str


def chunk_transcript(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Try to break on sentence boundary
        slice_text = text[start:end]
        if end < len(text):
            period_pos = slice_text.rfind(". ")
            newline_pos = slice_text.rfind("\n")
            split_pos = max(period_pos, newline_pos)
            if split_pos != -1 and split_pos > max_chars // 2:
                end = start + split_pos + 1
                slice_text = text[start:end]
        chunks.append(slice_text.strip())
        start = end
    return chunks


def call_openai_summary(prompt: str, client: OpenAI, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that strictly follows formatting instructions.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def summarize_chunk(
    chunk_text: str,
    *,
    chunk_index: int,
    total_chunks: int,
    base_prompt: str,
    client: OpenAI,
    model: str,
) -> ChunkSummary:
    prompt = (
        f"{base_prompt}\n\n"
        f"Transcript chunk {chunk_index + 1} of {total_chunks}."
        "\nFocus on accurately summarizing only this portion while preserving timestamps if present."
        "\n\nTranscript:\n" + chunk_text
    )
    summary_text = call_openai_summary(prompt, client, model)
    return ChunkSummary(index=chunk_index, text=summary_text)


def merge_summaries(
    summaries: Iterable[ChunkSummary],
    *,
    base_prompt: str,
    client: OpenAI,
    model: str,
) -> str:
    sorted_chunks = sorted(summaries, key=lambda s: s.index)
    parts = "\n\n".join(
        f"Chunk {chunk.index + 1} summary:\n{chunk.text}" for chunk in sorted_chunks
    )
    prompt = (
        f"{base_prompt}\n\n"
        "Combine the following partial summaries into a single cohesive summary."
        " Maintain all important facts, timestamps, and nuances while eliminating redundancy."
        " Ensure the final response follows the required section structure."
        f"\n\nPartial summaries:\n{parts}"
    )
    return call_openai_summary(prompt, client, model)


def summarize_transcript(
    transcript: str,
    *,
    client: OpenAI,
    model: str,
    max_chars_per_chunk: int,
    prompt_override: Optional[str] = None,
) -> str:
    base_prompt = (prompt_override or DEFAULT_PROMPT).strip()
    chunks = chunk_transcript(transcript, max_chars_per_chunk)
    if not chunks:
        raise ValueError("Transcript is empty")

    if len(chunks) == 1:
        return summarize_chunk(
            chunks[0],
            chunk_index=0,
            total_chunks=1,
            base_prompt=base_prompt,
            client=client,
            model=model,
        ).text

    partials = [
        summarize_chunk(
            chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            base_prompt=base_prompt,
            client=client,
            model=model,
        )
        for i, chunk in enumerate(chunks)
    ]
    return merge_summaries(
        partials,
        base_prompt=base_prompt,
        client=client,
        model=model,
    )


def compute_summary_cache_key(
    *, video_id: str, prompt: str, model: str
) -> str:
    payload = f"{video_id}:{model}:{prompt.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

