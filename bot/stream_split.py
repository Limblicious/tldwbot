# bot/stream_split.py
from __future__ import annotations
import math
try:
    import tiktoken
except Exception:
    tiktoken = None

def approx_tokens(s: str) -> int:
    """Approximate token count using ~4 chars/token heuristic."""
    return math.ceil(len(s) / 4)

def stream_split(text: str, model: str, max_tokens: int, overlap_tokens: int = 128, block_chars: int = 20000):
    """
    Generator that yields chunks ≤ max_tokens without encoding the entire transcript at once.
    Works in blocks to keep memory bounded; uses tiktoken when available, else char-heuristic.

    Args:
        text: The text to split
        model: Model name for tiktoken encoding
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        block_chars: Character block size for streaming (keeps memory bounded)

    Yields:
        String chunks of approximately max_tokens each
    """
    if not text:
        return

    i = 0
    n = len(text)
    enc = None

    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    carry_ids = []

    while i < n:
        # Read next block
        block = text[i:min(i+block_chars, n)]
        i += block_chars

        if enc:
            # Token-based splitting with bounded carry
            try:
                # Encode block and prepend carry (which is already bounded)
                block_ids = enc.encode(block)
                ids = carry_ids + block_ids if carry_ids else block_ids

                # Emit chunks while we have enough tokens
                pos = 0
                L = len(ids)
                while pos < L:
                    take = min(max_tokens, L - pos)
                    chunk_ids = ids[pos:pos + take]
                    yield enc.decode(chunk_ids)

                    # Advance position with overlap
                    step = take - min(overlap_tokens, take)
                    if step <= 0:
                        # Avoid infinite loop if overlap >= take
                        step = take
                    pos += step

                # Keep only last overlap_tokens for next block (bounded!)
                carry_len = min(overlap_tokens, L)
                carry_ids = ids[-carry_len:] if carry_len > 0 else []

            except Exception:
                # Fallback to char-based if encoding fails
                enc = None
                carry_ids = []
                # Process this block with char-based method below

        if not enc:
            # Character-based splitting (heuristic: 4 chars per token)
            approx_chars = max_tokens * 4
            overlap_chars = overlap_tokens * 4
            pos = 0
            L = len(block)

            while pos < L:
                take = min(approx_chars, L - pos)
                chunk = block[pos:pos + take]
                yield chunk

                # Advance with overlap
                step = take - min(overlap_chars, take)
                if step <= 0:
                    step = take
                pos += step

    # Yield final carry if any
    if enc and carry_ids:
        try:
            yield enc.decode(carry_ids)
        except Exception:
            pass
