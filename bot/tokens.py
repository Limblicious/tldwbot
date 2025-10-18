# bot/tokens.py
from __future__ import annotations
import math
try:
    import tiktoken
except Exception:
    tiktoken = None

def _naive_count(text: str) -> int:
    return math.ceil(len(text) / 4)

def count_tokens(model: str, text: str) -> int:
    """
    Count tokens in text. For very large texts (>500K chars), uses sampling
    to avoid memory/time issues with encoding the entire text.
    """
    if not text:
        return 0

    # For very large texts, use sampling to estimate
    if len(text) > 500000:
        # Sample first 100k, middle 100k, last 100k chars
        sample_size = 100000
        samples = [
            text[:sample_size],
            text[len(text)//2 - sample_size//2:len(text)//2 + sample_size//2],
            text[-sample_size:]
        ]

        if tiktoken:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            try:
                total_sample_tokens = sum(len(enc.encode(s)) for s in samples)
                total_sample_chars = sum(len(s) for s in samples)
                ratio = total_sample_tokens / total_sample_chars if total_sample_chars > 0 else 0.25
                return int(len(text) * ratio)
            except Exception:
                pass

        return _naive_count(text)

    # For normal-sized texts, count accurately
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return _naive_count(text)

def split_by_tokens(text: str, model: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Greedy token-aware splitter that tries to cut on sentence boundaries."""
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        chunks = []
        step = max_tokens - max(0, overlap_tokens)
        for i in range(0, len(ids), step):
            sub = ids[i:i+max_tokens]
            chunks.append(enc.decode(sub))
        return chunks
    # fallback by characters (approximate)
    approx_char = max_tokens * 4
    step = approx_char - max(0, overlap_tokens) * 4
    return [text[i:i+approx_char] for i in range(0, len(text), max(step, 1))]
