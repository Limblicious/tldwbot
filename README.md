# TL;DW Discord Bot

A production-ready Discord bot that turns any YouTube video into a structured, actionable summary. The bot automatically fetches YouTube captions and produces multi-section summaries powered by OpenAI's Chat API.

## Features

- `/summarize` slash command accepts a YouTube URL or video ID.
- Transcript acquisition pipeline (captions-only, no audio download):
  1. Direct subtitle extraction via [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) (primary method)
  2. YouTube captions via [`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/) (manual → auto-generated → translated) - currently disabled due to XML parsing issues
- **Timestamp preservation**: Transcripts maintain VTT timestamps in `[MM:SS]` format for quote attribution.
- Structured Markdown summaries with:
  - **TL;DW** — 2-3 sentence overview
  - **Key Points** — bulleted list of important insights
  - **Notable Quotes** — memorable quotes with timestamps in `[MM:SS]` format
  - **Caveats & Limitations** — uncertainties or missing context
- **Adaptive summarization strategy**:
  - One-shot summarization for short videos (fits in context window)
  - Hierarchical map-reduce for long videos with streaming chunk processing
- Advanced rate limiting and quota management to prevent YouTube API abuse.
- Persistent SQLite caching for transcripts and summaries with configurable TTL.
- Circuit breaker pattern for handling YouTube 429 rate limits.
- Discord-safe message splitting with automatic `.txt` attachments for long outputs.
- Docker-ready deployment with ffmpeg and all dependencies.

## Project Layout

```
bot/
├── __init__.py
├── main.py           # Discord bot + slash command orchestration
├── storage.py        # Thread-safe SQLite cache wrapper
├── summarize.py      # Adaptive summarization with hierarchical map-reduce
├── transcripts.py    # Transcript acquisition pipeline with rate limiting
├── tokens.py         # Token counting utilities (tiktoken-based)
├── stream_split.py   # Streaming text chunking for large transcripts
├── limiters.py       # Rate limiting, quotas, and circuit breaker
├── cache_sqlite.py   # Persistent SQLite cache with TTL
├── timetest.py       # Performance profiling utilities (dev only)
└── time_run.py       # Benchmarking script (dev only)
requirements.txt
Dockerfile
docker-compose.yml
.env.example
CLAUDE.md             # Development guardrails
README.md
```

## Prerequisites

- Python 3.11+
- Discord bot token with the `applications.commands` scope enabled
- Local LLM server (e.g., llama.cpp, vLLM, or OpenAI-compatible endpoint) running DeepSeek-R1-Distill-Qwen-7B or similar model

## Configuration

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

| Variable | Description |
| --- | --- |
| **Required** | |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `OPENAI_API_KEY` | API key for LLM endpoint (use `local` for local servers) |
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible API endpoint |
| **Summarization** | |
| `OPENAI_SUMMARY_MODEL` | Model name or path (e.g., DeepSeek-R1-Distill-Qwen-7B) |
| `ONE_SHOT_ENABLED` | Enable one-shot summarization for short videos (default `1`) |
| `CONTEXT_TOKENS` | LLM context window size (default `128000`) |
| `SUMMARY_CHAR_BUDGET` | Max summary character budget (default `3500`) |
| `SUMMARY_MAX_TOKENS` | Max tokens for summary generation (default `1100`) |
| `LLM_CONCURRENCY` | Concurrent LLM calls during chunking (default `3`) |
| **Caching** | |
| `CACHE_DB` | SQLite database path for legacy storage (default `cache.sqlite3`) |
| `CACHE_DB_PATH` | Persistent transcript cache path (default `/app/cache.db`) |
| `PERSIST_TTL_SECS` | Persistent cache TTL in seconds (default `86400`) |
| `YT_CACHE_TTL` | YouTube transcript cache TTL in seconds (default `7200`) |
| **YouTube API** | |
| `YT_COOKIES` | Optional path to YouTube cookies.txt for age-restricted videos |
| `YT_FORCE_IPV4` | Force IPv4 for YouTube requests (default `1`) |
| `YT_UA` | User-Agent string for YouTube requests |
| `YT_REQ_SLEEP` | Seconds between external YouTube requests (default `0`) |
| **Rate Limiting** | |
| `RATE_RPS` | Token bucket: requests per second (default `1.0`) |
| `RATE_BURST` | Token bucket: burst size (default `2`) |
| `USER_QUOTA_MAX` | Per-user quota: max videos per window (default `5`) |
| `USER_QUOTA_WINDOW` | Per-user quota window in seconds (default `600`) |
| `CHAN_QUOTA_MAX` | Per-channel quota: max videos per window (default `20`) |
| `CHAN_QUOTA_WINDOW` | Per-channel quota window in seconds (default `600`) |
| **Circuit Breaker** | |
| `CB_429_THRESHOLD` | Consecutive 429s to open circuit breaker (default `3`) |
| `CB_OPEN_SECS` | Circuit breaker open duration in seconds (default `1800`) |
| `CB_HALF_PROBE_SECS` | Half-open probe delay in seconds (default `120`) |
| `NEG_CACHE_TTL` | Per-video cooldown after 429 in seconds (default `3600`) |
| **Discord** | |
| `MAX_DISCORD_MSG_CHARS` | Max characters per Discord message chunk (default `1900`) |

## Local Development

1. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create `.env` using the template and populate required values.
3. Run the bot:
   ```bash
   python -m bot.main
   ```
4. Invite the bot to your server with the `applications.commands` scope and use `/summarize`.

## Docker Usage

The recommended way to run tldwbot in production is with Docker Compose:

1. Copy `.env.example` to `.env` and configure your tokens and API keys.
2. Start the bot:
   ```bash
   docker-compose up -d
   ```
3. View logs:
   ```bash
   docker-compose logs -f
   ```
4. Stop the bot:
   ```bash
   docker-compose down
   ```

**Data Persistence**: The `./data` directory is mounted to persist the SQLite cache. This directory is created automatically on first run.

## Docker Deployment

Build and run the container:

```bash
docker build -t tldw-bot .
docker run --env-file .env tldw-bot
```

Mount a volume for persistent caching if desired:

```bash
docker run \
  --env-file .env \
  -v $(pwd)/cache.sqlite3:/app/cache.sqlite3 \
  tldw-bot
```

## SQLite Cache Schema

- **transcripts**: `video_id`, `source`, `text`, `created_at`
- **summaries**: `video_id`, `prompt_hash`, `model`, `summary`, `created_at`

## Performance Features

- **Streaming chunk processing**: Uses `stream_split.py` to process large transcripts without loading the entire text into memory
- **Hierarchical map-reduce**: Breaks down very long videos into chunks, summarizes each in parallel, then merges hierarchically
- **Token counting**: Accurate token counting with `tiktoken` (with character-based fallback)
- **Concurrent LLM calls**: Configurable concurrency for parallel chunk summarization
- **Persistent caching**: SQLite-based caching with TTL to avoid redundant API calls

## Rate Limiting & Protection

- **Token bucket rate limiter**: Smooth out request bursts to YouTube
- **Per-user and per-channel quotas**: Prevent individual users or channels from overwhelming the bot
- **Circuit breaker for 429s**: Automatically backs off when YouTube rate limits are hit
- **Negative caching**: Per-video cooldown after rate limit errors
- **Stable jitter**: Deterministic jitter based on video ID to spread out requests

## Notes

- The bot **only** fetches YouTube captions; it does not download audio or use Whisper transcription.
- If a video has no captions, the bot will return an error message: "No captions available for this video."
- The bot automatically labels the transcript source in responses (e.g., `cache:yt-dlp`, `yt-dlp`).
- **Timestamps are preserved** from VTT captions and included in notable quotes as `[MM:SS]` format.
- Large summaries are attached as `.txt` files for convenience.
- Rate limiting and quotas prevent YouTube API abuse (all configurable via environment variables).
- Ensure your Discord bot has the `Message Content Intent` if you plan to extend functionality beyond slash commands.
- See `CLAUDE.md` for development guardrails and best practices.

