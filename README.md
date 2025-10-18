# TL;DW Discord Bot

A production-ready Discord bot that turns any YouTube video into a structured, actionable summary. The bot automatically fetches YouTube captions and produces multi-section summaries powered by OpenAI's Chat API.

## Features

- `/summarize` slash command accepts a YouTube URL or video ID.
- Transcript acquisition pipeline (captions-only, no audio download):
  1. YouTube captions via [`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/) (manual → auto-generated → translated)
  2. Direct subtitle extraction via [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) (fallback for subtitle parsing)
- Structured Markdown summaries with TL;DR, key points, timestamped outline, notable quotes, action items, and caveats.
- Automatic chunking for long transcripts and final merge summarization.
- SQLite caching for transcripts and summaries.
- Discord-safe message splitting with automatic `.txt` attachments for long outputs.
- Docker-ready deployment with ffmpeg and all dependencies.

## Project Layout

```
bot/
├── __init__.py
├── main.py           # Discord bot + slash command orchestration
├── storage.py        # Thread-safe SQLite cache wrapper
├── summarize.py      # Transcript chunking & OpenAI-powered summarization
└── transcripts.py    # Transcript acquisition pipeline & fallbacks
requirements.txt
Dockerfile
.env.example
README.md
```

## Prerequisites

- Python 3.11+
- Discord bot token with the `applications.commands` scope enabled
- OpenAI API key with access to `gpt-4.1-mini` (for summarization only)

## Configuration

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

| Variable | Description |
| --- | --- |
| `DISCORD_BOT_TOKEN` | Discord bot token (required) |
| `OPENAI_API_KEY` | OpenAI API key for summarization (required) |
| `OPENAI_SUMMARY_MODEL` | Chat model for summarization (default `gpt-4.1-mini`) |
| `CACHE_DB` | SQLite database path (default `cache.sqlite3`) |
| `MAX_CHARS_PER_CHUNK` | Max characters per transcript chunk before summarization (default `12000`) |
| `MAX_DISCORD_MSG_CHARS` | Max characters per Discord message chunk (default `1900`) |
| `YT_COOKIES` | Optional path to YouTube cookies.txt for age-restricted videos |
| `YT_FORCE_IPV4` | Force IPv4 for YouTube requests (default `1`) |
| `RATE_RPS` | Rate limit: requests per second (default `1.0`) |
| `USER_QUOTA_MAX` | Per-user quota: max videos per window (default `5`) |
| `CHAN_QUOTA_MAX` | Per-channel quota: max videos per window (default `20`) |
| `CACHE_DB_PATH` | Persistent transcript cache path (default `/app/cache.db`) |

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

## Notes

- The bot only fetches YouTube captions; it does not download audio or use Whisper transcription.
- If a video has no captions, the bot will return an error message.
- The bot automatically labels the transcript source in responses (e.g., `yt-api-manual`, `yt-dlp`).
- Large summaries are attached as `.txt` files for convenience.
- Rate limiting and quotas prevent YouTube API abuse (configurable via environment variables).
- Ensure your Discord bot has the `Message Content Intent` if you plan to extend functionality beyond slash commands.

