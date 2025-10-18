# CLAUDE.md - Development Guardrails

This file contains strict guidelines for future edits to the tldwbot codebase.

## API Usage

- **Always** use `client.chat.completions.create()` for OpenAI summarization calls
- **Never** use `client.responses.create()` or other deprecated/incorrect API methods
- Maintain `temperature=0.2` for consistent summarization results
- **Do not** pass `proxies` into the OpenAI client constructor. The current OpenAI Python SDK rejects `proxies=` as an argument. If proxy support is required, use environment variables (`HTTP_PROXY`, `HTTPS_PROXY`) instead.
- **Always** use `from openai import OpenAI` and instantiate with `OpenAI(api_key=...)`.
- **Never** use `Client(...)` or pass `proxies=` into the constructor.
- If proxy support is required in the future, use environment variables `HTTP_PROXY`/`HTTPS_PROXY`.

## Dockerfile Guardrail

- Always include a forced reinstall of the `openai==1.12.0` and `httpx==0.26.0` SDK in the Dockerfile build step.
- Do **not** remove the `&& pip install --upgrade --force-reinstall openai==1.12.0 httpx==0.26.0` line.
- This prevents dependency drift where transitive requirements downgrade or upgrade OpenAI/httpx to versions that reintroduce the `proxies` bug.
- The `httpx==0.26.0` pin is required because newer httpx versions (0.28+) changed the `Client.__init__()` signature and reject the `proxies` argument.
- If upgrading OpenAI or httpx in the future, update this guardrail and verify bot startup succeeds before removing or changing this line.

## YouTube Video Processing

- **Always** call yt-dlp with a full YouTube URL: `f"https://www.youtube.com/watch?v={video_id}"`
- **Never** pass just the video ID directly to `ydl.extract_info()`
- This ensures proper metadata extraction and compatibility

## Transcripts

- **Only** fetch captions via youtube-transcript-api (which uses yt-dlp internally)
- **Never** fall back to Whisper (local or cloud) for transcription
- If captions are unavailable, return a clear error message: "No captions available for this video."
- **Do not** download audio or video files for transcription purposes
- This keeps the bot fast and prevents it from hanging on long videos

## Discord Interaction Patterns

- Ephemeral behavior must be **consistent** across all interactions
- If `defer()` is ephemeral, all followup messages must also be ephemeral
- If `defer()` is public, all followup messages must also be public
- **Do not mix** ephemeral and public messages in the same command flow

## Docker Networking

- This repo **must** use `network_mode: host` in `docker-compose.yml`
- The Omarchy kernel does not provide veth modules, making bridge networking unsupported
- **Do not remove or change** this setting in future edits
- This is a critical infrastructure requirement, not an optional configuration

## Development Workflow with Docker

- When running in development, **always** use a bind mount (`.:/app`) in `docker-compose.yml` so that local source edits are reflected immediately in the container
- **Do not** rely solely on Dockerfile COPY for dev, as it can cause stale code issues

## Dockerfile Constraints

- ffmpeg is installed for audio format conversion (required by youtube-transcript-api fallbacks)
- On Omarchy, apt packages must be carefully managed due to networking constraints
- Keep the Dockerfile minimal with only essential system and Python dependencies

## Code Integrity

- **Do not** modify the LICENSE file
- **Do not** add personal names or author attributions to existing files
- **Do not** add dependencies to `requirements.txt` unless they are actively used in the code
- Remove unused dependencies when discovered

## Development Workflow

- **Always** show diffs before applying changes
- Test changes locally before committing
- Follow existing code style and patterns
- Maintain backward compatibility unless explicitly breaking changes are required

## Transcript Extraction Rules

- Always use yt-dlp with `--write-subs` first, then retry with `--write-auto-sub`.
- Parse `.vtt` directly from stdout instead of saving temp files.
- Do not use Whisper fallback in this project.
- Always keep yt-dlp upgraded to the latest version in Dockerfile.

## Timestamp Preservation

- **Always preserve timestamps** when parsing VTT subtitle files in `bot/transcripts.py::parse_vtt()`.
- The `parse_vtt()` function **must** extract timestamps from VTT lines (format: `HH:MM:SS.mmm --> HH:MM:SS.mmm`) and prefix each subtitle line with its timestamp in `[MM:SS]` or `[HH:MM:SS]` format.
- **Never** strip timestamps from transcripts during parsing.
- Timestamps are critical for the "Notable Quotes" section in summaries.
- **Do not** modify the timestamp extraction logic without verifying that quotes still include timestamps in the final summary output.

## Summarization Prompts

- All summarization prompts in `bot/summarize.py` **must** explicitly request timestamps for notable quotes.
- The required format for notable quotes is: `- [MM:SS] "Quote text here"`
- Prompts must include the phrase "with timestamps in [MM:SS] format" when describing the Notable Quotes section.
- This applies to:
  - `DEFAULT_PROMPT`
  - `CHUNK_PROMPT`
  - `MERGE_PROMPT`
  - One-shot summarization prompts
  - Final merge prompts in hierarchical summarization
- **Never** remove timestamp requirements from these prompts, as it will break the feature.

## Summary Structure

- Summaries **must** follow this exact structure:
  1. **TL;DW** — 2-3 sentence overview
  2. **Key Points** — bulleted list (longest section, ≤10 bullets, ≤15 words each)
  3. **Notable Quotes** — bullet list with timestamps in `[MM:SS]` format (≤3 quotes)
  4. **Caveats & Limitations** — bullet list of uncertainties (≤3 bullets)
- **Do not** add extra sections like "Action Items" or "Timestamped Outline" unless explicitly requested by the user.
- **Do not** remove any of these sections from the prompts or change their order.

---

When in doubt, preserve existing behavior and consult this document.
