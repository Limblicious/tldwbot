# CLAUDE.md - Development Guardrails

This file contains strict guidelines for future edits to the tldwbot codebase.

## API Usage

- **Always** use `client.chat.completions.create()` for OpenAI summarization calls
- **Never** use `client.responses.create()` or other deprecated/incorrect API methods
- Maintain `temperature=0.2` for consistent summarization results
- **Do not** pass `proxies` into the OpenAI client constructor. The current OpenAI Python SDK rejects `proxies=` as an argument. If proxy support is required, use environment variables (`HTTP_PROXY`, `HTTPS_PROXY`) instead.

## YouTube Video Processing

- **Always** call yt-dlp with a full YouTube URL: `f"https://www.youtube.com/watch?v={video_id}"`
- **Never** pass just the video ID directly to `ydl.extract_info()`
- This ensures proper metadata extraction and compatibility

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

- **Do not** reintroduce `apt-get install` of ffmpeg into the Dockerfile
- On Omarchy, Docker build cannot access apt repos due to missing veth modules
- If local transcription is required, install ffmpeg on the host OS or prebuild the image on another machine
- Keep the Dockerfile minimal with only Python dependencies from pip

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

---

When in doubt, preserve existing behavior and consult this document.
