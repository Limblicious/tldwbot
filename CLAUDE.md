# CLAUDE.md - Development Guardrails

This file contains strict guidelines for future edits to the tldwbot codebase.

## API Usage

- **Always** use `client.chat.completions.create()` for OpenAI summarization calls
- **Never** use `client.responses.create()` or other deprecated/incorrect API methods
- Maintain `temperature=0.2` for consistent summarization results

## YouTube Video Processing

- **Always** call yt-dlp with a full YouTube URL: `f"https://www.youtube.com/watch?v={video_id}"`
- **Never** pass just the video ID directly to `ydl.extract_info()`
- This ensures proper metadata extraction and compatibility

## Discord Interaction Patterns

- Ephemeral behavior must be **consistent** across all interactions
- If `defer()` is ephemeral, all followup messages must also be ephemeral
- If `defer()` is public, all followup messages must also be public
- **Do not mix** ephemeral and public messages in the same command flow

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
