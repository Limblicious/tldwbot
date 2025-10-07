import os
import sys

from bot.transcripts import fetch_captions, TranscriptError, BROWSER_HEADERS


def main():
    if len(sys.argv) < 2:
        print("usage: python -m bot.diag <youtube_url_or_id>")
        return
    arg = sys.argv[1]
    vid = arg.split("v=")[-1] if "youtube.com" in arg or "youtu.be" in arg else arg
    print("Video ID:", vid)
    print("Headers UA:", BROWSER_HEADERS.get("User-Agent")[:60], "...")
    print("Cookies:", os.getenv("YT_COOKIES") or "(none)")
    try:
        res = fetch_captions(vid)
        if not res:
            print("NO CAPTIONS")
        else:
            print("SOURCE:", res.source, "LEN:", len(res.text))
    except TranscriptError as e:
        print("TranscriptError:", e)


if __name__ == "__main__":
    main()
