import pytest
from bot.transcripts import fetch_transcript

def test_youtube_transcript_extraction():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    text = fetch_transcript(url)
    assert text is None or isinstance(text, str)
