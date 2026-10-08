from vidlizer.longvideo.transcript import parse_srt, parse_vtt


def test_parse_srt():
    text = """1
00:00:01,000 --> 00:00:03,500
Hello there!

2
00:00:04,000 --> 00:00:05,000
Second line.
"""
    result = parse_srt(text)
    assert result == [
        {"start": 1.0, "end": 3.5, "text": "Hello there!"},
        {"start": 4.0, "end": 5.0, "text": "Second line."},
    ]


def test_parse_vtt():
    text = """WEBVTT

00:00:01.000 --> 00:00:02.500
Hello
"""
    result = parse_vtt(text)
    assert result == [{"start": 1.0, "end": 2.5, "text": "Hello"}]
