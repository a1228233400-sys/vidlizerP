"""Transcript extraction for long-form movie analysis."""

from __future__ import annotations

import re
from pathlib import Path


_TIME_RE = re.compile(r"(\d{2}):([0-5]\d):([0-5]\d)[,.](\d{3})")


def _timecode(value: str) -> float:
    m = _TIME_RE.match(value.strip())
    if m:
        h, minute, sec, ms = m.groups()
        return int(h) * 3600 + int(minute) * 60 + int(sec) + int(ms) / 1000
    parts = value.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(value)


def parse_srt(text: str) -> list[dict]:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    out: list[dict] = []
    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.split("\n") if line.strip()]
        timing_idx = next((i for i, line in enumerate(lines[:3]) if "-->" in line), None)
        if timing_idx is None:
            continue
        left, right = [x.strip() for x in lines[timing_idx].split("-->", 1)]
        try:
            start, end = _timecode(left), _timecode(right.split()[0])
        except ValueError:
            continue
        value = re.sub(r"<[^>]+>", "", " ".join(lines[timing_idx + 1:])).strip()
        if value:
            out.append({"start": round(start, 3), "end": round(end, 3), "text": value})
    return out


def parse_vtt(text: str) -> list[dict]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", cleaned)
    out: list[dict] = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if idx is None:
            continue
        left, right = [x.strip() for x in lines[idx].split("-->", 1)]
        try:
            start = _timecode(left.replace(".", ","))
            end = _timecode(right.split()[0].replace(".", ","))
        except ValueError:
            continue
        value = re.sub(r"<[^>]+>", "", " ".join(lines[idx + 1:])).strip()
        if value:
            out.append({"start": round(start, 3), "end": round(end, 3), "text": value})
    return out


def sidecar_transcript(video: Path) -> list[dict]:
    for suffix, parser in ((".srt", parse_srt), (".vtt", parse_vtt)):
        path = video.with_suffix(suffix)
        if path.exists():
            try:
                return parser(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                pass
    return []


def transcribe_movie(video: Path) -> tuple[list[dict], str]:
    try:
        from vidlizer.transcribe import is_available, transcribe
        if is_available():
            result = transcribe(video) or []
            if result:
                return result, "mlx-whisper"
    except Exception:
        pass
    sidecar = sidecar_transcript(video)
    if sidecar:
        return sidecar, "sidecar"
    return [], "unavailable"


def overlapping_segments(segments: list[dict], start_s: float, end_s: float) -> list[dict]:
    return [
        s for s in segments
        if float(s.get("end", 0)) > start_s and float(s.get("start", 0)) < end_s
    ]
