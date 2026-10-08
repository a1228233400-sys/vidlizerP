"""Media probing and local media helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .models import MovieInfo


class MediaError(RuntimeError):
    """Raised when media metadata cannot be obtained."""


def _run_json(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise MediaError("ffprobe/ffmpeg not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise MediaError(f"ffprobe failed: {detail or 'unknown error'}") from exc

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid JSON") from exc


def probe_movie(path: Path) -> MovieInfo:
    """Read stable metadata needed by the long-video pipeline."""
    if not path.exists():
        raise MediaError(f"input file does not exist: {path}")

    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        raise MediaError("input does not contain a video stream")

    duration_raw = (data.get("format") or {}).get("duration")
    try:
        duration_s = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise MediaError("unable to determine video duration") from exc

    fps = None
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate")
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        try:
            if float(den) != 0:
                fps = float(num) / float(den)
        except ValueError:
            pass

    return MovieInfo(
        path=str(path.resolve()),
        duration_s=duration_s,
        width=int(video["width"]) if video.get("width") else None,
        height=int(video["height"]) if video.get("height") else None,
        fps=fps,
        video_codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name") if audio else None,
        has_audio=audio is not None,
    )


def extract_audio(video: Path, output_wav: Path, sample_rate: int = 16_000) -> Path:
    """Create a mono PCM WAV for transcription."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(output_wav),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MediaError("ffmpeg not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"ffmpeg audio extraction failed: {(exc.stderr or '').strip()}") from exc
    return output_wav
