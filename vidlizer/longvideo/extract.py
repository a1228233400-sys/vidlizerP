"""Bounded frame extraction for shot analysis and targeted rewatch."""

from __future__ import annotations

import subprocess
from pathlib import Path


def extract_frame(video: Path, timestamp_s: float, out: Path, scale: int = 640) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, timestamp_s):.3f}", "-i", str(video),
        "-frames:v", "1",
        "-vf", f"scale={scale}:-2:flags=lanczos,format=yuvj420p",
        "-q:v", "3", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"frame extraction failed at {timestamp_s:.3f}s: {(exc.stderr or '').strip()}"
        ) from exc
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no frame at {timestamp_s:.3f}s")
    return out


def extract_points(
    video: Path,
    points: list[tuple[str, float]],
    out_dir: Path,
    scale: int = 640,
) -> tuple[list[Path], list[float]]:
    paths: list[Path] = []
    timestamps: list[float] = []
    for sample_id, timestamp in points:
        path = out_dir / f"{sample_id}.jpg"
        extract_frame(video, timestamp, path, scale)
        paths.append(path)
        timestamps.append(timestamp)
    return paths, timestamps
