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
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no frame at {timestamp_s:.3f}s")
    return out


def extract_points(
    video: Path,
    points: list[tuple[str, float]],
    out_dir: Path,
    scale: int = 640,
) -> tuple[list[Path], list[float]]:
    """Compatibility helper. Prefer extract_uniform_window for long-video work."""
    paths: list[Path] = []
    timestamps: list[float] = []
    for sample_id, timestamp in points:
        path = out_dir / f"{sample_id}.jpg"
        extract_frame(video, timestamp, path, scale)
        paths.append(path)
        timestamps.append(timestamp)
    return paths, timestamps


def extract_uniform_window(
    video: Path,
    start_s: float,
    end_s: float,
    count: int,
    out_dir: Path,
    *,
    scale: int = 640,
    prefix: str = "frame",
) -> tuple[list[Path], list[float]]:
    """Extract all observations in one FFmpeg process."""
    if count < 1:
        raise ValueError("count must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = max(0.2, end_s - start_s)
    inset = min(0.15, duration / 10)
    first = start_s + inset
    last = max(first, end_s - inset)
    span = max(0.1, last - first)
    fps = 1.0 if count == 1 else max(0.001, (count - 1) / span)

    pattern = out_dir / f"{prefix}_%03d.jpg"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, first):.3f}", "-i", str(video),
        "-t", f"{span:.3f}",
        "-vf", f"fps={fps:.8f},scale={scale}:-2:flags=lanczos,format=yuvj420p",
        "-frames:v", str(count),
        "-q:v", "3",
        str(pattern),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg window extraction failed: {(exc.stderr or '').strip()}") from exc

    paths = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    if not paths:
        raise RuntimeError(f"ffmpeg produced no frames for {start_s:.3f}-{end_s:.3f}s")
    actual = len(paths)
    timestamps = (
        [first]
        if actual == 1
        else [first + (span * i / (actual - 1)) for i in range(actual)]
    )
    return paths, timestamps
