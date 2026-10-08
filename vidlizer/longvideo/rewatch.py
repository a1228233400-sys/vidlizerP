"""Targeted high-density rewatch for uncertain movie questions."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from .extract import extract_points
from .provider import ProviderClient
from .transcript import overlapping_segments


REWATCH_PROMPT = """You are performing a targeted rewatch of an important movie segment.

Carefully compare the ordered visual observations and transcript.
State what is directly visible/audible and what remains uncertain.
Return ONLY JSON:
{
  "finding":"...",
  "events":["..."],
  "characters":["..."],
  "objects":["..."],
  "visual_evidence":["..."],
  "uncertainties":["..."],
  "confidence":0.0
}
"""


def rewatch_window(
    client: ProviderClient,
    video: Path,
    transcript: list[dict],
    start_s: float,
    end_s: float,
    workspace: Path,
    scale: int = 768,
    max_frames: int = 24,
) -> dict:
    start_s = max(0.0, start_s)
    end_s = max(start_s + 0.5, end_s)
    duration = end_s - start_s
    n = min(max_frames, max(6, int(math.ceil(duration * 1.5))))
    inset = min(0.25, duration / 10)
    first = start_s + inset
    last = max(first, end_s - inset)
    step = (last - first) / max(1, n - 1)
    points = [(f"rw_{i:03d}", first + i * step) for i in range(n)]

    out_dir = workspace / f"rewatch_{int(start_s)}_{int(end_s)}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        images, timestamps = extract_points(video, points, out_dir, scale)
        segments = overlapping_segments(transcript, start_s, end_s)
        transcript_text = "\\n".join(
            f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments
        )
        prompt = (
            REWATCH_PROMPT
            + f"\\nWindow: {start_s:.2f}-{end_s:.2f}s\\n"
            + f"Transcript:\\n{transcript_text or '(none)'}"
        )
        result = client.analyze(
            prompt,
            images,
            timestamps,
            max_output_tokens=3072,
        )
        result["start_s"] = start_s
        result["end_s"] = end_s
        result["timestamps"] = [round(t, 3) for t in timestamps]
        return result
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
