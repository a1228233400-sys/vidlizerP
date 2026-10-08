"""Shot-level multimodal analysis with bounded per-shot work."""

from __future__ import annotations

import shutil
from pathlib import Path

from .extract import extract_points
from .models import Shot
from .provider import ProviderClient
from .sampling import adaptive_sample_points
from .transcript import overlapping_segments


SHOT_PROMPT = """You are the shot-level observer in a long-form movie understanding system.

You are given multiple observations from ONE continuous camera shot, ordered in time.
Do not describe the images independently. Reconstruct the change that happens across the shot.

Use ONLY evidence in the images and supplied transcript. Do not invent identities, motives, or off-screen events.

Return ONLY valid JSON:
{
  "location": "...",
  "characters": [{"label": "char_01", "description": "...", "appearance_notes": "..."}],
  "objects": ["..."],
  "actions": [{"start_offset_s": 0.0, "end_offset_s": 1.2, "description": "..."}],
  "visual_changes": ["..."],
  "text_visible": ["..."],
  "dialogue_context": "...",
  "emotion_observations": ["..."],
  "importance": "low|medium|high",
  "summary": "...",
  "confidence": 0.0
}

Labels like char_01 are temporary within this shot. Later stages resolve identities.
Offsets are relative to shot start. When something cannot be established, say so.
"""


def analyze_shot(
    client: ProviderClient,
    video: Path,
    shot: Shot,
    transcript: list[dict],
    workspace: Path,
    *,
    scale: int = 640,
    max_samples: int = 12,
    max_output_tokens: int = 2048,
) -> dict:
    samples = adaptive_sample_points(shot, maximum=max_samples)
    points = [(s.sample_id, s.timestamp_s) for s in samples]
    shot_dir = workspace / shot.shot_id
    if shot_dir.exists():
        shutil.rmtree(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    try:
        images, timestamps = extract_points(video, points, shot_dir, scale)
        segments = overlapping_segments(transcript, shot.start_s, shot.end_s)
        transcript_text = "\\n".join(
            f"[{float(s['start']):.2f}-{float(s['end']):.2f}] {s['text']}"
            for s in segments
        )
        prompt = (
            f"{SHOT_PROMPT}\\n"
            f"Shot: {shot.shot_id}\\n"
            f"Absolute time: {shot.start_s:.3f}-{shot.end_s:.3f}s\\n"
            f"Transcript:\\n{transcript_text or '(none)'}\\n"
        )
        result = client.analyze(prompt, images, timestamps, max_output_tokens=max_output_tokens)
        result.update(
            {
                "shot_id": shot.shot_id,
                "start_s": shot.start_s,
                "end_s": shot.end_s,
                "duration_s": shot.duration_s,
                "observed_at": [round(t, 3) for t in timestamps],
            }
        )
        return result
    finally:
        shutil.rmtree(shot_dir, ignore_errors=True)


def compact_observation(observation: dict) -> str:
    return "\\n".join(
        [
            f"Location: {observation.get('location', 'unknown')}",
            "Characters: " + ", ".join(
                x.get("label", "?") if isinstance(x, dict) else str(x)
                for x in observation.get("characters", [])
            ),
            "Actions: " + "; ".join(
                x.get("description", "") if isinstance(x, dict) else str(x)
                for x in observation.get("actions", [])
            ),
            "Objects: " + ", ".join(map(str, observation.get("objects", []))),
            "Text: " + ", ".join(map(str, observation.get("text_visible", []))),
            "Changes: " + "; ".join(map(str, observation.get("visual_changes", []))),
            f"Dialogue: {observation.get('dialogue_context', '')}",
            f"Summary: {observation.get('summary', '')}",
            f"Importance: {observation.get('importance', 'medium')}",
        ]
    )
