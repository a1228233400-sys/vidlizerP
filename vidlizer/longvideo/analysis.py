"""Shot-level multimodal analysis with bounded per-shot work."""

from __future__ import annotations

import shutil
from pathlib import Path

from .extract import extract_uniform_window
from .models import Shot
from .provider import ProviderClient
from .sampling import sample_count
from .transcript import overlapping_segments


SHOT_PROMPT = """You are the shot-level observer in a long-form movie understanding system.

You are given multiple observations from ONE continuous camera shot, ordered in time.
Do not describe the images independently. Reconstruct the change that happens across the shot.

Use ONLY evidence in the images and supplied transcript. Do not invent identities, motives, or off-screen events.

Keep the response compact. Lists must contain at most 4 items. Each description/string should be short (preferably under 120 characters). Do not quote long dialogue. Summary should stay under 240 characters.

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
    deep_pass: bool = False,
) -> dict:
    count = min(max_samples, sample_count(shot.duration_s, maximum=max_samples))
    shot_dir = workspace / shot.shot_id
    if shot_dir.exists():
        shutil.rmtree(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    try:
        images, timestamps = extract_uniform_window(
            video,
            shot.start_s,
            shot.end_s,
            count,
            shot_dir,
            scale=scale,
            prefix="shot",
        )
        segments = overlapping_segments(transcript, shot.start_s, shot.end_s)
        transcript_text = "\n".join(
            f"[{float(s['start']):.2f}-{float(s['end']):.2f}] {s['text']}"
            for s in segments
        )
        depth_note = (
            "This is a SECOND, HIGH-DENSITY PASS. Re-check subtle visual changes, identity continuity, "
            "objects, and action ordering. Resolve uncertainty where the added observations support it.\n"
            if deep_pass
            else ""
        )
        prompt = (
            f"{SHOT_PROMPT}\n"
            f"{depth_note}"
            f"Shot: {shot.shot_id}\n"
            f"Absolute time: {shot.start_s:.3f}-{shot.end_s:.3f}s\n"
            f"Transcript:\n{transcript_text or '(none)'}\n"
        )
        try:
            result = client.analyze(
                prompt,
                images,
                timestamps,
                max_output_tokens=max_output_tokens,
            )
        except Exception as first_exc:
            message = str(first_exc).lower()
            if not any(
                marker in message
                for marker in (
                    "invalid json",
                    "unterminated string",
                    "expecting property name",
                    "expecting value",
                    "expecting ',' delimiter",
                )
            ):
                raise
            compact_prompt = (
                f"{SHOT_PROMPT}
"
                "COMPACT RETRY: Return a small JSON object only. "
                "Use at most 3 items in any array. Keep each string under 120 characters. "
                "Keep summary under 240 characters and dialogue_context under 180 characters. "
                "No markdown, no commentary, no long quotations.
"
                f"{depth_note}"
                f"Shot: {shot.shot_id}
"
                f"Absolute time: {shot.start_s:.3f}-{shot.end_s:.3f}s
"
                f"Transcript:
{transcript_text or '(none)'}
"
            )
            # Reuse the extracted frames; do not launch another FFmpeg process.
            result = client.analyze(
                compact_prompt,
                images,
                timestamps,
                max_output_tokens=min(max_output_tokens, 1024),
            )
        result.update(
            {
                "shot_id": shot.shot_id,
                "start_s": shot.start_s,
                "end_s": shot.end_s,
                "duration_s": shot.duration_s,
                "observed_at": [round(t, 3) for t in timestamps],
                "analysis_pass": 2 if deep_pass else 1,
            }
        )
        return result
    finally:
        shutil.rmtree(shot_dir, ignore_errors=True)


def compact_observation(observation: dict) -> str:
    return "\n".join(
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
