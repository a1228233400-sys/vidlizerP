"""Narrative scene construction from ordered shot observations."""

from __future__ import annotations

from .db import load_shot_observations, replace_scenes
from .models import Scene
from .provider import ProviderClient


BOUNDARY_PROMPT = """You are a film editor helping build a long-video memory.

Below are sequential SHOT observations. Decide after which shot IDs a NEW NARRATIVE SCENE begins.
A camera cut does not necessarily mean a new scene. A new scene usually involves a meaningful
change in location, time, or story context. Keep the order unchanged.

Return ONLY JSON:
{"boundary_after_shot_ids":["shot_..."]}

Be conservative: false scene breaks are worse than keeping shots together.
"""


SUMMARY_PROMPT = """You are the scene synthesizer for a feature-length movie.

Given ordered observations for one narrative scene, summarize what actually happens.
Do not invent motives or facts not supported by the evidence.

Return ONLY JSON:
{
  "title":"...",
  "location":"...",
  "characters":["..."],
  "summary":"...",
  "dramatic_purpose":"...",
  "importance":"low|medium|high"
}
"""


def _items(rows: list[dict]) -> str:
    return "\\n\\n".join(
        f"[{r['shot_id']}] {r['start_s']:.2f}-{r['end_s']:.2f}s\\n{r.get('observation_text', r.get('summary', ''))}"
        for r in rows
    )


def build_scenes(
    conn,
    movie_id: int,
    client: ProviderClient,
    window_size: int = 36,
    overlap: int = 4,
    max_output_tokens: int = 2048,
) -> list[Scene]:
    rows = load_shot_observations(conn, movie_id)
    if not rows:
        raise RuntimeError("No shot observations found. Run shot analysis first.")

    boundaries: set[str] = set()
    step = max(1, window_size - overlap)
    for start in range(0, len(rows), step):
        window = rows[start:start + window_size]
        if len(window) <= 1:
            break
        result = client.complete_json(
            BOUNDARY_PROMPT + "\\n\\n" + _items(window),
            max_output_tokens=1024,
        )
        valid = {row["shot_id"] for row in window[:-1]}
        boundaries.update(
            str(shot_id) for shot_id in result.get("boundary_after_shot_ids", [])
            if str(shot_id) in valid
        )

    groups: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        current.append(row)
        if row["shot_id"] in boundaries:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    scenes: list[Scene] = []
    for idx, group in enumerate(groups, start=1):
        summary = client.complete_json(
            SUMMARY_PROMPT + "\\n\\n" + _items(group),
            max_output_tokens=max_output_tokens,
        )
        scenes.append(
            Scene(
                scene_id=f"scene_{idx:04d}",
                start_s=group[0]["start_s"],
                end_s=group[-1]["end_s"],
                shot_ids=[r["shot_id"] for r in group],
                title=summary.get("title", f"Scene {idx}"),
                location=summary.get("location", "unknown"),
                characters=[str(x) for x in summary.get("characters", [])],
                summary=summary.get("summary", ""),
                dramatic_purpose=summary.get("dramatic_purpose", ""),
                importance=summary.get("importance", "medium"),
                details=summary,
            )
        )

    replace_scenes(conn, movie_id, scenes)
    conn.commit()
    return scenes
