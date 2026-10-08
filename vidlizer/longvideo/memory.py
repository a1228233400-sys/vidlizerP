"""Hierarchical movie memory: characters, events, sequences and global story model."""

from __future__ import annotations

import json
from typing import Any

from .db import (
    load_scenes,
    upsert_character,
    upsert_event,
    upsert_event_relation,
    replace_story_sequences,
    upsert_memory,
)
from .provider import ProviderClient


MEMORY_PROMPT = """You are the continuity editor of a feature-length movie memory.

Maintain identity continuity across these chronological scenes.
- Reuse an existing character_key when the same person is clearly present.
- Create a new key only when necessary.
- Never invent names; use Unknown when the film has not established one.
- Extract concrete events supported by the scenes.
- Track meaningful relationship/state changes.
- Causality hints must be evidence-based.

Return ONLY JSON:
{
  "characters":[
    {"character_key":"char_001","name":"Alice","aliases":[],"description":"...","arc_notes":"..."}
  ],
  "mentions":[
    {"character_key":"char_001","scene_id":"scene_0001","role":"present|speaker|participant|mentioned"}
  ],
  "events":[
    {
      "event_key":"local_evt_001",
      "scene_id":"scene_0001",
      "timestamp_s":123.4,
      "type":"meeting|discovery|decision|conflict|reveal|travel|death|other",
      "description":"...",
      "characters":["char_001"],
      "importance":"low|medium|high"
    }
  ],
  "relations":[
    {"from_event":"local_evt_001","to_event":"local_evt_002","type":"causes|enables|reveals|changes_belief|escalates|resolves"}
  ]
}
"""


def _registry_text(registry: dict[str, dict[str, Any]], limit: int = 40) -> str:
    items = list(registry.items())[:limit]
    return "\\n".join(
        f"{key}: {value.get('name', 'Unknown')} | {value.get('description', '')} | arc={value.get('arc_notes', '')}"
        for key, value in items
    ) or "(empty)"


def _scene_text(scene: dict) -> str:
    return (
        f"[{scene['scene_id']}] {scene['start_s']:.2f}-{scene['end_s']:.2f}s "
        f"title={scene['title']} location={scene['location']} importance={scene['importance']}\\n"
        f"characters={', '.join(scene['characters'])}\\n"
        f"summary={scene['summary']}\\n"
        f"purpose={scene['dramatic_purpose'] or ''}"
    )


def build_character_event_memory(
    conn,
    movie_id: int,
    client: ProviderClient,
    batch_scenes: int = 8,
    max_output_tokens: int = 3072,
) -> tuple[dict[str, dict], list[dict]]:
    scenes = load_scenes(conn, movie_id)
    if not scenes:
        raise RuntimeError("No scenes found.")

    registry: dict[str, dict[str, Any]] = {}
    all_events: list[dict[str, Any]] = []
    next_event_number = 1

    for start in range(0, len(scenes), batch_scenes):
        batch = scenes[start:start + batch_scenes]
        recent = "\\n".join(
            f"{e['event_id']}: {e['description']}" for e in all_events[-20:]
        ) or "(none)"

        result = client.complete_json(
            MEMORY_PROMPT
            + "\\nCURRENT CHARACTER REGISTRY:\\n" + _registry_text(registry)
            + "\\nRECENT EVENTS:\\n" + recent
            + "\\nNEW SCENES:\\n" + "\\n\\n".join(_scene_text(s) for s in batch),
            max_output_tokens=max_output_tokens,
        )

        for c in result.get("characters", []):
            key = str(c.get("character_key", "")).strip()
            if not key:
                continue
            registry[key] = {
                "character_key": key,
                "name": c.get("name") or "Unknown",
                "aliases": list(c.get("aliases", [])),
                "description": c.get("description", ""),
                "arc_notes": c.get("arc_notes", ""),
            }
            upsert_character(conn, movie_id, registry[key])

        for mention in result.get("mentions", []):
            key = str(mention.get("character_key", ""))
            if key in registry:
                data = {**registry[key], "mention": mention}
                upsert_character(conn, movie_id, data)

        local_to_real: dict[str, str] = {}
        for e in result.get("events", []):
            local_key = str(e.get("event_key", "")).strip()
            if not local_key:
                continue
            event_id = f"evt_{next_event_number:06d}"
            next_event_number += 1
            local_to_real[local_key] = event_id
            event = {
                "event_id": event_id,
                "scene_id": e.get("scene_id"),
                "timestamp_s": float(e.get("timestamp_s") or 0.0),
                "type": e.get("type", "other"),
                "description": e.get("description", ""),
                "characters": list(e.get("characters", [])),
                "importance": e.get("importance", "medium"),
            }
            all_events.append(event)
            upsert_event(conn, movie_id, event)

        for rel in result.get("relations", []):
            from_key = str(rel.get("from_event", ""))
            to_key = str(rel.get("to_event", ""))
            from_id = local_to_real.get(from_key, from_key if from_key.startswith("evt_") else "")
            to_id = local_to_real.get(to_key, to_key if to_key.startswith("evt_") else "")
            if from_id and to_id:
                upsert_event_relation(
                    conn, movie_id, from_id, to_id, str(rel.get("type", "related"))
                )
        conn.commit()

    return registry, all_events


def build_hierarchical_memory(
    conn,
    movie_id: int,
    client: ProviderClient,
    registry: dict[str, dict],
    events: list[dict],
    scene_batch: int = 10,
) -> dict:
    scenes = load_scenes(conn, movie_id)
    sequences: list[dict] = []

    for start in range(0, len(scenes), scene_batch):
        chunk = scenes[start:start + scene_batch]
        prompt = (
            "Summarize this chronological group of movie scenes as one story sequence. "
            "Focus on goals, turning points, discoveries, conflicts, and character changes. "
            "Return ONLY JSON with sequence_title, summary, key_events, character_changes.\\n\\n"
            + "\\n\\n".join(_scene_text(s) for s in chunk)
        )
        summary = client.complete_json(prompt, max_output_tokens=2048)
        sequence = {
            "sequence_id": f"seq_{start // scene_batch + 1:04d}",
            "start_s": float(chunk[0]["start_s"]),
            "end_s": float(chunk[-1]["end_s"]),
            "title": summary.get("sequence_title", f"Sequence {start // scene_batch + 1}"),
            "summary": summary.get("summary", ""),
            "key_events": summary.get("key_events", []),
            "character_changes": summary.get("character_changes", []),
        }
        sequences.append(sequence)

    replace_story_sequences(conn, movie_id, sequences)
    conn.commit()

    high_events = [e for e in events if e.get("importance") == "high"][:40]
    global_prompt = (
        "You are the senior story analyst. Build global memory for a feature-length movie. "
        "Use only the supplied material. Separate established facts from interpretation. "
        "Return ONLY JSON with: logline, full_summary, acts, character_arcs, major_themes, "
        "foreshadowing_candidates, open_questions.\\n\\n"
        "SEQUENCES:\\n" + "\\n\\n".join(json.dumps(x, ensure_ascii=False) for x in sequences)
        + "\\n\\nCHARACTERS:\\n" + _registry_text(registry)
        + "\\n\\nIMPORTANT EVENTS:\\n"
        + "\\n".join(f"{e['event_id']}: {e['description']}" for e in high_events)
    )
    global_result = client.complete_json(global_prompt, max_output_tokens=4096)
    memory = {
        "sequence_count": len(sequences),
        "character_count": len(registry),
        "event_count": len(events),
        "sequences": sequences,
        **global_result,
    }
    upsert_memory(conn, movie_id, "global", memory)
    conn.commit()
    return memory
