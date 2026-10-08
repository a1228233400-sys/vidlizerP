"""Evidence retrieval, grounded QA and optional deep rewatch."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .db import get_global_memory, rebuild_search_index, search_evidence
from .provider import ProviderClient
from .rewatch import rewatch_window
from .transcript import transcribe_movie


@dataclass(frozen=True)
class EvidenceHit:
    evidence_id: str
    source_type: str
    start_s: float
    end_s: float
    content: str


def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fts_query(question: str) -> str:
    tokens = re.findall(r"[\w\u4e00-\u9fff]{2,}", question.lower())
    if not tokens:
        return "movie"
    return " AND ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:12])


def build_index(conn, movie_id: int) -> int:
    return rebuild_search_index(conn, movie_id)


def ask(
    conn,
    movie_id: int,
    question: str,
    client: ProviderClient,
    video: Path | None = None,
    *,
    deep: bool = False,
    workspace: Path | None = None,
    max_hits: int = 12,
) -> dict:
    hits_raw = search_evidence(conn, movie_id, _fts_query(question), limit=max_hits)
    hits = [
        EvidenceHit(
            evidence_id=row["evidence_id"],
            source_type=row["source_type"],
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            content=row["content"],
        )
        for row in hits_raw
    ]

    rewatch_results: list[dict] = []
    if deep and video and workspace and hits:
        transcript, _ = transcribe_movie(video)
        windows: list[tuple[float, float]] = []
        for hit in hits[:3]:
            start = max(0.0, hit.start_s - 12)
            end = hit.end_s + 12
            if not any(abs(start - a) < 8 and abs(end - b) < 8 for a, b in windows):
                windows.append((start, end))
        for start, end in windows[:2]:
            rewatch_results.append(
                rewatch_window(
                    client, video, transcript, start, end, workspace
                )
            )

    memory = get_global_memory(conn, movie_id) or {}
    evidence_text = "\n\n".join(
        f"[EVIDENCE {h.evidence_id}] [{_timestamp(h.start_s)}-{_timestamp(h.end_s)}] "
        f"{h.source_type}\n{h.content}"
        for h in hits
    )
    rewatch_text = "\n\n".join(
        f"[REWATCH {i + 1}] {json.dumps(item, ensure_ascii=False)}"
        for i, item in enumerate(rewatch_results)
    )

    prompt = (
        "You are the grounded question-answering layer for a movie that has already been analyzed. "
        "Answer only from the supplied movie memory, evidence, and targeted rewatch. "
        "Do not invent. If evidence is insufficient, explicitly say so. "
        "Return ONLY JSON: "
        '{"answer":"...","confidence":0.0,"evidence_ids":["ev_..."],"needs_rewatch":false}'
        "\n\nQUESTION:\n" + question
        + "\n\nGLOBAL MEMORY:\n" + json.dumps(memory, ensure_ascii=False)
        + "\n\nEVIDENCE:\n" + (evidence_text or "(none)")
        + "\n\nTARGETED REWATCH:\n" + (rewatch_text or "(none)")
    )
    result = client.complete_json(prompt, max_output_tokens=3072)
    if rewatch_results:
        result["rewatch"] = rewatch_results
    return result
