"""Evidence retrieval, grounded QA and optional deep rewatch."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .db import get_global_memory, like_search_evidence, rebuild_search_index, search_evidence, time_evidence
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


def _terms(question: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]{2,}", question.lower())


def _fts_query(question: str) -> str:
    tokens = _terms(question)
    if not tokens:
        return "movie"
    return " AND ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:12])



def _time_targets(question: str) -> list[tuple[float, float]]:
    targets: list[tuple[float, float]] = []

    for h, minute, second in re.findall(
        r"(?:at|@|around|约|大约|在)?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?",
        question.lower(),
    ):
        if second:
            total = int(h) * 3600 + int(minute) * 60 + int(second)
        else:
            total = int(h) * 60 + int(minute)
        targets.append((max(0.0, total - 30), total + 30))

    for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|分钟|分)", question.lower()):
        seconds = float(value) * 60
        targets.append((max(0.0, seconds - 45), seconds + 45))

    return targets


QUERY_EXPANSION_PROMPT = """Expand a movie question into a small set of retrieval queries.
The movie evidence may be in English even when the user asks in Chinese. Produce up to five
short keyword phrases covering people, objects, locations, events, and likely English equivalents.
Do not answer the question.
Return ONLY JSON: {"queries":["..."]}"""


def _expand_terms(client: ProviderClient, question: str) -> list[str]:
    try:
        result = client.complete_json(
            QUERY_EXPANSION_PROMPT + "\nQUESTION:\n" + question,
            max_output_tokens=512,
        )
        queries = result.get("queries", [])
        return [str(q).strip() for q in queries if str(q).strip()][:5]
    except Exception:
        return []

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
    if not hits_raw:
        tokens = _terms(question)
        if tokens:
            or_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:16])
            hits_raw = search_evidence(conn, movie_id, or_query, limit=max_hits)
            if not hits_raw:
                hits_raw = like_search_evidence(conn, movie_id, tokens, limit=max_hits)

    if len(hits_raw) < max(3, max_hits // 3):
        for expanded in _expand_terms(client, question):
            expanded_terms = re.findall(r"[\w\u4e00-\u9fff]{2,}", expanded.lower())
            if not expanded_terms:
                continue
            expanded_query = " OR ".join(
                f'"{token.replace(chr(34), "")}"' for token in expanded_terms[:10]
            )
            extra = search_evidence(conn, movie_id, expanded_query, limit=max_hits)
            if not extra:
                extra = like_search_evidence(conn, movie_id, expanded_terms, limit=max_hits)
            seen = {row["evidence_id"] for row in hits_raw}
            hits_raw.extend(row for row in extra if row["evidence_id"] not in seen)
            if len(hits_raw) >= max_hits:
                hits_raw = hits_raw[:max_hits]
                break

    for start_s, end_s in _time_targets(question):
        timed = time_evidence(conn, movie_id, start_s, end_s, limit=max_hits)
        seen = {row["evidence_id"] for row in hits_raw}
        hits_raw.extend(row for row in timed if row["evidence_id"] not in seen)
        if len(hits_raw) >= max_hits:
            hits_raw = hits_raw[:max_hits]
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

    if (
        result.get("needs_rewatch")
        and not rewatch_results
        and video
        and workspace
        and hits
    ):
        transcript, _ = transcribe_movie(video)
        target = hits[0]
        rewatch_results.append(
            rewatch_window(
                client,
                video,
                transcript,
                max(0.0, target.start_s - 20),
                target.end_s + 20,
                workspace,
                scale=768,
                max_frames=24,
            )
        )
        refined_prompt = prompt + (
            "\n\nThe first answer requested a rewatch. A targeted rewatch is now available:\n"
            + json.dumps(rewatch_results, ensure_ascii=False)
            + "\nRe-evaluate the answer using this new evidence."
        )
        result = client.complete_json(refined_prompt, max_output_tokens=3072)

    if rewatch_results:
        result["rewatch"] = rewatch_results

    result["evidence"] = [
        {
            "evidence_id": h.evidence_id,
            "source_type": h.source_type,
            "start_s": h.start_s,
            "end_s": h.end_s,
            "timestamp": (None if h.source_type in {"global", "character"} else _timestamp(h.start_s)),
            "content": h.content,
        }
        for h in hits
    ]
    return result
