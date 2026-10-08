"""Resumable end-to-end MovieMind pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console

from .analysis import analyze_shot
from .db import (
    connect,
    get_global_memory,
    clear_memory_layers,
    insert_samples,
    insert_transcript,
    load_transcript,
    load_movie_info,
    mark_shot_status,
    movie_id_for_path,
    pending_shots,
    record_run,
    stage_complete,
    store_shot_observation,
    rebuild_search_index,
    replace_shots,
    upsert_movie,
)
from .media import probe_movie
from .models import Shot
from .resources import PROFILES, check_safe_to_start, format_preflight, recommend_profile, snapshot
from .sampling import adaptive_sample_points
from .scenes import build_scenes
from .memory import build_character_event_memory, build_hierarchical_memory
from .provider import ProviderClient
from .shots import detect_shots
from .transcript import transcribe_movie


_console = Console()


def _movie_id(conn, video: Path) -> int:
    movie_id = movie_id_for_path(conn, video)
    if movie_id is None:
        info = probe_movie(video)
        movie_id = upsert_movie(conn, info)
        conn.commit()
    return movie_id


def ingest_index(
    video: Path,
    db_path: Path,
    *,
    scene_threshold: float = 3.0,
    min_scene_len: int = 12,
    max_samples_per_shot: int = 12,
) -> dict:
    info = probe_movie(video)
    conn = connect(db_path)
    try:
        movie_id = upsert_movie(conn, info)
        if stage_complete(conn, movie_id, "shot_detection"):
            shots_count = conn.execute(
                "SELECT COUNT(*) FROM shots WHERE movie_id=?", (movie_id,)
            ).fetchone()[0]
            sample_count = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE movie_id=?", (movie_id,)
            ).fetchone()[0]
            return {
                "movie_id": movie_id,
                "duration_s": info.duration_s,
                "shot_count": int(shots_count),
                "sample_count": int(sample_count),
                "db_path": str(db_path),
            }

        shots = detect_shots(video, threshold=scene_threshold, min_scene_len=min_scene_len)
        replace_shots(conn, movie_id, shots)
        all_samples = []
        for shot in shots:
            all_samples.extend(
                adaptive_sample_points(shot, maximum=max_samples_per_shot)
            )
        insert_samples(conn, movie_id, all_samples)
        record_run(conn, movie_id, "shot_detection", "complete", {"shot_count": len(shots)})
        record_run(
            conn,
            movie_id,
            "sampling_plan",
            "complete",
            {"sample_count": len(all_samples), "max_samples_per_shot": max_samples_per_shot},
        )
        conn.commit()
        return {
            "movie_id": movie_id,
            "duration_s": info.duration_s,
            "shot_count": len(shots),
            "sample_count": len(all_samples),
            "db_path": str(db_path),
        }
    finally:
        conn.close()


def analyze_all_shots(
    video: Path,
    db_path: Path,
    client: ProviderClient,
    *,
    scale: int,
    max_samples_per_shot: int,
    max_output_tokens: int,
) -> dict:
    conn = connect(db_path)
    movie_id = _movie_id(conn, video)
    try:
        if stage_complete(conn, movie_id, "transcript"):
            transcript = load_transcript(conn, movie_id)
            source = "stored"
        else:
            transcript, source = transcribe_movie(video)
            insert_transcript(conn, movie_id, transcript, source)
            record_run(
                conn,
                movie_id,
                "transcript",
                "complete" if transcript else "unavailable",
                {"segments": len(transcript), "source": source},
            )
            conn.commit()

        shots = pending_shots(conn, movie_id)
        if not shots:
            rebuild_search_index(conn, movie_id)
            return {"movie_id": movie_id, "pending": 0, "completed": True}

        with tempfile.TemporaryDirectory(prefix="moviemind_work_") as temp:
            workspace = Path(temp)
            for index, row in enumerate(shots, start=1):
                current = snapshot(video)
                if current.memory_available_gb is not None and current.memory_available_gb < 1.5:
                    raise RuntimeError(
                        f"Available RAM fell to {current.memory_available_gb:.1f} GB; "
                        "pausing to protect the computer. Re-run to resume."
                    )
                _console.print(
                    f"[cyan]→[/cyan] shot {index}/{len(shots)} "
                    f"[dim]{row['shot_id']} {row['start_s']:.1f}-{row['end_s']:.1f}s[/dim]"
                )
                shot = Shot(
                    shot_id=row["shot_id"],
                    start_s=row["start_s"],
                    end_s=row["end_s"],
                    duration_s=row["duration_s"],
                )
                mark_shot_status(conn, movie_id, shot.shot_id, "running")
                conn.commit()
                try:
                    observation = analyze_shot(
                        client,
                        video,
                        shot,
                        transcript,
                        workspace,
                        scale=scale,
                        max_samples=max_samples_per_shot,
                        max_output_tokens=max_output_tokens,
                    )
                    store_shot_observation(conn, movie_id, observation)
                    mark_shot_status(conn, movie_id, shot.shot_id, "complete")
                    conn.commit()
                except KeyboardInterrupt:
                    mark_shot_status(conn, movie_id, shot.shot_id, "pending")
                    conn.commit()
                    raise
                except Exception as exc:
                    mark_shot_status(conn, movie_id, shot.shot_id, "failed")
                    record_run(
                        conn, movie_id, "shot_analysis", "failed",
                        {"shot_id": shot.shot_id, "error": str(exc)},
                    )
                    conn.commit()
                    _console.print(
                        f"[yellow]⚠[/yellow] {shot.shot_id} failed: {exc}"
                    )

        rebuild_search_index(conn, movie_id)
        record_run(
            conn,
            movie_id,
            "shot_analysis",
            "complete",
            {
                "completed": conn.execute(
                    "SELECT COUNT(*) FROM shots WHERE movie_id=? AND analysis_status='complete'",
                    (movie_id,),
                ).fetchone()[0],
                "failed": conn.execute(
                    "SELECT COUNT(*) FROM shots WHERE movie_id=? AND analysis_status='failed'",
                    (movie_id,),
                ).fetchone()[0],
            },
        )
        conn.commit()
        return {
            "movie_id": movie_id,
            "pending": len(pending_shots(conn, movie_id)),
            "completed": True,
        }
    finally:
        conn.close()


def build_memory(
    db_path: Path,
    client: ProviderClient,
) -> dict:
    conn = connect(db_path)
    try:
        movie_id = conn.execute("SELECT id FROM movies ORDER BY id DESC LIMIT 1").fetchone()
        if not movie_id:
            raise RuntimeError("No movie is indexed in this database.")
        mid = int(movie_id[0])
        if stage_complete(conn, mid, "global_memory"):
            return get_global_memory(conn, mid) or {}

        if not stage_complete(conn, mid, "scene_memory"):
            scenes = build_scenes(conn, mid, client)
            record_run(conn, mid, "scene_memory", "complete", {"scene_count": len(scenes)})
            conn.commit()

        clear_memory_layers(conn, mid)
        registry, events = build_character_event_memory(conn, mid, client)
        record_run(
            conn,
            mid,
            "character_event_memory",
            "complete",
            {"character_count": len(registry), "event_count": len(events)},
        )
        conn.commit()

        memory = build_hierarchical_memory(conn, mid, client, registry, events)
        rebuild_search_index(conn, mid)
        record_run(
            conn,
            mid,
            "global_memory",
            "complete",
            {
                "character_count": len(registry),
                "event_count": len(events),
                "sequence_count": memory.get("sequence_count", 0),
            },
        )
        conn.commit()
        return memory
    finally:
        conn.close()


def process_movie(
    video: Path,
    db_path: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    profile: str = "balanced",
    max_cost: float = 0.0,
    timeout: int = 600,
    scene_threshold: float = 3.0,
    min_scene_len: int = 12,
) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    selected = PROFILES[profile]
    snap, warnings = check_safe_to_start(video, db_path)
    if snap.memory_available_gb is not None and snap.memory_available_gb < 4.0 and profile != "safe":
        profile = "safe"
        selected = PROFILES["safe"]
        warnings.append("Low available RAM forced the safe profile.")
    _console.print(format_preflight(snap, warnings, selected))

    client = ProviderClient.from_env(
        provider=provider,
        model=model,
        timeout=timeout,
        max_cost=max_cost,
        context_tokens=selected.context_tokens,
    )
    client.preflight()

    index = ingest_index(
        video,
        db_path,
        scene_threshold=scene_threshold,
        min_scene_len=min_scene_len,
        max_samples_per_shot=selected.max_samples_per_shot,
    )
    analysis = analyze_all_shots(
        video,
        db_path,
        client,
        scale=selected.frame_scale,
        max_samples_per_shot=selected.max_samples_per_shot,
        max_output_tokens=selected.max_output_tokens,
    )

    conn = connect(db_path)
    try:
        mid = int(index["movie_id"])
        if analysis["pending"] > 0:
            raise RuntimeError(
                "Some shots failed. Re-run the same command to resume failed/pending work."
            )
    finally:
        conn.close()

    memory = build_memory(db_path, client)
    return {
        "movie_id": index["movie_id"],
        "duration_s": index["duration_s"],
        "shots": index["shot_count"],
        "samples": index["sample_count"],
        "memory": memory,
        "provider": client.stats(),
        "db": str(db_path),
    }


def status(db_path: Path) -> dict:
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT id FROM movies ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return {"ready": False}
        mid = int(row[0])
        counts = {
            "shots": conn.execute("SELECT COUNT(*) FROM shots WHERE movie_id=?", (mid,)).fetchone()[0],
            "shots_complete": conn.execute(
                "SELECT COUNT(*) FROM shots WHERE movie_id=? AND analysis_status='complete'", (mid,)
            ).fetchone()[0],
            "scenes": conn.execute("SELECT COUNT(*) FROM scenes WHERE movie_id=?", (mid,)).fetchone()[0],
            "characters": conn.execute("SELECT COUNT(*) FROM characters WHERE movie_id=?", (mid,)).fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM events WHERE movie_id=?", (mid,)).fetchone()[0],
            "transcript_segments": conn.execute("SELECT COUNT(*) FROM transcripts WHERE movie_id=?", (mid,)).fetchone()[0],
            "evidence": conn.execute("SELECT COUNT(*) FROM evidence WHERE movie_id=?", (mid,)).fetchone()[0],
        }
        return {
            "ready": counts["shots"] > 0 and counts["shots"] == counts["shots_complete"] and counts["scenes"] > 0,
            "movie": load_movie_info(conn, mid),
            **counts,
            "global_memory": bool(get_global_memory(conn, mid)),
        }
    finally:
        conn.close()
