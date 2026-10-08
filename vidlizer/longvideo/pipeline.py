"""V0.1 long-video ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .db import connect, insert_samples, record_run, replace_shots, upsert_movie
from .media import probe_movie
from .sampling import adaptive_sample_points
from .shots import detect_shots

_console = Console()


def ingest_movie(
    video: Path,
    db_path: Path,
    *,
    scene_threshold: float = 3.0,
    min_scene_len: int = 12,
    max_samples_per_shot: int = 12,
) -> dict:
    """Create the long-video index without invoking a VLM yet.

    V0.1 purpose: replace the old whole-movie frame cap with a persistent,
    shot-aware timeline that later analysis stages can consume incrementally.
    """
    info = probe_movie(video)
    conn = connect(db_path)
    try:
        movie_id = upsert_movie(conn, info)
        record_run(
            conn,
            movie_id,
            "probe",
            "complete",
            info.to_dict(),
        )
        conn.commit()

        shots = detect_shots(
            video,
            threshold=scene_threshold,
            min_scene_len=min_scene_len,
        )
        replace_shots(conn, movie_id, shots)
        conn.commit()
        record_run(
            conn,
            movie_id,
            "shot_detection",
            "complete",
            {"shot_count": len(shots)},
        )

        all_samples = []
        for shot in shots:
            all_samples.extend(
                adaptive_sample_points(
                    shot,
                    maximum=max_samples_per_shot,
                )
            )
        insert_samples(conn, movie_id, all_samples)
        record_run(
            conn,
            movie_id,
            "sampling_plan",
            "complete",
            {
                "sample_count": len(all_samples),
                "max_samples_per_shot": max_samples_per_shot,
            },
        )
        conn.commit()

        return {
            "movie_id": movie_id,
            "duration_s": info.duration_s,
            "shot_count": len(shots),
            "sample_count": len(all_samples),
            "db_path": str(db_path),
        }
    except Exception as exc:
        if "movie_id" in locals():
            record_run(
                conn,
                movie_id,
                "ingest",
                "failed",
                {"error": str(exc)},
            )
            conn.commit()
        raise
    finally:
        conn.close()


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_ingest_result(result: dict) -> None:
    _console.print()
    _console.print("[bold]MovieMind / Vidlizer Long V0.1[/bold]")
    _console.print(f"Duration:       [cyan]{format_duration(result['duration_s'])}[/cyan]")
    _console.print(f"Shots detected: [cyan]{result['shot_count']}[/cyan]")
    _console.print(f"Sample points:  [cyan]{result['sample_count']}[/cyan]")
    _console.print(f"Movie DB:       [cyan]{result['db_path']}[/cyan]")
    _console.print()
    _console.print(
        "[dim]Next stage will attach transcript + shot-level VLM analysis to this index.[/dim]"
    )
