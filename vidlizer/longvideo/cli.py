"""User-facing CLI for MovieMind."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from .pipeline import (
    analyze_all_shots,
    build_memory,
    ingest_index,
    process_movie,
    status,
)
from .provider import ProviderClient
from .qa import ask
from .resources import PROFILES, check_safe_to_start, format_preflight, recommend_profile


_console = Console()


def _db_for(video: Path) -> Path:
    return video.with_suffix(video.suffix + ".movie.db")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movieai",
        description="MovieMind: long-form video understanding built on Vidlizer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    process = sub.add_parser("process", help="Fully index and understand a movie.")
    process.add_argument("video", type=Path)
    process.add_argument("--db", type=Path, default=None)
    process.add_argument("--provider", choices=["ollama", "openrouter", "openai"], default=None)
    process.add_argument("--model", default=None)
    process.add_argument("--profile", choices=["auto", "safe", "balanced", "deep"], default="auto")
    process.add_argument("--max-cost", type=float, default=0.0)
    process.add_argument("--timeout", type=int, default=600)
    process.add_argument("--scene-threshold", type=float, default=3.0)
    process.add_argument("--min-scene-len", type=int, default=12)

    ingest = sub.add_parser("ingest", help="Create the persistent shot/sample index only.")
    ingest.add_argument("video", type=Path)
    ingest.add_argument("--db", type=Path, default=None)
    ingest.add_argument("--profile", choices=["safe", "balanced", "deep"], default="safe")
    ingest.add_argument("--scene-threshold", type=float, default=3.0)
    ingest.add_argument("--min-scene-len", type=int, default=12)

    analyze = sub.add_parser("analyze", help="Run transcript and shot-level VLM analysis.")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--db", type=Path, default=None)
    analyze.add_argument("--provider", choices=["ollama", "openrouter", "openai"], default=None)
    analyze.add_argument("--model", default=None)
    analyze.add_argument("--profile", choices=["safe", "balanced", "deep"], default="safe")
    analyze.add_argument("--max-cost", type=float, default=0.0)
    analyze.add_argument("--timeout", type=int, default=600)

    memory = sub.add_parser("memory", help="Build scenes, characters, events and global movie memory.")
    memory.add_argument("db", type=Path)
    memory.add_argument("--provider", choices=["ollama", "openrouter", "openai"], default=None)
    memory.add_argument("--model", default=None)
    memory.add_argument("--max-cost", type=float, default=0.0)
    memory.add_argument("--timeout", type=int, default=600)

    ask_cmd = sub.add_parser("ask", help="Ask a grounded question about an indexed movie.")
    ask_cmd.add_argument("db", type=Path)
    ask_cmd.add_argument("question", nargs="+")
    ask_cmd.add_argument("--video", type=Path, default=None)
    ask_cmd.add_argument("--provider", choices=["ollama", "openrouter", "openai"], default=None)
    ask_cmd.add_argument("--model", default=None)
    ask_cmd.add_argument("--max-cost", type=float, default=0.0)
    ask_cmd.add_argument("--timeout", type=int, default=600)
    ask_cmd.add_argument("--deep", action="store_true", help="Perform targeted high-density rewatch before answering.")

    status_cmd = sub.add_parser("status", help="Show analysis progress.")
    status_cmd.add_argument("db", type=Path)

    doctor = sub.add_parser("doctor", help="Check local resources without processing.")
    doctor.add_argument("video", type=Path)
    doctor.add_argument("--db", type=Path, default=None)
    doctor.add_argument("--profile", choices=["auto", "safe", "balanced", "deep"], default="auto")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "status":
        print(json.dumps(status(args.db), ensure_ascii=False, indent=2))
        return 0

    if args.command == "doctor":
        db = args.db or _db_for(args.video)
        snap, warnings = check_safe_to_start(args.video, db)
        profile = recommend_profile(snap) if args.profile == "auto" else PROFILES[args.profile]
        _console.print(format_preflight(snap, warnings, profile))
        return 0

    if args.command == "ingest":
        db = args.db or _db_for(args.video)
        profile = PROFILES[args.profile]
        result = ingest_index(
            args.video,
            db,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            max_samples_per_shot=profile.max_samples_per_shot,
        )
        _console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "analyze":
        db = args.db or _db_for(args.video)
        profile = PROFILES[args.profile]
        client = ProviderClient.from_env(
            args.provider, args.model, timeout=args.timeout, max_cost=args.max_cost
        )
        client.preflight()
        result = analyze_all_shots(
            args.video,
            db,
            client,
            scale=profile.frame_scale,
            max_samples_per_shot=profile.max_samples_per_shot,
            max_output_tokens=profile.max_output_tokens,
        )
        _console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("pending", 0) == 0 else 1

    if args.command == "memory":
        client = ProviderClient.from_env(
            args.provider, args.model, timeout=args.timeout, max_cost=args.max_cost
        )
        client.preflight()
        result = build_memory(args.db, client)
        _console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ask":
        question = " ".join(args.question)
        conn = __import__("vidlizer.longvideo.db", fromlist=["connect"]).connect(args.db)
        try:
            row = conn.execute("SELECT source_path, id FROM movies ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                raise RuntimeError("No movie found in database.")
            movie_path = args.video or Path(row["source_path"])
            client = ProviderClient.from_env(
                args.provider, args.model, timeout=args.timeout, max_cost=args.max_cost
            )
            client.preflight()
            with tempfile.TemporaryDirectory(prefix="moviemind_ask_") as temp:
                result = ask(
                    conn,
                    int(row["id"]),
                    question,
                    client,
                    video=movie_path,
                    deep=args.deep,
                    workspace=Path(temp),
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            conn.close()
        return 0

    if args.command == "process":
        db = args.db or _db_for(args.video)
        profile = args.profile
        if profile == "auto":
            snap, warnings = check_safe_to_start(args.video, db)
            selected = recommend_profile(snap)
            _console.print(format_preflight(snap, warnings, selected))
            profile = selected.name
        result = process_movie(
            args.video,
            db,
            provider=args.provider,
            model=args.model,
            profile=profile,
            max_cost=args.max_cost,
            timeout=args.timeout,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
        )
        _console.print()
        _console.print("[bold green]MovieMind movie understanding is ready.[/bold green]")
        _console.print(json.dumps({
            "db": result["db"],
            "duration_s": result["duration_s"],
            "shots": result["shots"],
            "samples": result["samples"],
            "provider": result["provider"],
        }, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
