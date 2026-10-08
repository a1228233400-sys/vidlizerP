"""CLI entry point for the long-video foundation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ingest_movie, print_ingest_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movieai",
        description="Long-form video / movie understanding engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index a movie into shots and sample points.")
    ingest.add_argument("video", type=Path)
    ingest.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite output path (default: <video>.movie.db)",
    )
    ingest.add_argument(
        "--scene-threshold",
        type=float,
        default=3.0,
        help="PySceneDetect AdaptiveDetector threshold.",
    )
    ingest.add_argument(
        "--min-scene-len",
        type=int,
        default=12,
        help="Minimum detected shot length in frames.",
    )
    ingest.add_argument(
        "--max-samples-per-shot",
        type=int,
        default=12,
        help="Maximum planned visual observations per shot.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        db = args.db or args.video.with_suffix(args.video.suffix + ".movie.db")
        result = ingest_movie(
            args.video,
            db,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            max_samples_per_shot=args.max_samples_per_shot,
        )
        print_ingest_result(result)


if __name__ == "__main__":
    main()
