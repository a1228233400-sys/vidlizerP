"""SQLite persistence for long-video analysis artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import MovieInfo, SamplePoint, Shot


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    duration_s REAL NOT NULL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    video_codec TEXT,
    audio_codec TEXT,
    has_audio INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    shot_id TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    duration_s REAL NOT NULL,
    UNIQUE(movie_id, shot_id)
);

CREATE INDEX IF NOT EXISTS idx_shots_movie_time
ON shots(movie_id, start_s);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    shot_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    timestamp_s REAL NOT NULL,
    ordinal INTEGER NOT NULL,
    total INTEGER NOT NULL,
    UNIQUE(movie_id, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_samples_movie_time
ON samples(movie_id, timestamp_s);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_movie(conn: sqlite3.Connection, info: MovieInfo) -> int:
    conn.execute(
        """
        INSERT INTO movies (
            source_path, duration_s, width, height, fps,
            video_codec, audio_codec, has_audio
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            duration_s=excluded.duration_s,
            width=excluded.width,
            height=excluded.height,
            fps=excluded.fps,
            video_codec=excluded.video_codec,
            audio_codec=excluded.audio_codec,
            has_audio=excluded.has_audio
        """,
        (
            info.path,
            info.duration_s,
            info.width,
            info.height,
            info.fps,
            info.video_codec,
            info.audio_codec,
            int(info.has_audio),
        ),
    )
    row = conn.execute(
        "SELECT id FROM movies WHERE source_path = ?",
        (info.path,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def replace_shots(
    conn: sqlite3.Connection,
    movie_id: int,
    shots: Iterable[Shot],
) -> None:
    conn.execute("DELETE FROM samples WHERE movie_id = ?", (movie_id,))
    conn.execute("DELETE FROM shots WHERE movie_id = ?", (movie_id,))
    conn.executemany(
        """
        INSERT INTO shots (movie_id, shot_id, start_s, end_s, duration_s)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (movie_id, s.shot_id, s.start_s, s.end_s, s.duration_s)
            for s in shots
        ],
    )


def insert_samples(
    conn: sqlite3.Connection,
    movie_id: int,
    samples: Iterable[SamplePoint],
) -> None:
    conn.executemany(
        """
        INSERT INTO samples (
            movie_id, shot_id, sample_id, timestamp_s, ordinal, total
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                movie_id,
                s.shot_id,
                s.sample_id,
                s.timestamp_s,
                s.ordinal,
                s.total,
            )
            for s in samples
        ],
    )


def record_run(
    conn: sqlite3.Connection,
    movie_id: int,
    stage: str,
    status: str,
    details: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_runs (movie_id, stage, status, details_json)
        VALUES (?, ?, ?, ?)
        """,
        (movie_id, stage, status, json.dumps(details, ensure_ascii=False)),
    )
