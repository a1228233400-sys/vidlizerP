"""SQLite persistence and evidence search for MovieMind."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import MovieInfo, SamplePoint, Scene, Shot


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
    analysis_status TEXT NOT NULL DEFAULT 'pending',
    UNIQUE(movie_id, shot_id)
);
CREATE INDEX IF NOT EXISTS idx_shots_movie_time ON shots(movie_id, start_s);

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
CREATE INDEX IF NOT EXISTS idx_samples_movie_time ON samples(movie_id, timestamp_s);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_transcripts_movie_time ON transcripts(movie_id, start_s);

CREATE TABLE IF NOT EXISTS shot_observations (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    shot_id TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    summary TEXT NOT NULL,
    location TEXT,
    importance TEXT,
    data_json TEXT NOT NULL,
    UNIQUE(movie_id, shot_id)
);

CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    scene_id TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    title TEXT NOT NULL,
    location TEXT,
    characters_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    dramatic_purpose TEXT,
    importance TEXT,
    data_json TEXT NOT NULL,
    UNIQUE(movie_id, scene_id)
);

CREATE TABLE IF NOT EXISTS scene_shots (
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    scene_id TEXT NOT NULL,
    shot_id TEXT NOT NULL,
    PRIMARY KEY(movie_id, scene_id, shot_id)
);

CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    character_key TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    description TEXT NOT NULL,
    arc_notes TEXT NOT NULL,
    data_json TEXT NOT NULL,
    UNIQUE(movie_id, character_key)
);

CREATE TABLE IF NOT EXISTS character_mentions (
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    character_key TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(movie_id, character_key, scene_id, role)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    scene_id TEXT,
    timestamp_s REAL NOT NULL,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    characters_json TEXT NOT NULL,
    importance TEXT NOT NULL,
    data_json TEXT NOT NULL,
    UNIQUE(movie_id, event_id)
);

CREATE TABLE IF NOT EXISTS event_relations (
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    from_event TEXT NOT NULL,
    to_event TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    PRIMARY KEY(movie_id, from_event, to_event, relation_type)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY(movie_id, memory_type)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
    evidence_id UNINDEXED,
    movie_id UNINDEXED,
    source_type UNINDEXED,
    start_s UNINDEXED,
    end_s UNINDEXED,
    content
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
            info.path, info.duration_s, info.width, info.height, info.fps,
            info.video_codec, info.audio_codec, int(info.has_audio),
        ),
    )
    row = conn.execute("SELECT id FROM movies WHERE source_path = ?", (info.path,)).fetchone()
    assert row is not None
    return int(row["id"])


def replace_shots(conn: sqlite3.Connection, movie_id: int, shots: Iterable[Shot]) -> None:
    conn.execute("DELETE FROM samples WHERE movie_id = ?", (movie_id,))
    conn.execute("DELETE FROM shots WHERE movie_id = ?", (movie_id,))
    conn.executemany(
        "INSERT INTO shots (movie_id, shot_id, start_s, end_s, duration_s) VALUES (?, ?, ?, ?, ?)",
        [(movie_id, s.shot_id, s.start_s, s.end_s, s.duration_s) for s in shots],
    )


def insert_samples(conn: sqlite3.Connection, movie_id: int, samples: Iterable[SamplePoint]) -> None:
    conn.executemany(
        """
        INSERT INTO samples (movie_id, shot_id, sample_id, timestamp_s, ordinal, total)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(movie_id, s.shot_id, s.sample_id, s.timestamp_s, s.ordinal, s.total) for s in samples],
    )


def insert_transcript(conn: sqlite3.Connection, movie_id: int, segments: Iterable[dict], source: str) -> None:
    conn.execute("DELETE FROM transcripts WHERE movie_id = ?", (movie_id,))
    conn.executemany(
        "INSERT INTO transcripts (movie_id, start_s, end_s, text, source) VALUES (?, ?, ?, ?, ?)",
        [(movie_id, float(s["start"]), float(s["end"]), s["text"], source) for s in segments],
    )


def pending_shots(conn: sqlite3.Connection, movie_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT shot_id, start_s, end_s, duration_s FROM shots WHERE movie_id=? AND analysis_status!='complete' ORDER BY start_s",
        (movie_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_shot_status(conn: sqlite3.Connection, movie_id: int, shot_id: str, status: str) -> None:
    conn.execute(
        "UPDATE shots SET analysis_status=? WHERE movie_id=? AND shot_id=?",
        (status, movie_id, shot_id),
    )


def store_shot_observation(conn: sqlite3.Connection, movie_id: int, observation: dict) -> None:
    conn.execute(
        """
        INSERT INTO shot_observations
        (movie_id, shot_id, start_s, end_s, summary, location, importance, data_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id, shot_id) DO UPDATE SET
          start_s=excluded.start_s,
          end_s=excluded.end_s,
          summary=excluded.summary,
          location=excluded.location,
          importance=excluded.importance,
          data_json=excluded.data_json
        """,
        (
            movie_id,
            observation["shot_id"],
            observation["start_s"],
            observation["end_s"],
            observation.get("summary", ""),
            observation.get("location", ""),
            observation.get("importance", "medium"),
            json.dumps(observation, ensure_ascii=False),
        ),
    )


def load_shot_observations(conn: sqlite3.Connection, movie_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT shot_id, start_s, end_s, summary, location, importance, data_json
        FROM shot_observations WHERE movie_id=? ORDER BY start_s
        """,
        (movie_id,),
    ).fetchall()
    out = []
    for row in rows:
        item = json.loads(row["data_json"])
        item["observation_text"] = row["summary"]
        item["start_s"] = row["start_s"]
        item["end_s"] = row["end_s"]
        out.append(item)
    return out


def replace_scenes(conn: sqlite3.Connection, movie_id: int, scenes: Iterable[Scene]) -> None:
    conn.execute("DELETE FROM scene_shots WHERE movie_id = ?", (movie_id,))
    conn.execute("DELETE FROM scenes WHERE movie_id = ?", (movie_id,))
    for scene in scenes:
        conn.execute(
            """
            INSERT INTO scenes
            (movie_id, scene_id, start_s, end_s, title, location, characters_json,
             summary, dramatic_purpose, importance, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                movie_id, scene.scene_id, scene.start_s, scene.end_s, scene.title,
                scene.location, json.dumps(scene.characters, ensure_ascii=False),
                scene.summary, scene.dramatic_purpose, scene.importance,
                json.dumps(scene.details, ensure_ascii=False),
            ),
        )
        conn.executemany(
            "INSERT INTO scene_shots (movie_id, scene_id, shot_id) VALUES (?, ?, ?)",
            [(movie_id, scene.scene_id, shot_id) for shot_id in scene.shot_ids],
        )


def load_scenes(conn: sqlite3.Connection, movie_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM scenes WHERE movie_id=? ORDER BY start_s", (movie_id,)
    ).fetchall()
    return [
        {
            **dict(row),
            "characters": json.loads(row["characters_json"]),
            "details": json.loads(row["data_json"]),
        }
        for row in rows
    ]


def upsert_character(conn: sqlite3.Connection, movie_id: int, data: dict) -> None:
    key = str(data["character_key"])
    conn.execute(
        """
        INSERT INTO characters
        (movie_id, character_key, name, aliases_json, description, arc_notes, data_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id, character_key) DO UPDATE SET
          name=excluded.name,
          aliases_json=excluded.aliases_json,
          description=excluded.description,
          arc_notes=excluded.arc_notes,
          data_json=excluded.data_json
        """,
        (
            movie_id, key, data.get("name", "Unknown"),
            json.dumps(data.get("aliases", []), ensure_ascii=False),
            data.get("description", ""), data.get("arc_notes", ""),
            json.dumps(data, ensure_ascii=False),
        ),
    )
    mention = data.get("mention")
    if mention:
        conn.execute(
            """
            INSERT OR IGNORE INTO character_mentions
            (movie_id, character_key, scene_id, role)
            VALUES (?, ?, ?, ?)
            """,
            (
                movie_id, key, mention.get("scene_id", ""),
                mention.get("role", "present"),
            ),
        )


def upsert_event(conn: sqlite3.Connection, movie_id: int, data: dict) -> None:
    conn.execute(
        """
        INSERT INTO events
        (movie_id, event_id, scene_id, timestamp_s, type, description, characters_json,
         importance, data_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id, event_id) DO UPDATE SET
          scene_id=excluded.scene_id,
          timestamp_s=excluded.timestamp_s,
          type=excluded.type,
          description=excluded.description,
          characters_json=excluded.characters_json,
          importance=excluded.importance,
          data_json=excluded.data_json
        """,
        (
            movie_id, data["event_id"], data.get("scene_id"), data.get("timestamp_s", 0.0),
            data.get("type", "other"), data.get("description", ""),
            json.dumps(data.get("characters", []), ensure_ascii=False),
            data.get("importance", "medium"), json.dumps(data, ensure_ascii=False),
        ),
    )


def upsert_event_relation(conn: sqlite3.Connection, movie_id: int, from_event: str, to_event: str, relation_type: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO event_relations
        (movie_id, from_event, to_event, relation_type)
        VALUES (?, ?, ?, ?)
        """,
        (movie_id, from_event, to_event, relation_type),
    )


def upsert_memory(conn: sqlite3.Connection, movie_id: int, memory_type: str, data: dict) -> None:
    conn.execute(
        """
        INSERT INTO memories (movie_id, memory_type, data_json)
        VALUES (?, ?, ?)
        ON CONFLICT(movie_id, memory_type) DO UPDATE SET data_json=excluded.data_json
        """,
        (movie_id, memory_type, json.dumps(data, ensure_ascii=False)),
    )


def get_global_memory(conn: sqlite3.Connection, movie_id: int) -> dict | None:
    row = conn.execute(
        "SELECT data_json FROM memories WHERE movie_id=? AND memory_type='global'",
        (movie_id,),
    ).fetchone()
    return json.loads(row["data_json"]) if row else None


def rebuild_search_index(conn: sqlite3.Connection, movie_id: int) -> int:
    conn.execute("DELETE FROM search_index WHERE movie_id = ?", (str(movie_id),))
    conn.execute("DELETE FROM evidence WHERE movie_id = ?", (movie_id,))

    evidence_rows = []
    for row in conn.execute(
        "SELECT shot_id, start_s, end_s, summary, data_json FROM shot_observations WHERE movie_id=?",
        (movie_id,),
    ):
        try:
            detail = json.loads(row["data_json"])
        except (TypeError, json.JSONDecodeError):
            detail = {}
        content_text = " ".join(
            [
                row["summary"],
                str(detail.get("location", "")),
                str(detail.get("dialogue_context", "")),
                " ".join(map(str, detail.get("objects", []))),
                " ".join(map(str, detail.get("text_visible", []))),
                " ".join(
                    x.get("description", "") if isinstance(x, dict) else str(x)
                    for x in detail.get("actions", [])
                ),
            ]
        ).strip()
        evidence_rows.append(
            (f"ev_shot_{row['shot_id']}", movie_id, "shot", row["shot_id"],
             row["start_s"], row["end_s"], content_text)
        )
    for row in conn.execute(
        "SELECT id, start_s, end_s, text FROM transcripts WHERE movie_id=?",
        (movie_id,),
    ):
        evidence_rows.append(
            (f"ev_tr_{row['id']}", movie_id, "transcript", str(row["id"]),
             row["start_s"], row["end_s"], row["text"])
        )
    for row in conn.execute(
        "SELECT scene_id, start_s, end_s, title, summary, dramatic_purpose FROM scenes WHERE movie_id=?",
        (movie_id,),
    ):
        content = f"{row['title']}. {row['summary']} Purpose: {row['dramatic_purpose'] or ''}"
        evidence_rows.append(
            (f"ev_scene_{row['scene_id']}", movie_id, "scene", row["scene_id"],
             row["start_s"], row["end_s"], content)
        )
    for row in conn.execute(
        "SELECT event_id, timestamp_s, description, type FROM events WHERE movie_id=?",
        (movie_id,),
    ):
        evidence_rows.append(
            (f"ev_event_{row['event_id']}", movie_id, "event", row["event_id"],
             row["timestamp_s"], row["timestamp_s"], f"{row['type']}: {row['description']}")
        )
    memory = get_global_memory(conn, movie_id)
    if memory:
        memory_text = json.dumps(memory, ensure_ascii=False)
        evidence_rows.append(
            ("ev_global_movie", movie_id, "global", "global", 0.0, 0.0, memory_text)
        )

    for row in conn.execute(
        "SELECT character_key, name, description, arc_notes FROM characters WHERE movie_id=?",
        (movie_id,),
    ):
        evidence_rows.append(
            (f"ev_char_{row['character_key']}", movie_id, "character", row["character_key"],
             0.0, 0.0, f"{row['name']}. {row['description']} Arc: {row['arc_notes']}")
        )

    conn.executemany(
        "INSERT INTO evidence (evidence_id, movie_id, source_type, source_id, start_s, end_s, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
        evidence_rows,
    )
    conn.executemany(
        "INSERT INTO search_index (evidence_id, movie_id, source_type, start_s, end_s, content) VALUES (?, ?, ?, ?, ?, ?)",
        evidence_rows,
    )
    conn.commit()
    return len(evidence_rows)


def search_evidence(conn, movie_id: int, query: str, limit: int = 12) -> list[dict]:
    rows = conn.execute(
        """
        SELECT evidence_id, source_type, start_s, end_s, content
        FROM search_index
        WHERE movie_id=? AND search_index MATCH ?
        ORDER BY bm25(search_index)
        LIMIT ?
        """,
        (str(movie_id), query, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def load_movie_info(conn, movie_id: int) -> dict:
    row = conn.execute("SELECT * FROM movies WHERE id=?", (movie_id,)).fetchone()
    if not row:
        raise RuntimeError(f"Movie id {movie_id} not found.")
    return dict(row)


def movie_id_for_path(conn, path: Path) -> int | None:
    row = conn.execute(
        "SELECT id FROM movies WHERE source_path=?",
        (str(path.resolve()),),
    ).fetchone()
    return int(row["id"]) if row else None


def record_run(conn, movie_id: int, stage: str, status: str, details: dict) -> None:
    conn.execute(
        "INSERT INTO pipeline_runs (movie_id, stage, status, details_json) VALUES (?, ?, ?, ?)",
        (movie_id, stage, status, json.dumps(details, ensure_ascii=False)),
    )


def stage_complete(conn, movie_id: int, stage: str) -> bool:
    row = conn.execute(
        "SELECT status FROM pipeline_runs WHERE movie_id=? AND stage=? ORDER BY id DESC LIMIT 1",
        (movie_id, stage),
    ).fetchone()
    return bool(row and row["status"] == "complete")


def clear_memory_layers(conn: sqlite3.Connection, movie_id: int) -> None:
    """Clear derived memory so a failed memory build can be restarted cleanly."""
    conn.execute("DELETE FROM event_relations WHERE movie_id=?", (movie_id,))
    conn.execute("DELETE FROM events WHERE movie_id=?", (movie_id,))
    conn.execute("DELETE FROM character_mentions WHERE movie_id=?", (movie_id,))
    conn.execute("DELETE FROM characters WHERE movie_id=?", (movie_id,))
    conn.execute("DELETE FROM memories WHERE movie_id=?", (movie_id,))
    conn.commit()


def like_search_evidence(
    conn: sqlite3.Connection,
    movie_id: int,
    terms: list[str],
    limit: int = 12,
) -> list[dict]:
    clean = [term.strip() for term in terms if term.strip()]
    if not clean:
        return []
    clauses = " OR ".join("content LIKE ?" for _ in clean)
    params = [f"%{term}%" for term in clean]
    rows = conn.execute(
        f"""
        SELECT evidence_id, source_type, start_s, end_s, content
        FROM evidence
        WHERE movie_id=? AND ({clauses})
        ORDER BY start_s
        LIMIT ?
        """,
        (movie_id, *params, limit),
    ).fetchall()
    return [dict(r) for r in rows]
