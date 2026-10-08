from pathlib import Path

from vidlizer.longvideo.db import connect, rebuild_search_index
from vidlizer.longvideo.models import MovieInfo
from vidlizer.longvideo.db import upsert_movie


def test_evidence_search(tmp_path: Path):
    conn = connect(tmp_path / "movie.db")
    movie = MovieInfo(
        path=str((tmp_path / "movie.mp4").resolve()),
        duration_s=120.0,
        width=1920,
        height=1080,
        fps=24.0,
        video_codec="h264",
        audio_codec="aac",
        has_audio=True,
    )
    movie_id = upsert_movie(conn, movie)
    conn.execute(
        """
        INSERT INTO shot_observations
        (movie_id, shot_id, start_s, end_s, summary, location, importance, data_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (movie_id, "shot_00001", 10.0, 12.0, "Alice finds a red box in the hotel room.", "hotel room", "high", "{}"),
    )
    conn.commit()
    rebuild_search_index(conn, movie_id)
    rows = conn.execute(
        """
        SELECT evidence_id, content FROM search_index
        WHERE movie_id=? AND search_index MATCH ?
        """,
        (str(movie_id), '"red" AND "box"'),
    ).fetchall()
    assert rows
    assert "red box" in rows[0]["content"]
