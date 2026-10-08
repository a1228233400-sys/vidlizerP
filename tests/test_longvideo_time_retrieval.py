from pathlib import Path

from vidlizer.longvideo.db import connect, time_evidence, rebuild_search_index, upsert_movie
from vidlizer.longvideo.models import MovieInfo


def test_time_evidence_hits_near_requested_minute(tmp_path: Path):
    conn = connect(tmp_path / "movie.db")
    movie = MovieInfo(
        path=str((tmp_path / "movie.mp4").resolve()),
        duration_s=7200.0,
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
        (movie_id, "shot_00010", 2790.0, 2820.0, "A train arrives at the station.", "station", "high", "{}"),
    )
    conn.commit()
    rebuild_search_index(conn, movie_id)

    rows = time_evidence(conn, movie_id, 2790.0, 2820.0, limit=5)
    assert rows
    assert rows[0]["evidence_id"] == "ev_shot_shot_00010"
