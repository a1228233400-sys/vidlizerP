from pathlib import Path

from vidlizer.longvideo.db import connect, insert_samples, replace_shots, upsert_movie
from vidlizer.longvideo.models import MovieInfo, Shot
from vidlizer.longvideo.sampling import adaptive_sample_points


def test_db_persists_movie_shots_and_samples(tmp_path: Path):
    db_path = tmp_path / "movie.db"
    conn = connect(db_path)
    info = MovieInfo(
        path=str(tmp_path / "movie.mp4"),
        duration_s=120.0,
        width=1920,
        height=1080,
        fps=24.0,
        video_codec="h264",
        audio_codec="aac",
        has_audio=True,
    )
    movie_id = upsert_movie(conn, info)
    shot = Shot("shot_00001", 0.0, 4.0, 4.0)
    replace_shots(conn, movie_id, [shot])
    insert_samples(conn, movie_id, adaptive_sample_points(shot))
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 4
    conn.close()
