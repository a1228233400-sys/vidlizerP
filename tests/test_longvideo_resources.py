from pathlib import Path

from vidlizer.longvideo.resources import PROFILES, recommend_profile, snapshot


def test_profiles_are_bounded():
    assert PROFILES["safe"].max_samples_per_shot < PROFILES["deep"].max_samples_per_shot
    assert PROFILES["safe"].frame_scale <= 512


def test_snapshot_has_cpu_and_disk(tmp_path: Path):
    snap = snapshot(tmp_path / "movie.mp4")
    assert snap.cpu_count >= 1
    assert snap.disk_free_gb > 0
