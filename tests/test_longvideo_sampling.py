from vidlizer.longvideo.models import Shot
from vidlizer.longvideo.sampling import adaptive_sample_points, sample_count


def make_shot(duration: float) -> Shot:
    return Shot("shot_00001", 100.0, 100.0 + duration, duration)


def test_sample_count_scales_with_duration():
    assert sample_count(1.0) == 3
    assert sample_count(5.0) == 5
    assert sample_count(12.0) == 7
    assert sample_count(60.0) == 12


def test_sample_points_stay_inside_shot():
    shot = make_shot(12.0)
    points = adaptive_sample_points(shot)

    assert len(points) == 7
    assert points[0].timestamp_s > shot.start_s
    assert points[-1].timestamp_s < shot.end_s
    assert all(shot.start_s < p.timestamp_s < shot.end_s for p in points)
    assert [p.ordinal for p in points] == list(range(1, 8))
    assert all(p.total == 7 for p in points)
