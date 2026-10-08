"""Adaptive temporal sampling for individual shots."""

from __future__ import annotations

from .models import SamplePoint, Shot


def sample_count(duration_s: float, minimum: int = 3, maximum: int = 12) -> int:
    """Choose a baseline sample count from shot duration.

    This deliberately stays deterministic in V0.1. Later versions can add
    motion/dialogue/importance signals without changing the stored interface.
    """
    if duration_s <= 1.5:
        return minimum
    if duration_s <= 4:
        return min(maximum, 4)
    if duration_s <= 8:
        return min(maximum, 5)
    if duration_s <= 15:
        return min(maximum, 7)
    if duration_s <= 30:
        return min(maximum, 9)
    return maximum


def adaptive_sample_points(shot: Shot, maximum: int = 12) -> list[SamplePoint]:
    """Generate evenly distributed observations inside a shot.

    Endpoints are slightly inset to avoid duplicate frames at adjacent cuts.
    """
    count = sample_count(shot.duration_s, maximum=maximum)
    if count == 1:
        times = [shot.start_s]
    else:
        inset = min(0.15, shot.duration_s / 10)
        start = shot.start_s + inset
        end = max(start, shot.end_s - inset)
        step = (end - start) / (count - 1) if count > 1 else 0.0
        times = [start + i * step for i in range(count)]

    return [
        SamplePoint(
            sample_id=f"{shot.shot_id}_s{i:02d}",
            shot_id=shot.shot_id,
            timestamp_s=round(t, 3),
            ordinal=i + 1,
            total=count,
        )
        for i, t in enumerate(times)
    ]
