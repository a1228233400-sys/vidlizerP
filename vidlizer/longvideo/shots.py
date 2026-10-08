"""Shot detection for long-form video."""

from __future__ import annotations

from pathlib import Path

from .models import Shot


def detect_shots(video: Path, threshold: float = 3.0, min_scene_len: int = 12) -> list[Shot]:
    """Detect visual shots with PySceneDetect.

    The import stays optional so the original Vidlizer installation remains
    lightweight. Shot detection is the first long-video-specific stage.
    """
    try:
        from scenedetect import AdaptiveDetector, SceneManager, open_video
    except ImportError as exc:
        raise RuntimeError(
            "PySceneDetect is required for long-video ingestion. "
            "Install with: pip install 'vidlizer[longvideo]'"
        ) from exc

    try:
        video_manager = open_video(str(video))
        manager = SceneManager()
        manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=min_scene_len,
            )
        )
        manager.detect_scenes(video=video_manager)
        raw_scenes = manager.get_scene_list()
    except Exception as exc:
        raise RuntimeError(f"shot detection failed: {exc}") from exc

    shots: list[Shot] = []
    for idx, (start_tc, end_tc) in enumerate(raw_scenes, start=1):
        start_s = start_tc.get_seconds()
        end_s = end_tc.get_seconds()
        if end_s <= start_s:
            continue
        shots.append(
            Shot(
                shot_id=f"shot_{idx:05d}",
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                duration_s=round(end_s - start_s, 3),
            )
        )

    # Some videos yield no cuts. Preserve the whole movie as one shot.
    if not shots:
        from .media import probe_movie

        info = probe_movie(video)
        shots = [
            Shot(
                shot_id="shot_00001",
                start_s=0.0,
                end_s=round(info.duration_s, 3),
                duration_s=round(info.duration_s, 3),
            )
        ]

    return shots
