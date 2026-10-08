"""Data models for MovieMind."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MovieInfo:
    path: str
    duration_s: float
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Shot:
    shot_id: str
    start_s: float
    end_s: float
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplePoint:
    sample_id: str
    shot_id: str
    timestamp_s: float
    ordinal: int
    total: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scene:
    scene_id: str
    start_s: float
    end_s: float
    shot_ids: list[str]
    title: str
    location: str
    characters: list[str]
    summary: str
    dramatic_purpose: str
    importance: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
