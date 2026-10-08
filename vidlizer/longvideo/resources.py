"""Hardware/resource guardrails for long-video processing."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    name: str
    max_samples_per_shot: int
    frame_scale: int
    max_output_tokens: int
    context_tokens: int


PROFILES = {
    "safe": Profile("safe", 8, 512, 1536, 8192),
    "balanced": Profile("balanced", 12, 640, 2048, 12288),
    "deep": Profile("deep", 16, 768, 3072, 16384),
}


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_count: int
    memory_total_gb: float | None
    memory_available_gb: float | None
    disk_free_gb: float
    platform: str

    def to_dict(self) -> dict:
        return {
            "cpu_count": self.cpu_count,
            "memory_total_gb": self.memory_total_gb,
            "memory_available_gb": self.memory_available_gb,
            "disk_free_gb": self.disk_free_gb,
            "platform": self.platform,
        }


def _memory_gb() -> tuple[float | None, float | None]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None, None
    vm = psutil.virtual_memory()
    return vm.total / (1024**3), vm.available / (1024**3)


def snapshot(path: Path) -> ResourceSnapshot:
    usage = shutil.disk_usage(path.parent if path.parent.exists() else Path.cwd())
    total, available = _memory_gb()
    return ResourceSnapshot(
        cpu_count=os.cpu_count() or 1,
        memory_total_gb=total,
        memory_available_gb=available,
        disk_free_gb=usage.free / (1024**3),
        platform=f"{platform.system()} {platform.machine()}",
    )


def check_safe_to_start(video: Path, db_path: Path) -> tuple[ResourceSnapshot, list[str]]:
    snap = snapshot(video)
    warnings: list[str] = []
    if snap.disk_free_gb < 2.0:
        raise RuntimeError(
            f"Only {snap.disk_free_gb:.1f} GB disk space is free. "
            "MovieMind refuses to start below 2 GB."
        )
    if snap.memory_available_gb is not None:
        if snap.memory_available_gb < 1.5:
            raise RuntimeError(
                f"Only {snap.memory_available_gb:.1f} GB RAM is currently available. "
                "Free memory before starting the analysis."
            )
        if snap.memory_available_gb < 4.0:
            warnings.append("Available RAM is below 4 GB; use the safe profile.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return snap, warnings


def recommend_profile(snap: ResourceSnapshot) -> Profile:
    if snap.memory_available_gb is None or snap.memory_available_gb < 8:
        return PROFILES["safe"]
    if snap.memory_available_gb < 16:
        return PROFILES["balanced"]
    return PROFILES["deep"]


def format_preflight(snap: ResourceSnapshot, warnings: list[str], profile: Profile) -> str:
    lines = [
        f"Platform: {snap.platform}",
        f"CPU: {snap.cpu_count} logical cores",
        (
            f"RAM available: {snap.memory_available_gb:.1f} GB"
            if snap.memory_available_gb is not None
            else "RAM available: unknown"
        ),
        f"Disk free: {snap.disk_free_gb:.1f} GB",
        f"Profile: {profile.name} | max samples/shot: {profile.max_samples_per_shot}",
        "Workers: 1",
    ]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines)
