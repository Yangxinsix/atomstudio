from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RenderResult:
    success: bool
    output_path: str
    frame_index: int
    message: str = ""
    elapsed_seconds: float = 0.0
    adjustments: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobReport:
    job_id: str
    success: bool
    outputs: list[str] = field(default_factory=list)
    failed_frames: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    message: str = ""
    frame_reports: list[RenderResult] = field(default_factory=list)


@dataclass
class BatchResult:
    success: bool
    reports: list[JobReport] = field(default_factory=list)


@dataclass
class AnimationResult:
    success: bool
    output_dir: str
    outputs: list[str] = field(default_factory=list)
    failed_frames: list[int] = field(default_factory=list)
    frame_reports: list[RenderResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    message: str = ""
