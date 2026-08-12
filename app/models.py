"""Immutable view models shared by evidence, inference, and UI layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AppMode(StrEnum):
    """Runtime capabilities exposed by the portfolio UI."""

    LIVE = "live"
    EVIDENCE = "evidence"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class Metric:
    label: str
    display_value: str
    context: str
    evidence_path: str


@dataclass(frozen=True)
class EvidenceSummary:
    defect_classes: int
    grouped_map50: Metric
    leakage_effect: Metric
    ort_cuda_p50: Metric
    strict_parity_passed: bool


@dataclass(frozen=True)
class AppState:
    mode: AppMode
    evidence: EvidenceSummary | None
    contract: dict
    status_title: str
    status_detail: str
    inference_enabled: bool
    errors: tuple[str, ...] = ()
