from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunArgs:
    report_type: str  # "competitive" | "brand-health" | "category"
    bundle: bool
    main: str
    terms: list[str]
    geo: str
    timeframe: str
    gprop: str
    include_related: bool
    include_region: bool
    include_suggestions: bool
    cache_dir: Path
    cache_ttl_hours: float
    refresh: bool
    min_request_interval_seconds: float
    max_retries: int
    include_js: str
    offline: bool
    ai_model: str | None
    no_ai: bool
    out: Path | None
    csv: Path | None
    clusters: list[str]


