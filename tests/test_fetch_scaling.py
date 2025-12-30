from __future__ import annotations

import pandas as pd

from gtrends_analyzer.fetch import _compute_anchor_scale_factor


def test_anchor_scale_factor_median_ratio() -> None:
    idx = pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-21", "2024-01-28"])
    base = pd.Series([50, 50, 50, 50], index=idx)
    batch = pd.Series([25, 25, 25, 25], index=idx)
    scale = _compute_anchor_scale_factor(base_anchor=base, batch_anchor=batch)
    assert scale == 2.0


