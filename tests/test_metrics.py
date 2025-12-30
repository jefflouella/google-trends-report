from __future__ import annotations

import pandas as pd

from gtrends_analyzer.metrics import compute_metrics


def test_compute_metrics_yearly_yoy() -> None:
    idx = pd.to_datetime(
        [
            "2021-01-03",
            "2021-01-10",
            "2022-01-02",
            "2022-01-09",
        ]
    )
    df = pd.DataFrame({"A": [10, 10, 20, 20], "B": [50, 50, 25, 25]}, index=idx)

    m = compute_metrics(df)
    # yearly avg: 2021 A=10, 2022 A=20 => +100%
    assert round(float(m.yearly_yoy.loc[2022, "A"]), 6) == 100.0
    # yearly avg: 2021 B=50, 2022 B=25 => -50%
    assert round(float(m.yearly_yoy.loc[2022, "B"]), 6) == -50.0


