from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_multitimeline_csv(path: str | Path) -> pd.DataFrame:
    """
    Loads a Google Trends export CSV (multiTimeline.csv).

    The file often starts with:
      Category: ...
      <blank line>
      Week,<term1>,<term2>,...
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find header row starting with "Week,"
    header_idx = None
    for i, line in enumerate(raw):
        if line.startswith("Week,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find 'Week,' header row in {p}")

    csv_text = "\n".join(raw[header_idx:])
    from io import StringIO

    df = pd.read_csv(StringIO(csv_text))
    if "Week" not in df.columns:
        raise ValueError("Expected a 'Week' column in the CSV.")

    df["Week"] = pd.to_datetime(df["Week"], errors="coerce")
    df = df.dropna(subset=["Week"]).set_index("Week").sort_index()

    # Normalize columns like "JR Cigars: (United States)" -> "JR Cigars"
    def clean_col(c: str) -> str:
        c = str(c)
        if ": (" in c:
            return c.split(": (", 1)[0].strip()
        return c.strip()

    df = df.rename(columns={c: clean_col(c) for c in df.columns})
    return df.apply(pd.to_numeric, errors="coerce").astype(float)


