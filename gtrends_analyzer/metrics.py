from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class MetricsBundle:
    weekly: pd.DataFrame
    yearly_avg: pd.DataFrame
    yearly_yoy: pd.DataFrame
    yearly_rank: pd.DataFrame
    yearly_gap_to_leader: pd.DataFrame
    rebased_yearly: pd.DataFrame
    month_of_year_avg: pd.DataFrame
    share_of_search: pd.DataFrame
    corr: pd.DataFrame


def compute_metrics(weekly_df: pd.DataFrame) -> MetricsBundle:
    if weekly_df is None or weekly_df.empty:
        raise ValueError("weekly_df is empty")

    df = weekly_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    # keep numeric columns only, coerce to float
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)

    years = df.index.year

    # Drop partial years (common when timeframe starts mid-year or CSV includes a partial first year).
    # This keeps yearly rollups + rebasing intuitive.
    year_counts = df.groupby(years).size()
    min_weeks_per_year = 26
    full_years = year_counts[year_counts >= min_weeks_per_year].index
    # If nothing meets the threshold (common in tiny unit tests / small samples), don't drop everything.
    if len(full_years) == 0:
        df_yearly = df
    else:
        df_yearly = df[df.index.year.isin(full_years)].copy()

    yearly_avg = df_yearly.groupby(df_yearly.index.year).mean(numeric_only=True)
    yearly_avg.index.name = "year"

    yearly_yoy = yearly_avg.pct_change().replace([np.inf, -np.inf], np.nan) * 100.0

    yearly_rank = yearly_avg.rank(axis=1, method="min", ascending=False)

    leader_val = yearly_avg.max(axis=1)
    yearly_gap_to_leader = yearly_avg.apply(lambda row: leader_val.loc[row.name] - row, axis=1)

    base_year = int(yearly_avg.index.min())
    base_vals = yearly_avg.loc[base_year].replace(0, np.nan)
    rebased_yearly = (yearly_avg.divide(base_vals, axis=1) * 100.0).replace([np.inf, -np.inf], np.nan)

    month_of_year_avg = df.groupby(df.index.month).mean(numeric_only=True)
    month_of_year_avg.index.name = "month"

    denom = df.sum(axis=1).replace(0, np.nan)
    share_of_search = df.divide(denom, axis=0)

    corr = df.corr()

    return MetricsBundle(
        weekly=df,
        yearly_avg=yearly_avg,
        yearly_yoy=yearly_yoy,
        yearly_rank=yearly_rank,
        yearly_gap_to_leader=yearly_gap_to_leader,
        rebased_yearly=rebased_yearly,
        month_of_year_avg=month_of_year_avg,
        share_of_search=share_of_search,
        corr=corr,
    )


def build_facts_packet(
    *,
    report_type: str,
    main_term: str,
    metrics: MetricsBundle,
    geo: str,
    timeframe: str,
    gprop: str,
    clusters: list[str] | None = None,
) -> dict:
    """
    A structured JSON-like dict used to ground LLM commentary.
    Keep this stable and explicit: prefer numbers/dates over prose.
    """
    yearly_avg = metrics.yearly_avg
    yearly_yoy = metrics.yearly_yoy
    yearly_rank = metrics.yearly_rank
    yearly_gap = metrics.yearly_gap_to_leader

    years = [int(y) for y in yearly_avg.index.tolist()]
    start_year = years[0]
    end_year = years[-1]
    terms = list(yearly_avg.columns)

    overall: dict[str, dict] = {}
    for t in terms:
        start = float(yearly_avg.loc[start_year, t])
        end = float(yearly_avg.loc[end_year, t])
        pct = None
        if start != 0:
            pct = (end - start) / start * 100.0
        overall[t] = {
            "start_year": start_year,
            "end_year": end_year,
            "start_avg": start,
            "end_avg": end,
            "pct_change": pct,
            "abs_change": end - start,
        }

    leaders: dict[int, dict] = {}
    for y in years:
        row = yearly_avg.loc[y]
        leader_term = str(row.idxmax())
        leaders[y] = {"term": leader_term, "avg": float(row.max())}

    rank_changes: list[dict] = []
    for t in terms:
        r0 = float(yearly_rank.loc[start_year, t])
        r1 = float(yearly_rank.loc[end_year, t])
        rank_changes.append({"term": t, "start_rank": r0, "end_rank": r1, "delta": r1 - r0})
    rank_changes = sorted(rank_changes, key=lambda x: x["delta"])

    # Biggest YoY drop/gain for main term and overall
    def _yoy_extrema(t: str) -> dict:
        s = yearly_yoy[t].dropna()
        if s.empty:
            return {"min": None, "max": None}
        min_year = int(s.idxmin())
        max_year = int(s.idxmax())
        return {
            "min": {"year": min_year, "pct": float(s.loc[min_year])},
            "max": {"year": max_year, "pct": float(s.loc[max_year])},
        }

    yoy_extrema = {t: _yoy_extrema(t) for t in terms}

    # Seasonality peaks: best/worst month-of-year averages per term
    seasonality_peaks: dict[str, dict] = {}
    moy = metrics.month_of_year_avg
    for t in terms:
        s = moy[t].dropna() if t in moy.columns else pd.Series(dtype=float)
        if s.empty:
            seasonality_peaks[t] = {"max": None, "min": None}
            continue
        max_m = int(s.idxmax())
        min_m = int(s.idxmin())
        seasonality_peaks[t] = {
            "max": {"month": max_m, "avg": float(s.loc[max_m])},
            "min": {"month": min_m, "avg": float(s.loc[min_m])},
        }

    # Main-term inflection: lowest YoY year and largest gap-to-leader year
    main_yoy = yearly_yoy[main_term].dropna()
    main_low_yoy = None
    if not main_yoy.empty:
        y = int(main_yoy.idxmin())
        main_low_yoy = {"year": y, "pct": float(main_yoy.loc[y])}

    main_gap = yearly_gap[main_term].dropna()
    main_max_gap = None
    if not main_gap.empty:
        y = int(main_gap.idxmax())
        main_max_gap = {"year": y, "gap": float(main_gap.loc[y])}

    # Main-term rolling drawdown (12-week window)
    main_dd = None
    if main_term in metrics.weekly.columns and not metrics.weekly[main_term].dropna().empty:
        s = metrics.weekly[main_term].astype(float)
        roll_max = s.rolling(window=12, min_periods=2).max()
        dd = (s / roll_max - 1.0) * 100.0
        dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
        if not dd.empty:
            trough_date = dd.idxmin()
            trough_pct = float(dd.loc[trough_date])
            window = s.loc[trough_date - pd.Timedelta(days=7 * 11) : trough_date]
            peak_date = None
            peak_val = None
            if not window.empty:
                peak_date = window.idxmax()
                peak_val = float(window.loc[peak_date])
            main_dd = {
                "window_weeks": 12,
                "trough_date": trough_date.date().isoformat(),
                "drawdown_pct": trough_pct,
                "peak_date": peak_date.date().isoformat() if peak_date is not None else None,
                "peak_value": peak_val,
            }

    base: dict = {
        "params": {"geo": geo, "timeframe": timeframe, "gprop": gprop},
        "report_type": report_type,
        "main_term": main_term,
        "terms": terms,
        "years": years,
        "overall": overall,
        "leaders_by_year": leaders,
        "rank_changes": rank_changes,
        "yoy_extrema": yoy_extrema,
        "seasonality_peaks": seasonality_peaks,
        "main_term_flags": {
            "worst_yoy": main_low_yoy,
            "max_gap_to_leader": main_max_gap,
            "max_12w_drawdown": main_dd,
        },
    }

    # Template-specific facts
    if report_type == "brand-health":
        base["brand_health"] = _facts_brand_health(main_term=main_term, metrics=metrics)
    elif report_type == "category":
        base["category"] = _facts_category(metrics=metrics, clusters=clusters or [])
    else:
        base["competitive"] = _facts_competitive(main_term=main_term, metrics=metrics)

    return base


def _facts_competitive(*, main_term: str, metrics: MetricsBundle) -> dict:
    df = metrics.weekly
    terms = list(df.columns)

    # Leader share: fraction of weeks each term is the weekly leader
    leader = df.idxmax(axis=1)
    leader_share = (leader.value_counts(normalize=True) * 100.0).to_dict()

    # Momentum: last 13w avg vs prior 13w avg (per term)
    window = 13
    last = df.tail(window)
    prev = df.iloc[-2 * window : -window] if len(df) >= 2 * window else df.head(0)
    momentum: dict[str, dict] = {}
    for t in terms:
        a = float(last[t].mean()) if not last.empty else None
        b = float(prev[t].mean()) if not prev.empty else None
        pct = None
        if a is not None and b is not None and b != 0:
            pct = (a - b) / b * 100.0
        momentum[t] = {"last_13w_avg": a, "prior_13w_avg": b, "pct_change": pct}

    # Head-to-head vs main: average yearly gap-to-leader + correlation with main
    h2h: dict[str, dict] = {}
    for t in terms:
        if t == main_term:
            continue
        corr = float(metrics.corr.loc[main_term, t]) if main_term in metrics.corr.index else None
        # average weekly delta (t - main)
        delta = float((df[t] - df[main_term]).mean()) if main_term in df.columns else None
        h2h[t] = {"corr_with_main": corr, "avg_weekly_delta_vs_main": delta}

    return {
        "leader_share_pct": leader_share,
        "momentum_13w": momentum,
        "head_to_head_vs_main": h2h,
    }


def _facts_brand_health(*, main_term: str, metrics: MetricsBundle) -> dict:
    df = metrics.weekly
    if main_term not in df.columns or df[main_term].dropna().empty:
        return {"note": "Main term has no weekly series in this report."}

    s = df[main_term].astype(float).dropna()

    # Rolling mean + volatility (last 52 weeks)
    last_52 = s.tail(52)
    vol_52 = float(last_52.std(ddof=0)) if len(last_52) >= 5 else None
    avg_52 = float(last_52.mean()) if not last_52.empty else None

    # Momentum: last 13w vs prior 13w
    window = 13
    last = s.tail(window)
    prev = s.iloc[-2 * window : -window] if len(s) >= 2 * window else s.head(0)
    mom_pct = None
    if not last.empty and not prev.empty and float(prev.mean()) != 0:
        mom_pct = (float(last.mean()) - float(prev.mean())) / float(prev.mean()) * 100.0

    # Simple slope (last 52 points) via least squares
    slope = None
    if len(last_52) >= 8:
        x = np.arange(len(last_52), dtype=float)
        y = last_52.values.astype(float)
        x = x - x.mean()
        y = y - y.mean()
        denom = float((x * x).sum())
        if denom != 0:
            slope = float((x * y).sum() / denom)

    # Biggest 4-week move (up/down)
    delta_4w = s.diff(4).dropna()
    biggest_up = None
    biggest_down = None
    if not delta_4w.empty:
        up_i = delta_4w.idxmax()
        dn_i = delta_4w.idxmin()
        biggest_up = {"date": up_i.date().isoformat(), "delta": float(delta_4w.loc[up_i])}
        biggest_down = {"date": dn_i.date().isoformat(), "delta": float(delta_4w.loc[dn_i])}

    return {
        "avg_52w": avg_52,
        "volatility_52w_std": vol_52,
        "slope_last_52w_per_week": slope,
        "momentum_13w_pct": mom_pct,
        "biggest_4w_up": biggest_up,
        "biggest_4w_down": biggest_down,
    }


def parse_clusters(specs: list[str]) -> dict[str, list[str]]:
    """
    Parse `--cluster` inputs in the form: "Name: term1, term2".
    Returns mapping cluster_name -> list of terms.
    """
    clusters: dict[str, list[str]] = {}
    for i, raw in enumerate(specs):
        s = (raw or "").strip()
        if not s:
            continue
        if ":" in s:
            name, rest = s.split(":", 1)
            name = name.strip() or f"Cluster {i+1}"
            terms = [t.strip() for t in rest.split(",") if t.strip()]
        else:
            # If no explicit name provided, treat the whole string as a single-term cluster.
            name = f"Cluster {i+1}"
            terms = [s]
        if not terms:
            continue

        base_name = name
        suffix = 2
        while name in clusters:
            name = f"{base_name} ({suffix})"
            suffix += 1
        clusters[name] = terms
    return clusters


def _facts_category(*, metrics: MetricsBundle, clusters: list[str]) -> dict:
    df = metrics.weekly
    parsed = parse_clusters(clusters)
    if not parsed:
        # Default: treat each term as its own cluster
        parsed = {t: [t] for t in df.columns}

    cluster_weekly: dict[str, pd.Series] = {}
    for cname, members in parsed.items():
        cols = [m for m in members if m in df.columns]
        if not cols:
            continue
        # Average member indices (more stable than sum for Trends indices)
        cluster_weekly[cname] = df[cols].mean(axis=1)

    if not cluster_weekly:
        return {"note": "No clusters matched the provided terms."}

    cdf = pd.DataFrame(cluster_weekly)
    denom = cdf.sum(axis=1).replace(0, np.nan)
    share = cdf.divide(denom, axis=0)

    # Yearly rollups
    yearly_avg = cdf.groupby(cdf.index.year).mean(numeric_only=True)
    yearly_yoy = yearly_avg.pct_change().replace([np.inf, -np.inf], np.nan) * 100.0

    years = [int(y) for y in yearly_avg.index.tolist()]

    return {
        "clusters": parsed,
        "cluster_years": years,
        "cluster_yearly_avg": yearly_avg.round(2).to_dict(orient="index"),
        "cluster_yearly_yoy": yearly_yoy.round(1).to_dict(orient="index"),
        "note": "Cluster indices are computed as the average of member term indices per week (Google Trends indices are relative).",
    }


