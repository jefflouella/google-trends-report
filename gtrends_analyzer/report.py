from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from .types import RunArgs
from .fetch import fetch_trends
from .metrics import build_facts_packet, compute_metrics, parse_clusters
from .ai import deterministic_commentary_sections, generate_commentary_sections
from .io import load_multitimeline_csv


def generate_report(args: RunArgs) -> Path:
    """
    Orchestrates fetching, metrics, optional AI commentary, and HTML report rendering.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(args.main)[:60] or "report"
        out_path = reports_dir / f"{slug}-{ts}.html"

    if args.csv:
        weekly_df = load_multitimeline_csv(args.csv)
        trends = type(
            "TrendsLike",
            (),
            {
                "interest_over_time": weekly_df,
                "interest_by_region": None,
                "related_queries": None,
                "related_topics": None,
                "suggestions": None,
            },
        )()
    else:
        trends = fetch_trends(
            main_term=args.main,
            terms=args.terms,
            geo=args.geo,
            timeframe=args.timeframe,
            gprop=args.gprop,
            include_region=args.include_region,
            include_related=args.include_related,
            include_suggestions=args.include_suggestions,
            cache_dir=args.cache_dir,
            cache_ttl_seconds=float(args.cache_ttl_hours) * 3600.0,
            refresh=bool(args.refresh),
            min_request_interval_seconds=float(args.min_request_interval_seconds),
            max_retries=int(args.max_retries),
            verbose=bool(getattr(args, "verbose", False)),
        )

    metrics = compute_metrics(trends.interest_over_time)

    if args.bundle:
        html = _generate_bundled_html(args=args, trends=trends, metrics=metrics)
    else:
        html = _generate_single_html(args=args, trends=trends, metrics=metrics)

    out_path.write_text(html, encoding="utf-8", errors="strict")
    return out_path


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _report_heading(args: RunArgs) -> str:
    if args.bundle:
        return "Google Trends Report Bundle"
    if args.report_type == "brand-health":
        return "Google Trends Brand Health"
    if args.report_type == "category":
        return "Google Trends Category Demand"
    return "Google Trends Competitive Analysis"

def _with_report_type(args: RunArgs, report_type: str) -> RunArgs:
    # RunArgs is frozen, so create a new instance with the desired report_type.
    return RunArgs(
        report_type=report_type,
        bundle=args.bundle,
        main=args.main,
        terms=args.terms,
        geo=args.geo,
        timeframe=args.timeframe,
        gprop=args.gprop,
        include_related=args.include_related,
        include_region=args.include_region,
        include_suggestions=args.include_suggestions,
        cache_dir=args.cache_dir,
        cache_ttl_hours=args.cache_ttl_hours,
        refresh=args.refresh,
        min_request_interval_seconds=args.min_request_interval_seconds,
        max_retries=args.max_retries,
        include_js=args.include_js,
        offline=args.offline,
        verbose=bool(getattr(args, "verbose", False)),
        ai_model=args.ai_model,
        no_ai=args.no_ai,
        out=args.out,
        csv=args.csv,
        clusters=args.clusters,
    )


def _generate_single_html(*, args: RunArgs, trends, metrics) -> str:
    facts = build_facts_packet(
        report_type=args.report_type,
        main_term=args.main,
        metrics=metrics,
        geo=args.geo,
        timeframe=args.timeframe,
        gprop=args.gprop,
        clusters=args.clusters,
    )
    facts = _attach_facts_extras(args=args, facts=facts, trends=trends)

    if args.no_ai or not args.ai_model:
        commentary = deterministic_commentary_sections(facts=facts, note="AI disabled via --no-ai.")
        ai_label = "disabled"
    else:
        commentary = generate_commentary_sections(model=args.ai_model, facts=facts)
        ai_label = args.ai_model

    sections = _build_sections(args=args, metrics=metrics, commentary=commentary, trends=trends)
    return _render_html(
        title=f"Google Trends Report — {args.main}",
        heading=_report_heading(args),
        subtitle="Interactive report generated from Google Trends data.",
        params={
            "report_type": args.report_type,
            "main": args.main,
            "geo": args.geo,
            "timeframe": args.timeframe,
            "gprop": args.gprop,
            "ai": ai_label,
        },
        sections=sections,
        tabs=None,
        include_js=args.include_js,
    )


def _generate_bundled_html(*, args: RunArgs, trends, metrics) -> str:
    report_types = ["competitive", "brand-health", "category"]
    tabs: list[dict] = []

    # Share plotly inline JS across all tabs if --include-js inline.
    plotly_inline_state = {"used": False}

    for rt in report_types:
        args_rt = _with_report_type(args, rt)
        facts = build_facts_packet(
            report_type=args_rt.report_type,
            main_term=args_rt.main,
            metrics=metrics,
            geo=args_rt.geo,
            timeframe=args_rt.timeframe,
            gprop=args_rt.gprop,
            clusters=args_rt.clusters,
        )
        facts = _attach_facts_extras(args=args_rt, facts=facts, trends=trends)

        if args_rt.no_ai or not args_rt.ai_model:
            commentary = deterministic_commentary_sections(facts=facts, note="AI disabled via --no-ai.")
            ai_label = "disabled"
        else:
            commentary = generate_commentary_sections(model=args_rt.ai_model, facts=facts)
            ai_label = args_rt.ai_model

        sections = _build_sections(
            args=args_rt,
            metrics=metrics,
            commentary=commentary,
            trends=trends,
            plotly_inline_state=plotly_inline_state,
        )
        tabs.append({"id": rt, "title": _title_for_report_type(rt), "sections": sections})

    return _render_html(
        title=f"Google Trends Report Bundle — {args.main}",
        heading=_report_heading(args),
        subtitle="Interactive report generated from Google Trends data.",
        params={
            "report_type": "bundle",
            "main": args.main,
            "geo": args.geo,
            "timeframe": args.timeframe,
            "gprop": args.gprop,
            "ai": ai_label,
        },
        sections=None,
        tabs=tabs,
        include_js=args.include_js,
    )


def _title_for_report_type(report_type: str) -> str:
    if report_type == "brand-health":
        return "Brand Health"
    if report_type == "category":
        return "Category"
    return "Competitive"

def _render_html(
    *,
    title: str,
    heading: str,
    subtitle: str,
    params: dict,
    sections: list[dict] | None,
    tabs: list[dict] | None,
    include_js: str,
) -> str:
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
    )
    template = env.get_template("report.html.j2")

    plotly_js_tag = ""
    if include_js == "cdn":
        plotly_js_tag = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    return template.render(
        title=title,
        heading=heading,
        subtitle=subtitle,
        params=params,
        sections=sections or [],
        tabs=tabs or [],
        plotly_js_tag=plotly_js_tag,
    )


def _build_sections(*, args: RunArgs, metrics, commentary, trends, plotly_inline_state: dict | None = None) -> list[dict]:
    # Create a lookup for commentary by id
    comm_by_id = {c.section_id: c for c in commentary}

    include_plotly_inline_once = args.include_js == "inline"
    if plotly_inline_state is None:
        plotly_inline_state = {"used": False}

    def fig_html(fig) -> str:
        include_plotlyjs = False
        if include_plotly_inline_once and not bool(plotly_inline_state.get("used")):
            include_plotlyjs = "inline"
            plotly_inline_state["used"] = True
        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    sections: list[dict] = []

    def add(section_id: str, title: str, html: str) -> None:
        c = comm_by_id.get(section_id)
        sections.append(
            {
                "title": title,
                "commentary": c.html if c else "",
                "html": html,
            }
        )

    def add_exec() -> None:
        add("executive_summary", "Executive Summary", "")

    def add_suggestions_if_any() -> None:
        if not trends.suggestions:
            return
        add(
            "suggestions",
            "Term Disambiguation (Suggestions)",
            (
                "<p><b>How to read this:</b> Google Trends suggestions help disambiguate terms (brands vs generic words). "
                "The <code>mid</code> field is a stable entity/topic identifier when available.</p>"
                + _suggestions_html(trends.suggestions)
            ),
        )

    def add_related_if_enabled() -> None:
        # Related queries/topics are not available when running from an exported multiTimeline CSV.
        # Avoid showing a confusing empty section in CSV mode.
        if args.csv:
            return
        if args.include_related or trends.related_queries or trends.related_topics:
            add(
                "related",
                "Related Queries & Topics (Main Term)",
                (
                    "<p><b>How to read this:</b> These are Google Trends “related queries/topics” for the main term. "
                    "They can help identify adjacent brand/keyword opportunities and shifting user intent. "
                    "Use <code>Rising</code> for momentum and <code>Top</code> for steady demand.</p>"
                    + _related_html(trends.related_queries, trends.related_topics, enabled=args.include_related)
                ),
            )

    def add_region_if_any() -> None:
        if trends.interest_by_region is None or trends.interest_by_region.empty:
            return
        add(
            "region",
            "Interest by Region",
            (
                "<p><b>How to read this:</b> Bars show where the selected term has the highest relative interest. "
                "This is based on Google Trends’ regional index and is best used to compare relative geographic concentration.</p>"
                + _region_html(trends.interest_by_region, main_term=args.main)
            ),
        )

    weekly = metrics.weekly.reset_index(names="date")
    terms = [c for c in metrics.weekly.columns]

    # --- Template: Competitive (default) ---
    if args.report_type == "competitive":
        add_exec()

        fig1 = px.line(weekly, x="date", y=terms, title="Interest over time (weekly)")
        fig1.update_layout(legend_title_text="Term")
        add(
            "rankings",
            "Competitive Trends (Weekly)",
            (
                "<p><b>How to read this:</b> Each line is a term’s weekly Google Trends index (0–100) within the selected timeframe and query set. "
                "Higher = more relative search interest that week. Compare lines to see who leads and when gaps widen/narrow.</p>"
                f"{fig_html(fig1)}"
            ),
        )

        yearly = metrics.yearly_avg.copy()
        yearly.index = yearly.index.astype(int)
        yearly_tbl = _df_to_html(yearly.round(2), index_label="Year")

        yoy = metrics.yearly_yoy.copy()
        yoy.index = yoy.index.astype(int)
        yoy_tbl = _df_to_html(yoy.round(1), index_label="Year")

        reb = metrics.rebased_yearly.copy()
        reb.index = reb.index.astype(int)
        fig2 = px.line(reb.reset_index(names="year"), x="year", y=terms, title="Rebased yearly index (start year = 100)")
        fig2.update_layout(legend_title_text="Term")

        add(
            "yearly",
            "Year-by-Year Dynamics",
            (
                "<p><b>How to read the tables:</b> <b>Yearly averages</b> are the mean weekly index for that year. "
                "<b>YoY change</b> is the percent change in yearly average vs the prior year.</p>"
                f"<h3>Yearly averages</h3>{yearly_tbl}"
                f"<h3>YoY change (%)</h3>{yoy_tbl}"
                "<p><b>How to read the rebased chart:</b> Each term is normalized to its own <b>first full year</b> in the dataset = 100. "
                "A value of 50 means that term’s yearly average is ~50% of its baseline; 200 means ~2× its baseline. "
                "This shows each term’s internal rise/decline independent of absolute size.</p>"
                f"{fig_html(fig2)}"
            ),
        )

        moy = metrics.month_of_year_avg.copy()
        moy.index = moy.index.astype(int)
        fig3 = px.line(moy.reset_index(names="month"), x="month", y=terms, title="Seasonality (month-of-year average)")
        fig3.update_xaxes(dtick=1)
        add(
            "seasonality",
            "Seasonality",
            (
                "<p><b>How to read this:</b> X-axis is month (1–12). Lines show the average index for that month across all years. "
                "Use this to spot consistent seasonal peaks and compare seasonality patterns across terms.</p>"
                f"{fig_html(fig3)}"
            ),
        )

        share = metrics.share_of_search.copy().fillna(0)
        share_long = share.reset_index(names="date").melt(id_vars=["date"], var_name="term", value_name="share")
        fig4 = px.area(share_long, x="date", y="share", color="term", title="Share of search (proxy, within compared set)")
        fig4.update_layout(yaxis_tickformat=".0%")
        add(
            "share",
            "Share of Search (Proxy)",
            (
                "<p><b>How to read this:</b> Each week, we compute each term’s share as "
                "<code>term / sum(all terms)</code> within this comparison set. "
                "This is a relative ‘share of attention’ proxy (not true market share). The stacked area sums to 100% each week.</p>"
                f"{fig_html(fig4)}"
            ),
        )

        corr = metrics.corr.copy()
        fig5 = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                zmin=-1,
                zmax=1,
                colorbar=dict(title="corr"),
            )
        )
        fig5.update_layout(title="Term similarity (correlation of weekly series)")
        add(
            "similarity",
            "Similarity",
            (
                "<p><b>How to read this:</b> Correlation measures how similarly two series move over time "
                "(+1 = move together, 0 = unrelated, -1 = move opposite).</p>"
                f"{fig_html(fig5)}"
            ),
        )

        add_suggestions_if_any()
        add_related_if_enabled()
        add_region_if_any()

        return sections

    # --- Template: Brand health ---
    if args.report_type == "brand-health":
        add_exec()

        if args.main in metrics.weekly.columns:
            s = metrics.weekly[[args.main]].reset_index(names="date")
            fig_h1 = px.line(s, x="date", y=args.main, title=f"Interest over time — {args.main} (weekly)")
            fig_h1.update_layout(showlegend=False)

            # Add 12-week rolling mean
            roll = metrics.weekly[args.main].rolling(window=12, min_periods=2).mean()
            roll_df = pd.DataFrame({"date": metrics.weekly.index, "12w_avg": roll.values})
            fig_h2 = px.line(roll_df, x="date", y="12w_avg", title=f"Smoothed trend — {args.main} (12-week average)")
            fig_h2.update_layout(showlegend=False)

            add(
                "brand_health",
                "Brand Health Overview",
                (
                    "<p><b>How to read this:</b> The first chart shows the raw weekly index. The second chart smooths noise with a 12-week average "
                    "to make underlying trend direction clearer.</p>"
                    f"{fig_html(fig_h1)}"
                    f"{fig_html(fig_h2)}"
                ),
            )

            # Drawdown chart
            s0 = metrics.weekly[args.main].astype(float)
            roll_max = s0.rolling(window=12, min_periods=2).max()
            dd = (s0 / roll_max - 1.0) * 100.0
            dd_df = pd.DataFrame({"date": metrics.weekly.index, "drawdown_pct": dd.values})
            fig_dd = px.area(dd_df, x="date", y="drawdown_pct", title=f"Short-term drawdowns — {args.main} (12-week)")
            fig_dd.update_layout(yaxis_title="% from rolling peak")
            add(
                "drawdowns",
                "Drawdowns",
                (
                    "<p><b>How to read this:</b> Drawdown measures how far the series is below its recent rolling peak. "
                    "More negative values indicate sharper pullbacks from a local high.</p>"
                    f"{fig_html(fig_dd)}"
                ),
            )

            # Seasonality for main term
            moy = metrics.month_of_year_avg[[args.main]].copy()
            moy.index = moy.index.astype(int)
            fig_seas = px.line(moy.reset_index(names="month"), x="month", y=args.main, title=f"Seasonality — {args.main} (month-of-year average)")
            fig_seas.update_xaxes(dtick=1)
            fig_seas.update_layout(showlegend=False)
            add(
                "seasonality",
                "Seasonality",
                (
                    "<p><b>How to read this:</b> This is the average index for each month across all years, which highlights recurring seasonal peaks.</p>"
                    f"{fig_html(fig_seas)}"
                ),
            )

        add_suggestions_if_any()
        add_related_if_enabled()
        add_region_if_any()
        return sections

    # --- Template: Category / clusters ---
    if args.report_type == "category":
        add_exec()

        clusters = parse_clusters(args.clusters)
        if not clusters:
            clusters = {t: [t] for t in metrics.weekly.columns}

        add(
            "clusters",
            "Cluster Definitions",
            (
                "<p><b>How to read this:</b> Clusters group multiple terms into a single category signal. "
                "Cluster indices are computed as the average of member term indices per week.</p>"
                + _clusters_html(clusters)
            ),
        )

        # Build cluster weekly series (avg of member terms)
        c_series: dict[str, pd.Series] = {}
        for cname, members in clusters.items():
            cols = [m for m in members if m in metrics.weekly.columns]
            if not cols:
                continue
            c_series[cname] = metrics.weekly[cols].mean(axis=1)
        if c_series:
            cdf = pd.DataFrame(c_series)
            c_weekly = cdf.reset_index(names="date")
            fig_c1 = px.line(c_weekly, x="date", y=list(cdf.columns), title="Cluster interest over time (weekly)")
            fig_c1.update_layout(legend_title_text="Cluster")
            add(
                "cluster_trends",
                "Cluster Trends (Weekly)",
                (
                    "<p><b>How to read this:</b> Each line is a cluster index (average of its member terms’ indices). "
                    "Higher values indicate more relative interest within this report’s timeframe/query set.</p>"
                    f"{fig_html(fig_c1)}"
                ),
            )

            denom = cdf.sum(axis=1).replace(0, np.nan)
            cshare = cdf.divide(denom, axis=0).fillna(0)
            cshare_long = cshare.reset_index(names="date").melt(id_vars=["date"], var_name="cluster", value_name="share")
            fig_c2 = px.area(cshare_long, x="date", y="share", color="cluster", title="Cluster share of search (proxy)")
            fig_c2.update_layout(yaxis_tickformat=".0%")
            add(
                "cluster_share",
                "Cluster Share (Proxy)",
                (
                    "<p><b>How to read this:</b> Each week, cluster share is computed as "
                    "<code>cluster_index / sum(cluster_indices)</code>. The stacked area sums to 100% each week.</p>"
                    f"{fig_html(fig_c2)}"
                ),
            )

            # Yearly rollups
            yearly = cdf.groupby(cdf.index.year).mean(numeric_only=True)
            yearly.index = yearly.index.astype(int)
            yoy = yearly.pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
            yoy.index = yoy.index.astype(int)
            add(
                "cluster_yearly",
                "Cluster Year-by-Year",
                (
                    "<p><b>How to read this:</b> Yearly averages are the mean weekly cluster index for that year; YoY is the percent change vs prior year.</p>"
                    f"<h3>Yearly averages</h3>{_df_to_html(yearly.round(2), index_label='Year')}"
                    f"<h3>YoY change (%)</h3>{_df_to_html(yoy.round(1), index_label='Year')}"
                ),
            )

        add_suggestions_if_any()
        add_related_if_enabled()
        add_region_if_any()
        return sections

    return sections


def _df_to_html(df: pd.DataFrame, *, index_label: str | None = None) -> str:
    out = df.copy()
    if index_label:
        out.index.name = index_label
    return out.to_html(border=0, classes="dataframe", na_rep="")


def _related_html(related_queries: dict | None, related_topics: dict | None, *, enabled: bool) -> str:
    if not enabled:
        return "<p class=\"muted\">(Not included. Re-run with <code>--include-related</code>.)</p>"
    if not related_queries and not related_topics:
        return "<p class=\"muted\">(Enabled, but Google Trends returned no related data.)</p>"

    parts: list[str] = []
    if related_queries:
        parts.append("<h3>Related queries</h3>")
        for term, payload in related_queries.items():
            parts.append(f"<h4>{term}</h4>")
            top = payload.get("top") if isinstance(payload, dict) else None
            rising = payload.get("rising") if isinstance(payload, dict) else None
            if isinstance(top, pd.DataFrame) and not top.empty:
                parts.append("<b>Top</b>")
                parts.append(_df_to_html(top.head(10)))
            if isinstance(rising, pd.DataFrame) and not rising.empty:
                parts.append("<b>Rising</b>")
                parts.append(_df_to_html(rising.head(10)))
    if related_topics:
        parts.append("<h3>Related topics</h3>")
        for term, payload in related_topics.items():
            parts.append(f"<h4>{term}</h4>")
            top = payload.get("top") if isinstance(payload, dict) else None
            rising = payload.get("rising") if isinstance(payload, dict) else None
            if isinstance(top, pd.DataFrame) and not top.empty:
                parts.append("<b>Top</b>")
                parts.append(_df_to_html(top.head(10)))
            if isinstance(rising, pd.DataFrame) and not rising.empty:
                parts.append("<b>Rising</b>")
                parts.append(_df_to_html(rising.head(10)))

    return "\n".join(parts) if parts else "<p class=\"muted\">(No related data returned.)</p>"


def _region_html(df: pd.DataFrame, *, main_term: str) -> str:
    cols = list(df.columns)
    col = main_term if main_term in cols else cols[0]
    top = df[[col]].sort_values(col, ascending=False).head(15)
    fig = px.bar(top.reset_index(names="region"), x=col, y="region", orientation="h", title=f"Top regions — {col}")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    # Inline is handled by to_html include_plotlyjs in the caller; here we assume CDN tag exists.
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _suggestions_html(suggestions: dict[str, list[dict]]) -> str:
    rows: list[dict] = []
    for term, items in (suggestions or {}).items():
        if not items:
            rows.append({"term": term, "title": "", "type": "", "mid": ""})
            continue
        for item in items[:5]:
            rows.append(
                {
                    "term": term,
                    "title": str(item.get("title", "")),
                    "type": str(item.get("type", "")),
                    "mid": str(item.get("mid", "")),
                }
            )
    df = pd.DataFrame(rows, columns=["term", "title", "type", "mid"])
    return _df_to_html(df, index_label=None)


def _clusters_html(clusters: dict[str, list[str]]) -> str:
    rows: list[dict] = []
    for name, members in clusters.items():
        rows.append({"cluster": name, "terms": ", ".join(members)})
    df = pd.DataFrame(rows, columns=["cluster", "terms"])
    return _df_to_html(df, index_label=None)


def _attach_facts_extras(*, args: RunArgs, facts: dict, trends) -> dict:
    """
    Add non-metric, fetch-derived facts to support grounded AI commentary:
    - availability flags
    - suggestions summaries
    - related queries/topics excerpts (main term)
    - region top rows (main term)
    """
    out = dict(facts)
    out["data_availability"] = {
        "include_related": bool(args.include_related),
        "include_region": bool(args.include_region),
        "include_suggestions": bool(args.include_suggestions),
        "has_related_queries": bool(getattr(trends, "related_queries", None)),
        "has_related_topics": bool(getattr(trends, "related_topics", None)),
        "has_region": bool(getattr(trends, "interest_by_region", None) is not None and not getattr(trends, "interest_by_region").empty),
        "has_suggestions": bool(getattr(trends, "suggestions", None)),
    }

    # Suggestions summary (counts + top suggestion)
    sugg = getattr(trends, "suggestions", None) or {}
    if sugg:
        summary: dict[str, dict] = {}
        for term, items in sugg.items():
            top = items[0] if isinstance(items, list) and items else None
            summary[term] = {
                "count": len(items) if isinstance(items, list) else 0,
                "top": {
                    "title": str(top.get("title", "")) if isinstance(top, dict) else "",
                    "type": str(top.get("type", "")) if isinstance(top, dict) else "",
                    "mid": str(top.get("mid", "")) if isinstance(top, dict) else "",
                }
                if top
                else None,
            }
        out["suggestions_summary"] = summary

    # Related excerpts (main term)
    related_queries = getattr(trends, "related_queries", None) or {}
    related_topics = getattr(trends, "related_topics", None) or {}

    def _top_rows(payload: dict | None, *, key: str) -> list[dict]:
        if not isinstance(payload, dict):
            return []
        df = payload.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        cols = [c for c in ["query", "topic_title", "value"] if c in df.columns]
        use = df[cols].head(10).copy()
        return use.to_dict(orient="records")

    if related_queries or related_topics:
        rq = related_queries.get(args.main) if isinstance(related_queries, dict) else None
        rt = related_topics.get(args.main) if isinstance(related_topics, dict) else None
        out["related_excerpts"] = {
            "main_term": args.main,
            "queries_top": _top_rows(rq, key="top"),
            "queries_rising": _top_rows(rq, key="rising"),
            "topics_top": _top_rows(rt, key="top"),
            "topics_rising": _top_rows(rt, key="rising"),
        }

    # Region excerpt
    region_df = getattr(trends, "interest_by_region", None)
    if isinstance(region_df, pd.DataFrame) and not region_df.empty:
        cols = list(region_df.columns)
        col = args.main if args.main in cols else cols[0]
        top = region_df[[col]].sort_values(col, ascending=False).head(10).reset_index(names="region")
        out["region_top"] = top.to_dict(orient="records")

    return out


