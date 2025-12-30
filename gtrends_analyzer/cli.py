from __future__ import annotations

import argparse
from pathlib import Path

from .report import generate_report
from .env import load_dotenv_if_present
from .types import RunArgs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gtrends_analyzer", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Fetch Google Trends data and generate an HTML report.")
    run.add_argument(
        "--report-type",
        default="competitive",
        choices=["competitive", "brand-health", "category"],
        help='Report template: "competitive", "brand-health", or "category".',
    )
    run.add_argument(
        "--bundle",
        action="store_true",
        help="Generate a single HTML file with tabs for all report types (competitive, brand-health, category).",
    )
    run.add_argument("--main", required=True, help="Anchor/main term (used for >5 term rescaling).")
    run.add_argument(
        "--terms",
        nargs="+",
        required=True,
        help="Terms to compare (include the main term if you want it charted).",
    )
    run.add_argument("--geo", default="US", help='Geo code (e.g. "US", "" for worldwide).')
    run.add_argument("--timeframe", default="today 5-y", help='Timeframe (e.g. "today 5-y").')
    run.add_argument("--gprop", default="", help='Google property: "", "images", "news", "youtube", "froogle".')
    # Default to ON (can be disabled). Keep legacy flag names as aliases.
    run.add_argument(
        "--related",
        "--include-related",
        dest="include_related",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include related queries/topics (default: enabled).",
    )
    run.add_argument(
        "--region",
        "--include-region",
        dest="include_region",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include interest-by-region breakdown (default: enabled).",
    )
    run.add_argument(
        "--suggestions",
        dest="include_suggestions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include suggestions/disambiguation for terms (default: enabled).",
    )

    run.add_argument(
        "--cluster",
        action="append",
        default=[],
        help='(category report) Cluster definition, repeatable. Format: "Name: term1, term2, term3".',
    )

    run.add_argument(
        "--cache-dir",
        default=str(Path(".cache") / "gtrends_analyzer"),
        help="Disk cache directory for pytrends responses (default: ./.cache/gtrends_analyzer).",
    )
    run.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=24.0,
        help="Cache TTL in hours (default: 24).",
    )
    run.add_argument("--refresh", action="store_true", help="Bypass cache and fetch fresh data.")
    run.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=15.0,
        help="Minimum delay between pytrends network requests (default: 15s).",
    )
    run.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max attempts per pytrends endpoint when rate-limited (default: 4).",
    )
    run.add_argument(
        "--verbose",
        action="store_true",
        help="Print cache hit/miss and rate-limit/backoff timing to help diagnose fetching behavior.",
    )

    run.add_argument(
        "--include-js",
        default="cdn",
        choices=["cdn", "inline"],
        help='Plotly JS mode: "cdn" (small HTML) or "inline" (offline-capable but larger).',
    )
    run.add_argument("--offline", action="store_true", help="Alias for --include-js inline.")

    run.add_argument(
        "--ai",
        dest="ai_model",
        default="gemini-3-flash-preview",
        help='AI model name. Use "none" to disable.',
    )
    run.add_argument("--no-ai", action="store_true", help="Disable AI commentary (deterministic text only).")

    run.add_argument("--out", default=None, help="Output HTML path. Defaults to reports/<slug>-<ts>.html")
    run.add_argument(
        "--csv",
        default=None,
        help="Optional path to an exported Google Trends multiTimeline CSV (skips live fetching).",
    )
    return p


def _parse_run_args(ns: argparse.Namespace) -> RunArgs:
    ai_model: str | None = ns.ai_model
    if ai_model and ai_model.lower() == "none":
        ai_model = None

    include_js = "inline" if ns.offline else ns.include_js

    out: Path | None = Path(ns.out).expanduser() if ns.out else None
    csv: Path | None = Path(ns.csv).expanduser() if ns.csv else None

    # Data toggles default to ON unless explicitly disabled.
    include_related = True if ns.include_related is None else bool(ns.include_related)
    include_region = True if ns.include_region is None else bool(ns.include_region)
    include_suggestions = True if getattr(ns, "include_suggestions", None) is None else bool(ns.include_suggestions)

    return RunArgs(
        report_type=ns.report_type,
        bundle=bool(ns.bundle),
        main=ns.main,
        terms=list(ns.terms),
        geo=ns.geo,
        timeframe=ns.timeframe,
        gprop=ns.gprop,
        include_related=include_related,
        include_region=include_region,
        include_suggestions=include_suggestions,
        cache_dir=Path(ns.cache_dir).expanduser(),
        cache_ttl_hours=float(ns.cache_ttl_hours),
        refresh=bool(ns.refresh),
        min_request_interval_seconds=float(ns.min_request_interval_seconds),
        max_retries=int(ns.max_retries),
        include_js=include_js,
        offline=bool(ns.offline),
        verbose=bool(getattr(ns, "verbose", False)),
        ai_model=ai_model,
        no_ai=bool(ns.no_ai),
        out=out,
        csv=csv,
        clusters=list(ns.cluster or []),
    )


def main(argv: list[str] | None = None) -> int:
    # Convenience: load local .env if present (keeps secrets out of git).
    load_dotenv_if_present()
    p = _build_parser()
    ns = p.parse_args(argv)

    if ns.cmd == "run":
        args = _parse_run_args(ns)
        generate_report(args)
        return 0

    p.error(f"Unknown command: {ns.cmd}")
    return 2


