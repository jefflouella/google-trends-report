from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pathlib import Path
from typing import Callable
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq
from requests import exceptions as requests_exceptions  # type: ignore
from urllib3 import exceptions as urllib3_exceptions  # type: ignore

from .cache import DiskCache, RateLimiter, backoff_retry, cache_key


@dataclass(frozen=True)
class TrendsResult:
    interest_over_time: pd.DataFrame
    interest_by_region: pd.DataFrame | None
    related_queries: dict | None
    related_topics: dict | None
    suggestions: dict[str, list[dict]] | None


def fetch_trends(
    *,
    main_term: str,
    terms: list[str],
    geo: str,
    timeframe: str,
    gprop: str,
    include_region: bool,
    include_related: bool,
    include_suggestions: bool,
    cache_dir: str | None = None,
    cache_ttl_seconds: float = 24 * 3600,
    refresh: bool = False,
    min_request_interval_seconds: float = 15.0,
    max_retries: int = 4,
    verbose: bool = False,
) -> TrendsResult:
    terms = _normalize_terms(main_term=main_term, terms=terms)
    log_fn = print if verbose else None
    cache = DiskCache(
        cache_dir=cache_dir or (Path(".cache") / "gtrends_analyzer"),
        namespace="pytrends",
        log_fn=log_fn,
    )
    limiter = RateLimiter(min_interval_seconds=min_request_interval_seconds, log_fn=log_fn)
    pytrends = TrendReq(hl="en-US", tz=300, retries=2, backoff_factor=0.2, timeout=(10, 30))

    batches = _build_batches(main_term=main_term, terms=terms, max_terms_per_query=5)
    base_batch = batches[0]
    base_df = _fetch_interest_over_time(
        pytrends,
        base_batch,
        geo=geo,
        timeframe=timeframe,
        gprop=gprop,
        cache=cache,
        ttl_seconds=cache_ttl_seconds,
        refresh=refresh,
        limiter=limiter,
        max_retries=max_retries,
        log_fn=log_fn,
    )

    combined = base_df.copy()
    if len(batches) > 1:
        base_anchor = combined[main_term].copy()
        for batch in batches[1:]:
            batch_df = _fetch_interest_over_time(
                pytrends,
                batch,
                geo=geo,
                timeframe=timeframe,
                gprop=gprop,
                cache=cache,
                ttl_seconds=cache_ttl_seconds,
                refresh=refresh,
                limiter=limiter,
                max_retries=max_retries,
                log_fn=log_fn,
            )
            scale = _compute_anchor_scale_factor(
                base_anchor=base_anchor,
                batch_anchor=batch_df[main_term],
            )
            scaled = batch_df.copy()
            for col in scaled.columns:
                if col == main_term:
                    continue
                scaled[col] = scaled[col].astype(float) * scale

            # merge in any new columns
            for col in scaled.columns:
                if col == main_term:
                    continue
                if col not in combined.columns:
                    combined[col] = scaled[col]

    # Optional extras: use first batch (keeps requests lower)
    interest_by_region: pd.DataFrame | None = None
    related_queries: dict | None = None
    related_topics: dict | None = None
    suggestions: dict[str, list[dict]] | None = None

    if include_region:
        interest_by_region = _fetch_interest_by_region(
            pytrends,
            base_batch,
            geo=geo,
            timeframe=timeframe,
            gprop=gprop,
            cache=cache,
            ttl_seconds=cache_ttl_seconds,
            refresh=refresh,
            limiter=limiter,
            max_retries=max_retries,
            log_fn=log_fn,
        )
    if include_related:
        related_queries, related_topics = _fetch_related(
            pytrends,
            [main_term],
            geo=geo,
            timeframe=timeframe,
            gprop=gprop,
            cache=cache,
            ttl_seconds=cache_ttl_seconds,
            refresh=refresh,
            limiter=limiter,
            max_retries=max_retries,
            log_fn=log_fn,
        )
    if include_suggestions:
        suggestions = _fetch_suggestions(
            pytrends,
            terms=terms,
            cache=cache,
            ttl_seconds=cache_ttl_seconds,
            refresh=refresh,
            limiter=limiter,
            max_retries=max_retries,
            log_fn=log_fn,
        )

    return TrendsResult(
        interest_over_time=combined,
        interest_by_region=interest_by_region,
        related_queries=related_queries,
        related_topics=related_topics,
        suggestions=suggestions,
    )


def _normalize_terms(*, main_term: str, terms: list[str]) -> list[str]:
    # preserve order while de-duping
    seen: set[str] = set()
    out: list[str] = []

    def add(t: str) -> None:
        t = (t or "").strip()
        if not t:
            return
        if t in seen:
            return
        seen.add(t)
        out.append(t)

    add(main_term)
    for t in terms:
        add(t)
    return out


def _build_batches(*, main_term: str, terms: list[str], max_terms_per_query: int) -> list[list[str]]:
    if max_terms_per_query < 2:
        raise ValueError("max_terms_per_query must be >= 2")

    if len(terms) <= max_terms_per_query:
        return [terms]

    others = [t for t in terms if t != main_term]
    chunk_size = max_terms_per_query - 1
    batches: list[list[str]] = []
    for i in range(0, len(others), chunk_size):
        batches.append([main_term, *others[i : i + chunk_size]])
    return batches


def _fetch_interest_over_time(
    pytrends: TrendReq,
    terms: list[str],
    *,
    geo: str,
    timeframe: str,
    gprop: str,
    cache: DiskCache,
    ttl_seconds: float,
    refresh: bool,
    limiter: RateLimiter,
    max_retries: int,
    log_fn: Callable[[str], None] | None,
) -> pd.DataFrame:
    k = cache_key("interest_over_time", {"terms": terms, "geo": geo, "timeframe": timeframe, "gprop": gprop})

    def compute() -> pd.DataFrame:
        limiter.wait()

        def call() -> pd.DataFrame:
            pytrends.build_payload(terms, cat=0, timeframe=timeframe, geo=geo, gprop=gprop)
            return pytrends.interest_over_time()

        try:
            df = backoff_retry(
                fn=call,
                should_retry=_is_rate_limit_exception,
                max_attempts=max_retries,
                log_fn=log_fn,
            )
        except ResponseError as e:
            raise RuntimeError(
                "Google Trends request failed (rate limit or backend change). Try again later or reduce scope."
            ) from e
        return df

    df = cache.get_or_compute_df(
        key=k,
        ttl_seconds=ttl_seconds,
        refresh=refresh,
        compute_fn=compute,
        meta={"endpoint": "interest_over_time"},
    )

    if df is None or df.empty:
        raise RuntimeError("Google Trends returned no data for the given query.")

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    return df.astype(float)


def _compute_anchor_scale_factor(*, base_anchor: pd.Series, batch_anchor: pd.Series) -> float:
    joined = pd.concat([base_anchor.rename("base"), batch_anchor.rename("batch")], axis=1).dropna()
    joined = joined[(joined["base"] > 0) & (joined["batch"] > 0)]
    if joined.empty:
        return 1.0

    ratios = (joined["base"] / joined["batch"]).replace([pd.NA, pd.NaT], pd.NA).dropna()
    if ratios.empty:
        return 1.0

    # Median is robust to spikes.
    scale = float(ratios.median())
    if not (scale > 0):
        return 1.0
    return scale


def _fetch_interest_by_region(
    pytrends: TrendReq,
    terms: list[str],
    *,
    geo: str,
    timeframe: str,
    gprop: str,
    cache: DiskCache,
    ttl_seconds: float,
    refresh: bool,
    limiter: RateLimiter,
    max_retries: int,
    log_fn: Callable[[str], None] | None,
) -> pd.DataFrame:
    resolution = "COUNTRY" if geo == "" else "REGION"
    k = cache_key("interest_by_region", {"terms": terms, "geo": geo, "timeframe": timeframe, "gprop": gprop, "resolution": resolution})

    def compute() -> pd.DataFrame:
        limiter.wait()

        def call() -> pd.DataFrame:
            pytrends.build_payload(terms, cat=0, timeframe=timeframe, geo=geo, gprop=gprop)
            return pytrends.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=False)

        try:
            df = backoff_retry(
                fn=call,
                should_retry=_is_rate_limit_exception,
                max_attempts=max_retries,
                log_fn=log_fn,
            )
        except Exception:
            df = pd.DataFrame()
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    df = cache.get_or_compute_df(
        key=k,
        ttl_seconds=ttl_seconds,
        refresh=refresh,
        compute_fn=compute,
        meta={"endpoint": "interest_by_region"},
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return df.astype(float)


def _fetch_related(
    pytrends: TrendReq,
    terms: list[str],
    *,
    geo: str,
    timeframe: str,
    gprop: str,
    cache: DiskCache,
    ttl_seconds: float,
    refresh: bool,
    limiter: RateLimiter,
    max_retries: int,
    log_fn: Callable[[str], None] | None,
) -> tuple[dict, dict]:
    kq = cache_key("related_queries", {"terms": terms, "geo": geo, "timeframe": timeframe, "gprop": gprop})
    kt = cache_key("related_topics", {"terms": terms, "geo": geo, "timeframe": timeframe, "gprop": gprop})

    def compute_queries() -> dict:
        limiter.wait()

        def call() -> dict:
            pytrends.build_payload(terms, cat=0, timeframe=timeframe, geo=geo, gprop=gprop)
            return pytrends.related_queries() or {}

        try:
            return backoff_retry(
                fn=call,
                should_retry=_is_rate_limit_exception,
                max_attempts=max_retries,
                log_fn=log_fn,
            )
        except Exception:
            return {}

    def compute_topics() -> dict:
        limiter.wait()

        def call() -> dict:
            pytrends.build_payload(terms, cat=0, timeframe=timeframe, geo=geo, gprop=gprop)
            return pytrends.related_topics() or {}

        try:
            return backoff_retry(
                fn=call,
                should_retry=_is_rate_limit_exception,
                max_attempts=max_retries,
                log_fn=log_fn,
            )
        except Exception:
            return {}

    related_q = cache.get_or_compute_df_dict(
        key=kq,
        ttl_seconds=ttl_seconds,
        refresh=refresh,
        compute_fn=compute_queries,
        meta={"endpoint": "related_queries"},
    )
    related_t = cache.get_or_compute_df_dict(
        key=kt,
        ttl_seconds=ttl_seconds,
        refresh=refresh,
        compute_fn=compute_topics,
        meta={"endpoint": "related_topics"},
    )
    return related_q, related_t


def _fetch_suggestions(
    pytrends: TrendReq,
    *,
    terms: list[str],
    cache: DiskCache,
    ttl_seconds: float,
    refresh: bool,
    limiter: RateLimiter,
    max_retries: int,
    log_fn: Callable[[str], None] | None,
) -> dict[str, list[dict]]:
    """
    Fetch Google Trends suggestions (disambiguation/entity hints) for each term.
    Returns mapping term -> list of suggestion dicts (as returned by pytrends).
    """
    out: dict[str, list[dict]] = {}
    for t in terms:
        k = cache_key("suggestions", {"term": t})

        def compute() -> list[dict]:
            limiter.wait()

            def call() -> list[dict]:
                return pytrends.suggestions(t) or []

            try:
                return backoff_retry(
                    fn=call,
                    should_retry=_is_rate_limit_exception,
                    max_attempts=max_retries,
                    log_fn=log_fn,
                )
            except Exception:
                return []

        sugg = cache.get_or_compute_json(
            key=k,
            ttl_seconds=ttl_seconds,
            refresh=refresh,
            compute_fn=compute,
            meta={"endpoint": "suggestions"},
        ) or []
        # Keep only JSON-serializable dicts; truncate to a small, useful set.
        cleaned: list[dict] = []
        for item in sugg:
            if isinstance(item, dict):
                cleaned.append(dict(item))
            if len(cleaned) >= 8:
                break
        out[t] = cleaned
    return out


def _is_rate_limit_exception(e: Exception) -> bool:
    # pytrends may raise ResponseError, and requests/urllib3 may raise retry errors on 429/sorry pages.
    if isinstance(e, ResponseError):
        return True
    if isinstance(e, requests_exceptions.RetryError):
        return True
    if isinstance(e, urllib3_exceptions.MaxRetryError):
        return True
    msg = (str(e) or "").lower()
    return ("429" in msg) or ("too many" in msg) or ("sorry" in msg)


