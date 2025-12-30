from __future__ import annotations

import time

import pandas as pd

from gtrends_analyzer.cache import DiskCache, RateLimiter, backoff_retry, cache_key


def test_cache_key_stable() -> None:
    a = cache_key("x", {"b": 2, "a": 1})
    b = cache_key("x", {"a": 1, "b": 2})
    assert a == b


def test_disk_cache_df_hit_and_refresh(tmp_path) -> None:
    c = DiskCache(cache_dir=tmp_path, namespace="t")
    k = "k1"
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2]}, index=[0, 1])

    df1 = c.get_or_compute_df(key=k, ttl_seconds=3600, refresh=False, compute_fn=compute)
    df2 = c.get_or_compute_df(key=k, ttl_seconds=3600, refresh=False, compute_fn=compute)
    assert calls["n"] == 1
    assert df1.equals(df2)

    df3 = c.get_or_compute_df(key=k, ttl_seconds=3600, refresh=True, compute_fn=compute)
    assert calls["n"] == 2
    assert df3.equals(df1)


def test_disk_cache_ttl_expiry(tmp_path, monkeypatch) -> None:
    c = DiskCache(cache_dir=tmp_path, namespace="t")
    k = "k2"
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [calls["n"]]})

    # Freeze time for first write
    t0 = time.time()
    monkeypatch.setattr(time, "time", lambda: t0)
    df1 = c.get_or_compute_df(key=k, ttl_seconds=10, refresh=False, compute_fn=compute)
    assert calls["n"] == 1

    # Still fresh
    monkeypatch.setattr(time, "time", lambda: t0 + 5)
    df2 = c.get_or_compute_df(key=k, ttl_seconds=10, refresh=False, compute_fn=compute)
    assert calls["n"] == 1
    assert df1.equals(df2)

    # Expired
    monkeypatch.setattr(time, "time", lambda: t0 + 11)
    df3 = c.get_or_compute_df(key=k, ttl_seconds=10, refresh=False, compute_fn=compute)
    assert calls["n"] == 2
    assert float(df3.iloc[0, 0]) == 2.0


def test_rate_limiter_waits(monkeypatch) -> None:
    rl = RateLimiter(min_interval_seconds=10.0, jitter_ratio=0.0)
    slept: list[float] = []

    now = {"t": 100.0}

    def now_fn() -> float:
        return now["t"]

    def sleep_fn(s: float) -> None:
        slept.append(s)
        now["t"] += s

    rl.wait(now_fn=now_fn, sleep_fn=sleep_fn)
    # first call should not sleep (last=0)
    assert slept == []

    # immediate second call should sleep ~10s
    rl.wait(now_fn=now_fn, sleep_fn=sleep_fn)
    assert slept and slept[-1] == 10.0


def test_backoff_retry_attempts() -> None:
    calls = {"n": 0}

    def fn() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429")
        return 7

    out = backoff_retry(
        fn=fn,
        should_retry=lambda e: "429" in str(e),
        max_attempts=4,
        base_seconds=0.0,
        max_seconds=0.0,
        sleep_fn=lambda s: None,
    )
    assert out == 7
    assert calls["n"] == 3


