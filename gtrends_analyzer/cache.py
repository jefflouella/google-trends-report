from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def cache_key(*parts: Any) -> str:
    """
    Stable cache key based on JSON serialization of parts.
    Returns a short hex digest suitable for filenames.
    """
    payload = _stable_json(parts).encode("utf-8", errors="strict")
    return hashlib.sha256(payload).hexdigest()[:24]


@dataclass(frozen=True)
class CacheEntry:
    key: str
    created_at: float
    ttl_seconds: float
    meta: dict[str, Any]

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    def is_fresh(self, now: float | None = None) -> bool:
        n = time.time() if now is None else now
        return n <= self.expires_at


class DiskCache:
    """
    Simple file-based cache with TTL.

    Layout:
      <cache_dir>/<namespace>/<key>/
        meta.json
        payload.(json|csv.gz) or manifest.json for composite payloads
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        namespace: str = "default",
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(cache_dir).expanduser().resolve()
        self.ns = namespace
        self.base = self.root / namespace
        self.base.mkdir(parents=True, exist_ok=True)
        self._log_fn = log_fn

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            self._log_fn(msg)

    def _entry_dir(self, key: str) -> Path:
        return self.base / key

    def _meta_path(self, key: str) -> Path:
        return self._entry_dir(key) / "meta.json"

    def _read_meta(self, key: str) -> CacheEntry | None:
        mp = self._meta_path(key)
        if not mp.exists():
            return None
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            return CacheEntry(
                key=str(data["key"]),
                created_at=float(data["created_at"]),
                ttl_seconds=float(data["ttl_seconds"]),
                meta=dict(data.get("meta") or {}),
            )
        except Exception:
            return None

    def _write_meta(self, *, key: str, ttl_seconds: float, meta: dict[str, Any] | None = None) -> CacheEntry:
        ed = self._entry_dir(key)
        ed.mkdir(parents=True, exist_ok=True)
        entry = CacheEntry(
            key=key,
            created_at=time.time(),
            ttl_seconds=float(ttl_seconds),
            meta=dict(meta or {}),
        )
        self._meta_path(key).write_text(
            _stable_json(
                {
                    "key": entry.key,
                    "created_at": entry.created_at,
                    "ttl_seconds": entry.ttl_seconds,
                    "meta": entry.meta,
                }
            ),
            encoding="utf-8",
        )
        return entry

    def _is_hit(self, key: str, *, refresh: bool) -> bool:
        if refresh:
            return False
        entry = self._read_meta(key)
        if entry is None:
            return False
        return entry.is_fresh()

    def get_or_compute_json(
        self,
        *,
        key: str,
        ttl_seconds: float,
        refresh: bool,
        compute_fn: Callable[[], Any],
        meta: dict[str, Any] | None = None,
    ) -> Any:
        ed = self._entry_dir(key)
        payload_path = ed / "payload.json"

        if self._is_hit(key, refresh=refresh) and payload_path.exists():
            endpoint = (meta or {}).get("endpoint") or "json"
            self._log(f"[gtrends] cache hit ({endpoint}) key={key}")
            return json.loads(payload_path.read_text(encoding="utf-8"))

        endpoint = (meta or {}).get("endpoint") or "json"
        if refresh:
            self._log(f"[gtrends] cache bypass refresh ({endpoint}) key={key}")
        else:
            self._log(f"[gtrends] cache miss ({endpoint}) key={key}")
        value = compute_fn()
        ed.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(_stable_json(value), encoding="utf-8")
        self._write_meta(key=key, ttl_seconds=ttl_seconds, meta=meta)
        self._log(f"[gtrends] cache write ({endpoint}) key={key}")
        return value

    def get_or_compute_df(
        self,
        *,
        key: str,
        ttl_seconds: float,
        refresh: bool,
        compute_fn: Callable[[], pd.DataFrame],
        meta: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        ed = self._entry_dir(key)
        payload_path = ed / "payload.csv.gz"

        if self._is_hit(key, refresh=refresh) and payload_path.exists():
            endpoint = (meta or {}).get("endpoint") or "df"
            self._log(f"[gtrends] cache hit ({endpoint}) key={key}")
            entry = self._read_meta(key)
            with gzip.open(payload_path, "rt", encoding="utf-8") as f:
                df = pd.read_csv(f, index_col=0)
            # restore datetime index only if it was originally datetime
            try:
                if entry and entry.meta.get("index") == "datetime":
                    df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                # If parsing fails, keep as-is.
                pass
            return df

        endpoint = (meta or {}).get("endpoint") or "df"
        if refresh:
            self._log(f"[gtrends] cache bypass refresh ({endpoint}) key={key}")
        else:
            self._log(f"[gtrends] cache miss ({endpoint}) key={key}")
        df = compute_fn()
        ed.mkdir(parents=True, exist_ok=True)
        with gzip.open(payload_path, "wt", encoding="utf-8") as f:
            df.to_csv(f)
        m = dict(meta or {})
        if isinstance(df.index, pd.DatetimeIndex):
            m["index"] = "datetime"
        self._write_meta(key=key, ttl_seconds=ttl_seconds, meta=m)
        self._log(f"[gtrends] cache write ({endpoint}) key={key}")
        return df

    def get_or_compute_df_dict(
        self,
        *,
        key: str,
        ttl_seconds: float,
        refresh: bool,
        compute_fn: Callable[[], dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Cache a mapping where values may include pandas DataFrames (common for related queries/topics).
        Stored as:
          manifest.json + payload files for each dataframe.
        """
        ed = self._entry_dir(key)
        manifest_path = ed / "manifest.json"

        if self._is_hit(key, refresh=refresh) and manifest_path.exists():
            endpoint = (meta or {}).get("endpoint") or "df_dict"
            self._log(f"[gtrends] cache hit ({endpoint}) key={key}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return self._load_df_dict_from_manifest(ed, manifest)

        endpoint = (meta or {}).get("endpoint") or "df_dict"
        if refresh:
            self._log(f"[gtrends] cache bypass refresh ({endpoint}) key={key}")
        else:
            self._log(f"[gtrends] cache miss ({endpoint}) key={key}")
        value = compute_fn() or {}
        ed.mkdir(parents=True, exist_ok=True)
        manifest = self._write_df_dict_manifest(ed, value)
        manifest_path.write_text(_stable_json(manifest), encoding="utf-8")
        self._write_meta(key=key, ttl_seconds=ttl_seconds, meta=meta)
        self._log(f"[gtrends] cache write ({endpoint}) key={key}")
        return value

    def _write_df_dict_manifest(self, ed: Path, value: dict[str, Any]) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        json_payload: dict[str, Any] = {}

        def _store_df(df: pd.DataFrame, rel: str) -> None:
            p = ed / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "wt", encoding="utf-8") as f:
                df.to_csv(f)
            files.append({"path": rel})

        for top_k, top_v in value.items():
            # related endpoints look like: {term: {"top": df, "rising": df}}
            if isinstance(top_v, dict):
                json_payload[top_k] = {}
                for sub_k, sub_v in top_v.items():
                    if isinstance(sub_v, pd.DataFrame):
                        rel = f"dfs/{cache_key(key, top_k, sub_k)}.csv.gz"
                        _store_df(sub_v, rel)
                        json_payload[top_k][sub_k] = {"__df__": rel}
                    else:
                        json_payload[top_k][sub_k] = sub_v
            else:
                json_payload[top_k] = top_v

        return {"json": json_payload, "files": files}

    def _load_df_dict_from_manifest(self, ed: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        json_payload = manifest.get("json") or {}

        for top_k, top_v in json_payload.items():
            if isinstance(top_v, dict):
                out[top_k] = {}
                for sub_k, sub_v in top_v.items():
                    if isinstance(sub_v, dict) and "__df__" in sub_v:
                        rel = str(sub_v["__df__"])
                        p = ed / rel
                        if p.exists():
                            with gzip.open(p, "rt", encoding="utf-8") as f:
                                df = pd.read_csv(f, index_col=0)
                            out[top_k][sub_k] = df
                        else:
                            out[top_k][sub_k] = pd.DataFrame()
                    else:
                        out[top_k][sub_k] = sub_v
            else:
                out[top_k] = top_v
        return out


class RateLimiter:
    def __init__(
        self,
        *,
        min_interval_seconds: float = 15.0,
        jitter_ratio: float = 0.2,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self.jitter_ratio = float(jitter_ratio)
        self._last = 0.0
        self._log_fn = log_fn

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            self._log_fn(msg)

    def wait(self, *, sleep_fn: Callable[[float], None] = time.sleep, now_fn: Callable[[], float] = time.monotonic) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = float(now_fn())
        elapsed = now - self._last
        if elapsed < 0:
            elapsed = 0

        # jitter in [1-j, 1+j]
        j = max(0.0, self.jitter_ratio)
        factor = 1.0
        if j > 0:
            # os.urandom-based jitter without importing random
            b = int.from_bytes(os.urandom(2), "big") / 65535.0
            factor = (1.0 - j) + (2.0 * j * b)
        target = self.min_interval_seconds * factor

        remaining = target - elapsed
        if remaining > 0:
            self._log(f"[gtrends] throttle sleep {remaining:.1f}s")
            sleep_fn(remaining)
        self._last = float(now_fn())


def backoff_retry(
    *,
    fn: Callable[[], Any],
    should_retry: Callable[[Exception], bool],
    max_attempts: int = 4,
    base_seconds: float = 30.0,
    max_seconds: float = 300.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] | None = None,
) -> Any:
    """
    Exponential backoff retry wrapper.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts or not should_retry(e):
                raise
            wait = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
            if log_fn is not None:
                log_fn(f"[gtrends] backoff retry attempt {attempt}/{max_attempts - 1}, sleeping {wait:.0f}s ({type(e).__name__})")
            sleep_fn(wait)


