#!/usr/bin/env python3
"""
backfill.py — Seed / grow the hourly profile cache for better forecasts.

Fetches ERCOT generation data going back as many hours as you request
(default: 168 h = 1 week).  Each hour is merged into the existing cache
file with full deduplication by timestamp, so re-runs are always safe.

Strategy:
  • Today & yesterday → uses ``gridstatus`` ``get_fuel_mix`` (5-min, all fuels).
  • Older dates → uses the **EIA Open Data API** which provides *actual*
    hourly generation for every fuel type (nuclear, gas, coal, hydro, wind,
    solar, battery, other) back to 2019.  No estimation or heuristics.

Usage examples:
    python3 backfill.py                           # last 7 days (default)
    python3 backfill.py --hours 504               # last 3 weeks
    python3 backfill.py --hours 48                # just last 2 days
    python3 backfill.py --hours 336 --cache /tmp/my_cache.json
    EIA_API_KEY=xyz python3 backfill.py --hours 504   # use your own key

EIA API key:
  • Set via ``--eia-key`` flag or ``EIA_API_KEY`` environment variable.
  • Falls back to ``DEMO_KEY`` (rate-limited but functional for small runs).
  • Free registration: https://www.eia.gov/opendata/register.php

Notes:
  • The cache is saved after every batch so you can Ctrl-C without losing
    progress.
  • Re-running is always safe (idempotent) — duplicate timestamps are
    skipped automatically.
"""

import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌  Missing dependency.  pip install requests")
    sys.exit(1)

try:
    import gridstatus
    import pandas as pd
except ImportError:
    print("❌  Missing dependencies.  pip install gridstatus pandas")
    sys.exit(1)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CACHE = Path(__file__).resolve().parent / "profile_cache.json"
RENEWABLE_FUELS = {"Wind", "Solar"}
LOW_CARBON_FUELS = {"Wind", "Solar", "Nuclear", "Hydro"}

# EIA API
EIA_BASE = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
EIA_RESPONDENT = "ERCO"  # ERCOT's code in EIA data
EIA_MAX_PER_PAGE = 5000

# Map EIA fuel-type codes → our canonical names used in _pcts()
EIA_FUEL_MAP = {
    "WND": "Wind",
    "SUN": "Solar",
    "NUC": "Nuclear",
    "WAT": "Hydro",
    "NG":  "Natural Gas",
    "COL": "Coal and Lignite",
    "OTH": "Other",
    "BAT": "Power Storage",
}


def _pcts(mix: dict) -> tuple[float, float]:
    """Compute renewable % and low-carbon % from a fuel-type → MW dict."""
    total = sum(v for v in mix.values() if v > 0)
    if total <= 0:
        return 0.0, 0.0
    renew = sum(mix.get(f, 0) for f in RENEWABLE_FUELS)
    low_c = sum(mix.get(f, 0) for f in LOW_CARBON_FUELS)
    return max(renew, 0) / total, max(low_c, 0) / total


# ── gridstatus: today / yesterday (5-min, all fuels) ──────────────────────────

def _fetch_day(ercot, day) -> list[dict]:
    """Fetch a single day of fuel-mix data → list of profile-ready dicts.

    Only works for 'today', 'latest', or yesterday's date.
    """
    df: pd.DataFrame = ercot.get_fuel_mix(day)
    cols = [c for c in df.columns if c not in ("Time", "Interval Start", "Interval End")]
    points = []
    for _, row in df.iterrows():
        mix = {col: float(row[col]) for col in cols}
        r_pct, lc_pct = _pcts(mix)
        ts = row.get("Time") or row.get("Interval Start")
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        points.append({
            "timestamp": ts_str,
            "renewable_pct": r_pct,
            "low_carbon_pct": lc_pct,
        })
    return points


# ── EIA: older dates (hourly, all fuels, back to 2019) ────────────────────────

def _fetch_eia_range(start_date: date, end_date: date, api_key: str) -> list[dict]:
    """Fetch ERCOT hourly generation by fuel type from the EIA API.

    Returns a list of profile-ready dicts with actual renewable % and
    low-carbon % computed from real generation data (no estimation).
    """
    start_str = start_date.strftime("%Y-%m-%dT00")
    end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%dT00")

    # ── Paginated fetch ─────────────────────────────────────────────
    all_records: list[dict] = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": EIA_RESPONDENT,
            "start": start_str,
            "end": end_str,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": EIA_MAX_PER_PAGE,
        }
        # Retry with exponential backoff for rate-limiting
        for attempt in range(5):
            resp = requests.get(EIA_BASE, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 s
                print(f"\n      ⏸  Rate-limited, retrying in {wait}s …", end="", flush=True)
                time.sleep(wait)
                continue
            break
        resp.raise_for_status()
        body = resp.json()
        records = body.get("response", {}).get("data", [])
        if not records:
            break
        all_records.extend(records)
        total = int(body.get("response", {}).get("total", 0))
        offset += len(records)
        if offset >= total:
            break
        time.sleep(1)  # rate-limit courtesy

    if not all_records:
        raise ValueError(f"EIA returned no data for {start_date} – {end_date}")

    # ── Group by period (hour) and compute percentages ──────────────
    from collections import defaultdict
    hourly_mix: dict[str, dict[str, float]] = defaultdict(dict)

    for rec in all_records:
        period = rec.get("period", "")
        fuel_code = rec.get("fueltype", "")
        value = rec.get("value")
        if value is None or period == "":
            continue
        canonical = EIA_FUEL_MAP.get(fuel_code, fuel_code)
        try:
            hourly_mix[period][canonical] = float(value)
        except (ValueError, TypeError):
            continue

    points: list[dict] = []
    for period in sorted(hourly_mix):
        mix = hourly_mix[period]
        r_pct, lc_pct = _pcts(mix)
        # EIA hourly (UTC) period format: "2026-02-18T14"
        # Convert to ISO: "2026-02-18T14:00:00+00:00"
        if len(period) >= 13:
            ts = f"{period[:10]}T{period[11:13]}:00:00+00:00"
        else:
            ts = period
        points.append({
            "timestamp": ts,
            "renewable_pct": round(r_pct, 6),
            "low_carbon_pct": round(lc_pct, 6),
        })
    return points


def load_cache(path: Path) -> dict:
    """Load existing profile cache, or return empty structure."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"profiles": {}, "total_points_ingested": 0, "seen_timestamps": [], "last_updated": None}


def save_cache(path: Path, cache: dict) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def merge_points(cache: dict, points: list[dict]) -> int:
    """Merge data points into cache, deduplicating by timestamp.  Returns count of new points."""
    seen = set(cache.get("seen_timestamps", []))
    profiles = cache.setdefault("profiles", {})
    new = 0

    for pt in points:
        ts_str = pt["timestamp"]
        if ts_str in seen:
            continue
        seen.add(ts_str)
        new += 1

        try:
            dt = datetime.fromisoformat(ts_str)
        except Exception:
            continue

        key = f"{dt.weekday()}_{dt.hour}"
        bucket = profiles.setdefault(key, {"renewable_sum": 0.0, "low_carbon_sum": 0.0, "count": 0})
        bucket["renewable_sum"] += pt["renewable_pct"]
        bucket["low_carbon_sum"] += pt["low_carbon_pct"]
        bucket["count"] += 1

    # Keep seen_timestamps list bounded (last 30 days ≈ 8640 entries max)
    sorted_ts = sorted(seen)
    if len(sorted_ts) > 10_000:
        sorted_ts = sorted_ts[-10_000:]
    cache["seen_timestamps"] = sorted_ts
    cache["total_points_ingested"] = cache.get("total_points_ingested", 0) + new
    cache["last_updated"] = datetime.now(timezone.utc).isoformat()
    return new


def main():
    import os as _os

    parser = argparse.ArgumentParser(
        description="Seed / grow the ERCOT hourly-profile cache for forecasts.",
        epilog=(
            "examples:\n"
            "  python3 backfill.py                            # last 7 days\n"
            "  python3 backfill.py --hours 504                 # last 3 weeks\n"
            "  python3 backfill.py --hours 48                  # last 2 days\n"
            "  python3 backfill.py --hours 336 --cache /tmp/c.json\n"
            "  EIA_API_KEY=abc123 python3 backfill.py --hours 504\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hours", type=int, default=168,
        help="Hours of history to backfill (default: 168 = 1 week). Rounded up to full days.",
    )
    parser.add_argument(
        "--cache", type=str, default=None,
        help=f"Path to profile cache JSON (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--eia-key", type=str, default=None,
        help="EIA API key (or set EIA_API_KEY env var). Falls back to DEMO_KEY.",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve() if args.cache else DEFAULT_CACHE
    # EIA API key lookup order: CLI flag -> env var -> .env file -> DEMO_KEY
    eia_key = args.eia_key or _os.environ.get("EIA_API_KEY")
    if not eia_key:
        # check a simple .env file next to this script for convenience
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            try:
                for ln in env_path.read_text().splitlines():
                    if not ln or ln.strip().startswith("#"):
                        continue
                    if "EIA_API_KEY" in ln:
                        k = ln.split("=", 1)[1].strip().strip('"').strip("'")
                        if k:
                            eia_key = k
                            break
            except Exception:
                pass
    if not eia_key:
        eia_key = "DEMO_KEY"
    days = math.ceil(args.hours / 24)

    today = date.today()
    yesterday = today - timedelta(days=1)
    oldest = today - timedelta(days=days - 1)

    print(f"📂 Cache file: {cache_path}")
    print(f"🔑 EIA API key: {'(custom)' if eia_key != 'DEMO_KEY' else 'DEMO_KEY (rate-limited)'}")

    cache = load_cache(cache_path)
    existing = cache.get("total_points_ingested", 0)
    n_existing_slots = len(cache.get("profiles", {}))
    print(f"   Existing data: {existing} points ingested, {n_existing_slots} hourly slots")
    print(f"   Requested: {args.hours} hours → {days} day(s)  ({oldest} … {today})")
    print(f"   Strategy: gridstatus fuel-mix for today/yesterday,"
          f" EIA API for older dates (real data, all fuels)\n")

    ercot = gridstatus.Ercot()
    total_new = 0
    batches_ok = 0
    batches_fail = 0

    # ── Phase 1: today + yesterday via gridstatus (5-min resolution) ───
    for i, (label, day_arg) in enumerate([("today", "today"), ("yesterday", yesterday)]):
        if i >= days:
            break
        print(f"⏳ {label} (gridstatus fuel-mix) …", end=" ", flush=True)
        try:
            pts = _fetch_day(ercot, day_arg)
            n = merge_points(cache, pts)
            total_new += n
            batches_ok += 1
            print(f"✅ {len(pts)} pts ({n} new)")
        except Exception as e:
            batches_fail += 1
            err = str(e)[:120]
            print(f"❌ {err}")
        save_cache(cache_path, cache)

    # ── Phase 2: older dates via EIA API (hourly, all fuels) ───────────
    eia_days = days - 2  # subtract today + yesterday
    if eia_days > 0:
        eia_end = today - timedelta(days=2)       # day before yesterday
        eia_start = today - timedelta(days=days - 1)

        # Fetch in 7-day chunks to keep requests manageable
        chunk_size = 7
        chunk_start = eia_start
        chunk_idx = 0
        total_chunks = math.ceil((eia_end - eia_start).days + 1) / chunk_size

        while chunk_start <= eia_end:
            chunk_end = min(chunk_start + timedelta(days=chunk_size - 1), eia_end)
            chunk_idx += 1
            label = (f"{chunk_start} → {chunk_end}"
                     if chunk_start != chunk_end else str(chunk_start))
            print(f"⏳ EIA batch {chunk_idx}: {label} …", end=" ", flush=True)

            try:
                pts = _fetch_eia_range(chunk_start, chunk_end, eia_key)
                n = merge_points(cache, pts)
                total_new += n
                batches_ok += 1
                print(f"✅ {len(pts)} pts ({n} new)")
            except Exception as e:
                batches_fail += 1
                err = str(e)[:120]
                print(f"❌ {err}")

            save_cache(cache_path, cache)
            chunk_start = chunk_end + timedelta(days=1)
            if chunk_start <= eia_end:
                time.sleep(1)

    # ── Summary ────────────────────────────────────────────────────────────
    profiles = cache.get("profiles", {})
    total_pts = cache.get("total_points_ingested", 0)
    n_slots = len(profiles)

    print(f"\n{'─' * 50}")
    print(f"✅ Saved to {cache_path}")
    print(f"   Batches      : {batches_ok + batches_fail}  ({batches_ok} ok, {batches_fail} failed)")
    print(f"   Total points : {total_pts}")
    print(f"   New this run : {total_new}")
    print(f"   Hourly slots : {n_slots} / 168  ({n_slots / 168 * 100:.0f}% coverage)")

    if n_slots < 168:
        missing = 168 - n_slots
        print(f"\n💡 {missing} hourly slots still empty.")
        print(f"   Run with --hours {max(168, args.hours)} or more to fill remaining slots.")
    else:
        print(f"\n🎉 Full weekly coverage achieved! Every hour of every weekday has data.")


if __name__ == "__main__":
    main()
