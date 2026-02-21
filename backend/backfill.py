#!/usr/bin/env python3
"""
backfill.py — Seed / grow the hourly profile cache for better forecasts.

Fetches ERCOT data day-by-day going back as many hours as you request
(default: 168 h = 1 week).  Each day is merged into the existing cache
file with full deduplication by timestamp, so re-runs are always safe.

Strategy:
  • Today & yesterday → uses ``get_fuel_mix`` (fast, accurate, all fuels).
  • Older dates → downloads hourly wind + solar + load reports from ERCOT's
    MIS archive and computes renewable %.  Low-carbon % is estimated by
    adding ERCOT's typical nuclear + hydro share (~11 %).  Downloaded files
    are fetched inside a temp directory and deleted automatically.

Usage examples:
    python3 backfill.py                           # last 7 days (default)
    python3 backfill.py --hours 504               # last 3 weeks
    python3 backfill.py --hours 48                # just last 2 days
    python3 backfill.py --hours 336 --cache /tmp/my_cache.json

Notes:
  • The cache is saved after every day so you can Ctrl-C without losing
    progress.
  • Re-running is always safe (idempotent) — duplicate timestamps are
    skipped automatically.
  • Older-date fetches download report files from ERCOT; all temp files
    are cleaned up after each day.
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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

# ERCOT nuclear (Comanche Peak + South Texas Project) ≈ 5.1 GW on a ~45-55 GW
# grid.  Hydro is tiny (~0.3 %).  Combined they contribute roughly 11 % of load.
# Used as an adder when only wind+solar+load are available (older dates).
NUCLEAR_HYDRO_ESTIMATE = 0.11


def _pcts(mix: dict) -> tuple[float, float]:
    total = sum(v for v in mix.values() if v > 0)
    if total <= 0:
        return 0.0, 0.0
    renew = sum(mix.get(f, 0) for f in RENEWABLE_FUELS)
    low_c = sum(mix.get(f, 0) for f in LOW_CARBON_FUELS)
    return max(renew, 0) / total, max(low_c, 0) / total


@contextmanager
def _clean_workdir():
    """Run inside a temp directory so any files gridstatus downloads are
    automatically deleted when the block exits (even on error)."""
    orig = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="ercot_backfill_")
    os.chdir(tmp)
    try:
        yield tmp
    finally:
        os.chdir(orig)
        shutil.rmtree(tmp, ignore_errors=True)


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


def _fetch_day_historical(ercot, target_date: date) -> list[dict]:
    """Approximate renewable % for an older date using hourly wind + solar +
    load reports downloaded from ERCOT's MIS archive.

    Returns ~24 data points (one per hour).  ``low_carbon_pct`` is estimated
    by adding ``NUCLEAR_HYDRO_ESTIMATE`` to the computed renewable fraction.
    """
    wind_df = ercot.get_hourly_wind_report(target_date)
    solar_df = ercot.get_hourly_solar_report(target_date)
    load_df = ercot.get_load(target_date)

    # ── Wind: keep only actual-generation rows for the target date ─────
    wind_df = wind_df[wind_df["GEN SYSTEM WIDE"].notna()].copy()
    wind_by_hour: dict[int, float] = {}
    wind_ts_by_hour: dict[int, str] = {}
    for _, row in wind_df.iterrows():
        ts = row.get("Interval Start") or row.get("Time")
        if hasattr(ts, "date") and ts.date() != target_date:
            continue
        h = ts.hour
        wind_by_hour[h] = float(row["GEN SYSTEM WIDE"])
        wind_ts_by_hour[h] = ts.isoformat()

    # ── Solar: same treatment ──────────────────────────────────────────
    solar_df = solar_df[solar_df["GEN SYSTEM WIDE"].notna()].copy()
    solar_by_hour: dict[int, float] = {}
    for _, row in solar_df.iterrows():
        ts = row.get("Interval Start") or row.get("Time")
        if hasattr(ts, "date") and ts.date() != target_date:
            continue
        solar_by_hour[ts.hour] = float(row["GEN SYSTEM WIDE"])

    # ── Load: average the 5-min readings into hourly buckets ───────────
    load_sums: dict[int, float] = {}
    load_counts: dict[int, int] = {}
    for _, row in load_df.iterrows():
        ts = row.get("Interval Start") or row.get("Time")
        if hasattr(ts, "date") and ts.date() != target_date:
            continue
        h = ts.hour
        load_sums[h] = load_sums.get(h, 0.0) + float(row["Load"])
        load_counts[h] = load_counts.get(h, 0) + 1
    load_by_hour = {
        h: load_sums[h] / load_counts[h]
        for h in load_sums
        if load_counts.get(h, 0) > 0
    }

    # ── Combine into profile points ────────────────────────────────────
    # Need at least wind + load for the same hour
    common_hours = sorted(set(wind_by_hour) & set(load_by_hour))
    if not common_hours:
        raise ValueError(f"No overlapping wind+load data for {target_date}")

    points: list[dict] = []
    for h in common_hours:
        load_val = load_by_hour[h]
        if load_val <= 0:
            continue
        wind_gen = max(wind_by_hour.get(h, 0), 0)
        solar_gen = max(solar_by_hour.get(h, 0), 0)
        r = min((wind_gen + solar_gen) / load_val, 1.0)
        lc = min(r + NUCLEAR_HYDRO_ESTIMATE, 1.0)
        points.append({
            "timestamp": wind_ts_by_hour[h],
            "renewable_pct": round(r, 6),
            "low_carbon_pct": round(lc, 6),
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
    parser = argparse.ArgumentParser(
        description="Seed / grow the ERCOT hourly-profile cache for forecasts.",
        epilog=(
            "examples:\n"
            "  python3 backfill.py                        # last 7 days\n"
            "  python3 backfill.py --hours 504             # last 3 weeks\n"
            "  python3 backfill.py --hours 48              # last 2 days\n"
            "  python3 backfill.py --hours 336 --cache /tmp/cache.json\n"
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
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve() if args.cache else DEFAULT_CACHE
    days = math.ceil(args.hours / 24)

    print(f"📂 Cache file: {cache_path}")

    cache = load_cache(cache_path)
    existing = cache.get("total_points_ingested", 0)
    n_existing_slots = len(cache.get("profiles", {}))
    print(f"   Existing data: {existing} points ingested, {n_existing_slots} hourly slots")
    print(f"   Requested: {args.hours} hours → {days} day(s) of backfill")

    today = date.today()
    yesterday = today - timedelta(days=1)

    print(f"   Strategy: fuel-mix for today/yesterday,"
          f" wind+solar+load reports for older dates\n")

    ercot = gridstatus.Ercot()
    total_new = 0
    days_ok = 0
    days_fail = 0

    for i in range(days):
        target = today - timedelta(days=i)
        use_fuel_mix = target >= yesterday  # today or yesterday
        method = "fuel-mix" if use_fuel_mix else "wind+solar+load"
        label = "today" if target == today else str(target)

        print(f"⏳ [{i + 1}/{days}] {label} ({method}) …", end=" ", flush=True)
        try:
            with _clean_workdir():
                if use_fuel_mix:
                    day_arg = "today" if target == today else target
                    pts = _fetch_day(ercot, day_arg)
                else:
                    pts = _fetch_day_historical(ercot, target)
            n = merge_points(cache, pts)
            total_new += n
            days_ok += 1
            print(f"✅ {len(pts)} pts ({n} new)")
        except Exception as e:
            days_fail += 1
            err = str(e)
            if len(err) > 120:
                err = err[:120] + "…"
            print(f"❌ {err}")

        # Save after each day so Ctrl-C doesn't lose progress
        save_cache(cache_path, cache)

        # Pause between API requests to be polite (longer for historical
        # fetches which hit three endpoints per day)
        if i < days - 1:
            time.sleep(2 if not use_fuel_mix else 1)

    # ── Summary ────────────────────────────────────────────────────────────
    profiles = cache.get("profiles", {})
    total_pts = cache.get("total_points_ingested", 0)
    n_slots = len(profiles)

    print(f"\n{'─' * 50}")
    print(f"✅ Saved to {cache_path}")
    print(f"   Days attempted : {days}  ({days_ok} succeeded, {days_fail} failed)")
    print(f"   Total points   : {total_pts}")
    print(f"   New this run   : {total_new}")
    print(f"   Hourly slots   : {n_slots} / 168  ({n_slots / 168 * 100:.0f}% coverage)")

    if n_slots < 168:
        missing = 168 - n_slots
        print(f"\n💡 {missing} hourly slots still empty.")
        print(f"   Run daily to accumulate more weekday coverage.")
        print(f"   Full 3-week depth: ~{max(1, 21 - total_pts // 288)} more day(s) of runs.")
    else:
        print(f"\n🎉 Full weekly coverage achieved! Every hour of every weekday has data.")


if __name__ == "__main__":
    main()
