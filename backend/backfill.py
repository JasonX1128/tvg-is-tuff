#!/usr/bin/env python3
"""
backfill.py — Seed / grow ERCOT data caches for forecasts + ML training.

Produces **two** output files:

  1. ``profile_cache.json``  — aggregated weekday×hour buckets for the
     website's fast fallback forecast.  Small (~10 KB), loaded at startup.

  2. ``ml_training_data.json``  — raw hourly time series with per-fuel-type
     MW values *and* computed renewable/low-carbon percentages.  Used
     exclusively for offline model training (``--retrain``) so we never
     need to re-call the EIA API for data we already have.

Strategy:
  • Today & yesterday → ``gridstatus`` ``get_fuel_mix`` (5-min, all fuels).
  • Older dates → **EIA Open Data API** hourly generation for every fuel
    type (nuclear, gas, coal, hydro, wind, solar, battery, other) back to
    2019.  No estimation or heuristics.

Usage examples:
    python3 backfill.py                           # last 7 days
    python3 backfill.py --hours 504               # last 3 weeks
    python3 backfill.py --hours 2160              # 90 days
    python3 backfill.py --retrain                 # retrain ML model from cached data
    python3 backfill.py --hours 504 --retrain     # fetch + retrain in one go
    python3 backfill.py --retrain-only            # retrain only (no API calls)
    EIA_API_KEY=xyz python3 backfill.py --hours 504

EIA API key:
  • Set via ``--eia-key`` flag or ``EIA_API_KEY`` environment variable.
  • Or place in ``backend/.env`` as ``EIA_API_KEY=…``.
  • Falls back to ``DEMO_KEY`` (rate-limited but functional for small runs).
  • Free registration: https://www.eia.gov/opendata/register.php

Notes:
  • Both caches are saved after every batch — Ctrl-C safe.
  • Re-runs are idempotent (duplicate timestamps are skipped).
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
DEFAULT_ML_DATA = Path(__file__).resolve().parent / "ml_training_data.json"
DEFAULT_MODEL = Path(__file__).resolve().parent / "forecast_model.pkl"
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
    """Fetch a single day of fuel-mix data → list of dicts with mix + pcts.

    Only works for 'today', 'latest', or yesterday's date.
    Returns points that include the raw fuel-type mix for ML training.
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
            "mix": mix,
        })
    return points


# ── EIA: older dates (hourly, all fuels, back to 2019) ────────────────────────

def _fetch_eia_range(start_date: date, end_date: date, api_key: str) -> list[dict]:
    """Fetch ERCOT hourly generation by fuel type from the EIA API.

    Returns a list of dicts with actual renewable/low-carbon % AND the
    raw per-fuel MW mix (for ML training).
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
            "mix": {k: round(v, 2) for k, v in mix.items()},
        })
    return points


# ── Profile cache I/O ─────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict:
    """Load existing profile cache, or return empty structure."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Drop legacy seen_timestamps — ml_data is now authoritative
        data.pop("seen_timestamps", None)
        return data
    return {"profiles": {}, "total_points_ingested": 0, "last_updated": None}


def save_cache(path: Path, cache: dict) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def rebuild_profile_cache(ml_data: dict) -> dict:
    """Reconstruct profile cache entirely from ml_data['points'].

    Useful when the cache has been corrupted (e.g. double-counted buckets)
    or deleted.  Returns a fresh cache dict with accurate bucket sums.
    """
    profiles: dict[str, dict] = {}
    count = 0
    for p in ml_data.get("points", []):
        try:
            dt = datetime.fromisoformat(p["timestamp"])
        except Exception:
            continue
        key = f"{dt.weekday()}_{dt.hour}"
        bucket = profiles.setdefault(
            key, {"renewable_sum": 0.0, "low_carbon_sum": 0.0, "count": 0}
        )
        bucket["renewable_sum"] += p["renewable_pct"]
        bucket["low_carbon_sum"] += p["low_carbon_pct"]
        bucket["count"] += 1
        count += 1
    return {
        "profiles": profiles,
        "total_points_ingested": count,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ── ML training data I/O ─────────────────────────────────────────────────────

def load_ml_data(path: Path) -> dict:
    """Load existing ML training data, or return empty structure."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Drop legacy seen_timestamps — points list is authoritative
        data.pop("seen_timestamps", None)
        return data
    return {"points": [], "last_updated": None}


def save_ml_data(path: Path, ml_data: dict) -> None:
    # Sort points by timestamp before saving for clean diffs
    ml_data["points"] = sorted(ml_data["points"], key=lambda p: p["timestamp"])
    with open(path, "w") as f:
        json.dump(ml_data, f, indent=1)


# ── Coverage checking ─────────────────────────────────────────────────────────

def _build_hourly_coverage(ml_data: dict) -> set[str]:
    """Build a set of hour-level keys ('YYYY-MM-DDTHH') from ml_data points.

    This is the authoritative source for what we already have — unlike
    seen_timestamps, it's never truncated.
    """
    covered: set[str] = set()
    for p in ml_data.get("points", []):
        try:
            dt = datetime.fromisoformat(p["timestamp"])
            covered.add(dt.strftime("%Y-%m-%dT%H"))
        except Exception:
            continue
    return covered


def _expected_hours(start_date: date, end_date: date) -> set[str]:
    """Generate the set of hourly keys we'd expect from EIA for a date range.

    EIA returns UTC hours: 00–23 for each day.
    """
    hours: set[str] = set()
    d = start_date
    while d <= end_date:
        for h in range(24):
            hours.add(f"{d.isoformat()}T{h:02d}")
        d += timedelta(days=1)
    return hours


def _check_range_coverage(
    start_date: date,
    end_date: date,
    coverage: set[str],
) -> tuple[int, int]:
    """Check how many hourly slots in [start_date, end_date] are already covered.

    Returns (present, expected).
    """
    expected = _expected_hours(start_date, end_date)
    present = expected & coverage
    return len(present), len(expected)


# ── Merge into both stores ───────────────────────────────────────────────────

def merge_points(cache: dict, ml_data: dict, points: list[dict]) -> int:
    """Merge data points into both the profile cache and the ML training store.

    Uses ml_data["points"] as the authoritative dedup source (never truncated),
    so re-runs never double-count into profile buckets even across years of data.

    Returns count of new (previously unseen) points added to the profile cache.
    """
    # Build dedup set from the *full* ML points list (authoritative source).
    # This replaces the old bounded seen_timestamps approach that caused
    # double-counting when data exceeded the 10K cap.
    ml_existing_ts: set[str] = set()
    for p in ml_data.get("points", []):
        ml_existing_ts.add(p["timestamp"])

    profiles = cache.setdefault("profiles", {})
    ml_points = ml_data.setdefault("points", [])
    new = 0

    for pt in points:
        ts_str = pt["timestamp"]
        if ts_str in ml_existing_ts:
            continue  # Already have this exact timestamp — skip entirely

        ml_existing_ts.add(ts_str)
        new += 1

        # ── Profile cache (aggregated buckets) ─────────────────────────
        try:
            dt = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        key = f"{dt.weekday()}_{dt.hour}"
        bucket = profiles.setdefault(
            key, {"renewable_sum": 0.0, "low_carbon_sum": 0.0, "count": 0}
        )
        bucket["renewable_sum"] += pt["renewable_pct"]
        bucket["low_carbon_sum"] += pt["low_carbon_pct"]
        bucket["count"] += 1

        # ── ML training data (individual time series with mix) ─────────
        ml_points.append({
            "timestamp": ts_str,
            "renewable_pct": round(pt["renewable_pct"], 6),
            "low_carbon_pct": round(pt["low_carbon_pct"], 6),
            "mix": pt.get("mix", {}),
        })

    cache["total_points_ingested"] = cache.get("total_points_ingested", 0) + new
    cache["last_updated"] = datetime.now(timezone.utc).isoformat()
    ml_data["last_updated"] = datetime.now(timezone.utc).isoformat()
    return new


# ── Retrain ML model from local data ─────────────────────────────────────────

def _seasonal_tier(points: list[dict]) -> int:
    """Decide how many seasonal features the data can support.

    Tier 0 (< 6 distinct months): no seasonal features — too little coverage.
    Tier 1 (6–11 months):         sin/cos month — smooth annual cycle.
    Tier 2 (≥ 12 months):         full suite — doy, month×hour, season flags.
    """
    months_seen = set()
    for p in points:
        dt = p.get("dt")
        if dt:
            months_seen.add(dt.month)
    n = len(months_seen)
    if n >= 12:
        return 2
    if n >= 6:
        return 1
    return 0


def _build_feature_row(
    dt: datetime,
    t0: datetime,
    y_r: "np.ndarray",
    i: int,
    tier: int,
) -> list[float]:
    """Build one feature row.  Shared by training and (via main.py) inference.

    Base features (always, 19):
      hour, sin/cos_hour, dow, sin/cos_dow, is_weekend,
      lag_{1,2,3,6,12,24}, roll_mean_{6,12,24}, roll_std_24,
      hour_x_weekend, days_since_start

    Tier 1 adds (+2 = 21):
      sin_month, cos_month

    Tier 2 adds (+5 = 26):
      sin_doy, cos_doy, month_x_hour, is_summer, is_winter
    """
    import math as _m
    import numpy as _np

    hr = dt.hour
    dow = dt.weekday()
    is_wknd = 1.0 if dow >= 5 else 0.0
    n = len(y_r)

    row: list[float] = [
        hr,
        _m.sin(2 * _m.pi * hr / 24),
        _m.cos(2 * _m.pi * hr / 24),
        dow,
        _m.sin(2 * _m.pi * dow / 7),
        _m.cos(2 * _m.pi * dow / 7),
        is_wknd,
        # lags
        y_r[i - 1]  if i >= 1  else 0.0,
        y_r[i - 2]  if i >= 2  else 0.0,
        y_r[i - 3]  if i >= 3  else 0.0,
        y_r[i - 6]  if i >= 6  else 0.0,
        y_r[i - 12] if i >= 12 else 0.0,
        y_r[i - 24] if i >= 24 else 0.0,
        # rolling
        float(_np.mean(y_r[i-6:i]))  if i >= 6  else float('nan'),
        float(_np.mean(y_r[i-12:i])) if i >= 12 else float('nan'),
        float(_np.mean(y_r[i-24:i])) if i >= 24 else float('nan'),
        float(_np.std(y_r[i-24:i]))  if i >= 24 else float('nan'),
        # interactions / trend
        hr * is_wknd,
        (dt - t0).total_seconds() / 86400,
    ]

    if tier >= 1:
        mon = dt.month
        row.append(_m.sin(2 * _m.pi * mon / 12))   # sin_month
        row.append(_m.cos(2 * _m.pi * mon / 12))   # cos_month

    if tier >= 2:
        doy = dt.timetuple().tm_yday
        mon = dt.month
        row.append(_m.sin(2 * _m.pi * doy / 365.25))  # sin_doy
        row.append(_m.cos(2 * _m.pi * doy / 365.25))  # cos_doy
        row.append(float(mon * hr))                     # month_x_hour
        row.append(1.0 if mon in (6, 7, 8) else 0.0)   # is_summer
        row.append(1.0 if mon in (12, 1, 2) else 0.0)  # is_winter

    return row


def _feature_names_for_tier(tier: int) -> list[str]:
    """Return ordered feature-name list for a given seasonal tier."""
    names = [
        "hour", "sin_hour", "cos_hour",
        "dow", "sin_dow", "cos_dow", "is_weekend",
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24",
        "roll_mean_6", "roll_mean_12", "roll_mean_24", "roll_std_24",
        "hour_x_weekend", "days_since_start",
    ]
    if tier >= 1:
        names += ["sin_month", "cos_month"]
    if tier >= 2:
        names += ["sin_doy", "cos_doy", "month_x_hour", "is_summer", "is_winter"]
    return names


def retrain_model(ml_data_path: Path, model_path: Path) -> None:
    """Train XGBoost + Ridge models from ml_training_data.json and save to .pkl.

    Seasonal features are included **conditionally** based on how many
    distinct months the training data covers:

      Tier 0 (< 6 months):  no seasonal features (avoids overfitting)
      Tier 1 (6–11 months): sin/cos month (smooth annual cycle)
      Tier 2 (≥ 12 months): + day-of-year, month×hour, season flags

    The chosen tier is stored in the pickle so the inference side can
    build the matching feature vector automatically.
    """
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from xgboost import XGBRegressor
    except ImportError:
        print("❌  Missing ML deps.  pip install scikit-learn xgboost numpy")
        return

    ml_data = load_ml_data(ml_data_path)
    raw_points = ml_data.get("points", [])
    if len(raw_points) < 200:
        print(f"❌  Only {len(raw_points)} points — need ≥200 for training.")
        return

    # ── Deduplicate to hourly resolution ──────────────────────────────
    hourly: dict[str, dict] = {}
    for p in raw_points:
        try:
            dt = datetime.fromisoformat(p["timestamp"])
        except Exception:
            continue
        hr_key = dt.strftime("%Y-%m-%dT%H")
        hourly[hr_key] = {
            "timestamp": p["timestamp"],
            "dt": dt,
            "renewable_pct": p["renewable_pct"],
            "low_carbon_pct": p["low_carbon_pct"],
        }

    points = sorted(hourly.values(), key=lambda p: p["dt"])
    n = len(points)

    # ── Determine seasonal tier ───────────────────────────────────────
    tier = _seasonal_tier(points)
    tier_labels = {
        0: "Tier 0 — no seasonal features (< 6 months of data)",
        1: "Tier 1 — sin/cos month (6–11 months)",
        2: "Tier 2 — full seasonal suite (≥ 12 months)",
    }
    feat_names = _feature_names_for_tier(tier)
    n_feat = len(feat_names)
    months_in_data = sorted(set(p["dt"].month for p in points))

    print(f"📊 Training on {n} hourly data points …")
    print(f"   Months in data: {months_in_data}  →  {tier_labels[tier]}")
    print(f"   Feature count: {n_feat}")

    # ── Build feature matrix ──────────────────────────────────────────
    y_r = np.array([p["renewable_pct"] for p in points])
    y_lc = np.array([p["low_carbon_pct"] for p in points])
    t0 = points[0]["dt"]

    rows = []
    for i in range(n):
        rows.append(_build_feature_row(points[i]["dt"], t0, y_r, i, tier))
    X = np.array(rows, dtype=np.float64)

    # Drop rows with NaN (first 24 h that lack full lag/rolling features)
    valid = ~np.isnan(X).any(axis=1)
    X_v = np.nan_to_num(X[valid], 0.0)
    y_r_v = y_r[valid]
    y_lc_v = y_lc[valid]

    xgb_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )

    print("   Training XGBoost (renewable) …", flush=True)
    xgb_r = XGBRegressor(**xgb_params)
    xgb_r.fit(X_v, y_r_v)

    print("   Training XGBoost (low-carbon) …", flush=True)
    xgb_lc = XGBRegressor(**xgb_params)
    xgb_lc.fit(X_v, y_lc_v)

    print("   Training Ridge (renewable) …", flush=True)
    ridge_r = Ridge(alpha=1.0)
    ridge_r.fit(X_v, y_r_v)

    print("   Training Ridge (low-carbon) …", flush=True)
    ridge_lc = Ridge(alpha=1.0)
    ridge_lc.fit(X_v, y_lc_v)

    model_data = {
        "xgb_renewable": xgb_r,
        "xgb_low_carbon": xgb_lc,
        "ridge_renewable": ridge_r,
        "ridge_low_carbon": ridge_lc,
        "feature_names": feat_names,
        "seasonal_tier": tier,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_points": len(X_v),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"✅ Model saved → {model_path}  ({len(X_v)} training rows)")
    print(f"   XGBoost top features: ", end="")
    imp = xgb_r.feature_importances_
    top = sorted(range(len(imp)), key=lambda i: imp[i], reverse=True)[:5]
    print(", ".join(f"{feat_names[i]}={imp[i]:.3f}" for i in top))


# ── EIA API key resolution ────────────────────────────────────────────────────

def _resolve_eia_key(cli_key: str | None) -> str:
    """EIA API key lookup: CLI flag → env var → .env file → DEMO_KEY."""
    import os as _os
    key = cli_key or _os.environ.get("EIA_API_KEY")
    if not key:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            try:
                for ln in env_path.read_text().splitlines():
                    if not ln or ln.strip().startswith("#"):
                        continue
                    if "EIA_API_KEY" in ln:
                        k = ln.split("=", 1)[1].strip().strip('"').strip("'")
                        if k:
                            key = k
                            break
            except Exception:
                pass
    return key or "DEMO_KEY"


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed / grow ERCOT data caches and optionally retrain ML models.",
        epilog=(
            "examples:\n"
            "  python3 backfill.py                            # last 7 days\n"
            "  python3 backfill.py --hours 504                 # last 3 weeks\n"
            "  python3 backfill.py --hours 2160                # 90 days\n"
            "  python3 backfill.py --retrain                   # retrain from cached data only\n"
            "  python3 backfill.py --hours 504 --retrain        # fetch + retrain\n"
            "  python3 backfill.py --retrain-only               # retrain only (no fetch)\n"
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
        "--ml-data", type=str, default=None,
        help=f"Path to ML training data JSON (default: {DEFAULT_ML_DATA})",
    )
    parser.add_argument(
        "--eia-key", type=str, default=None,
        help="EIA API key (or set EIA_API_KEY env var). Falls back to DEMO_KEY.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain the ML forecast model from ml_training_data.json after fetching.",
    )
    parser.add_argument(
        "--retrain-only", action="store_true",
        help="Only retrain (skip fetching). Uses existing ml_training_data.json.",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Rebuild profile_cache.json from ml_training_data.json (fixes corrupted buckets).",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve() if args.cache else DEFAULT_CACHE
    ml_data_path = Path(args.ml_data).resolve() if args.ml_data else DEFAULT_ML_DATA
    eia_key = _resolve_eia_key(args.eia_key)

    # ── Retrain-only mode ──────────────────────────────────────────────
    if args.retrain_only:
        print("🧠 Retrain-only mode (no API calls)")
        retrain_model(ml_data_path, DEFAULT_MODEL)
        return

    # ── Rebuild-cache mode ─────────────────────────────────────────────
    if args.rebuild_cache:
        print("🔄 Rebuilding profile_cache.json from ml_training_data.json …")
        ml_data = load_ml_data(ml_data_path)
        ml_pts = len(ml_data.get("points", []))
        if ml_pts == 0:
            print("❌ No ML training data found — nothing to rebuild from.")
            return
        cache = rebuild_profile_cache(ml_data)
        save_cache(cache_path, cache)
        n_slots = len(cache["profiles"])
        print(f"✅ Rebuilt {n_slots}/168 hourly slots from {ml_pts} points")
        print(f"   → {cache_path}")
        return

    days = math.ceil(args.hours / 24)
    today = date.today()
    yesterday = today - timedelta(days=1)
    oldest = today - timedelta(days=days - 1)

    print(f"📂 Profile cache   : {cache_path}")
    print(f"📂 ML training data: {ml_data_path}")
    print(f"🔑 EIA API key: {'(custom)' if eia_key != 'DEMO_KEY' else 'DEMO_KEY (rate-limited)'}")

    cache = load_cache(cache_path)
    ml_data = load_ml_data(ml_data_path)
    existing = cache.get("total_points_ingested", 0)
    n_existing_slots = len(cache.get("profiles", {}))
    ml_pts_existing = len(ml_data.get("points", []))
    print(f"   Profile : {existing} points, {n_existing_slots} hourly slots")
    print(f"   ML data : {ml_pts_existing} time-series points")
    print(f"   Requested: {args.hours} hours → {days} day(s)  ({oldest} … {today})")
    print(f"   Strategy: gridstatus for today/yesterday,"
          f" EIA API for older dates\n")

    ercot = gridstatus.Ercot()
    total_new = 0
    batches_ok = 0
    batches_fail = 0
    batches_skipped = 0

    # Build hourly coverage set from existing ML data for skip-checking.
    coverage = _build_hourly_coverage(ml_data)
    print(f"   Coverage  : {len(coverage)} distinct hours already cached\n")

    # ── Phase 1: today + yesterday via gridstatus (5-min resolution) ───
    # Always fetch — live data changes throughout the day.
    for i, (label, day_arg) in enumerate([("today", "today"), ("yesterday", yesterday)]):
        if i >= days:
            break
        print(f"⏳ {label} (gridstatus fuel-mix) …", end=" ", flush=True)
        try:
            pts = _fetch_day(ercot, day_arg)
            n = merge_points(cache, ml_data, pts)
            total_new += n
            batches_ok += 1
            print(f"✅ {len(pts)} pts ({n} new)")
        except Exception as e:
            batches_fail += 1
            print(f"❌ {str(e)[:120]}")
        save_cache(cache_path, cache)
        save_ml_data(ml_data_path, ml_data)

    # ── Phase 2: older dates via EIA API (hourly, all fuels) ───────────
    eia_days = days - 2  # subtract today + yesterday
    if eia_days > 0:
        eia_end = today - timedelta(days=2)       # day before yesterday
        eia_start = today - timedelta(days=days - 1)

        # Fetch in 7-day chunks to keep requests manageable
        chunk_size = 7
        chunk_start = eia_start
        chunk_idx = 0

        while chunk_start <= eia_end:
            chunk_end = min(chunk_start + timedelta(days=chunk_size - 1), eia_end)
            chunk_idx += 1
            label = (f"{chunk_start} → {chunk_end}"
                     if chunk_start != chunk_end else str(chunk_start))

            # ── Pre-check: skip if we already have all hours ───────────
            present, expected = _check_range_coverage(
                chunk_start, chunk_end, coverage
            )
            if present >= expected and expected > 0:
                batches_skipped += 1
                print(f"⏩ EIA batch {chunk_idx}: {label}"
                      f"  ({present}/{expected} hours cached — skipped)")
                chunk_start = chunk_end + timedelta(days=1)
                continue

            missing = expected - present
            print(f"⏳ EIA batch {chunk_idx}: {label}"
                  f"  ({missing}/{expected} hours missing) …",
                  end=" ", flush=True)

            try:
                pts = _fetch_eia_range(chunk_start, chunk_end, eia_key)
                n = merge_points(cache, ml_data, pts)
                total_new += n
                batches_ok += 1
                # Update coverage set with newly fetched data
                for p in pts:
                    try:
                        dt = datetime.fromisoformat(p["timestamp"])
                        coverage.add(dt.strftime("%Y-%m-%dT%H"))
                    except Exception:
                        pass
                print(f"✅ {len(pts)} pts ({n} new)")
            except Exception as e:
                batches_fail += 1
                print(f"❌ {str(e)[:120]}")

            save_cache(cache_path, cache)
            save_ml_data(ml_data_path, ml_data)
            chunk_start = chunk_end + timedelta(days=1)
            if chunk_start <= eia_end:
                time.sleep(1)

    # ── Summary ────────────────────────────────────────────────────────
    profiles = cache.get("profiles", {})
    total_pts = cache.get("total_points_ingested", 0)
    n_slots = len(profiles)
    ml_total = len(ml_data.get("points", []))

    total_batches = batches_ok + batches_fail + batches_skipped
    print(f"\n{'─' * 50}")
    print(f"✅ Profile cache   → {cache_path}")
    print(f"✅ ML training data → {ml_data_path}")
    print(f"   Batches      : {total_batches}  ({batches_ok} fetched, {batches_skipped} skipped, {batches_fail} failed)")
    print(f"   Profile pts  : {total_pts}  |  Hourly slots: {n_slots}/168 ({n_slots/168*100:.0f}%)")
    print(f"   ML data pts  : {ml_total}")
    print(f"   New this run : {total_new}")

    if n_slots < 168:
        missing = 168 - n_slots
        print(f"\n💡 {missing} hourly slots still empty."
              f" Run with --hours {max(168, args.hours)} or more.")
    else:
        print(f"\n🎉 Full weekly coverage!")

    # ── Optional: retrain ML model ─────────────────────────────────────
    if args.retrain:
        print()
        retrain_model(ml_data_path, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
