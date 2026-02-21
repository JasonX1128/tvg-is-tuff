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
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

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

# ── Supabase (optional — enabled via --supabase flag) ─────────────────────────
_supabase_client = None


def _init_supabase():
    """Lazily initialise the Supabase client from env vars."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        print("❌  --supabase requires SUPABASE_URL and SUPABASE_KEY in .env")
        sys.exit(1)
    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        print(f"❌  Could not connect to Supabase: {exc}")
        sys.exit(1)


def _sb_upsert_batch(points: list[dict], *, quiet: bool = False) -> int:
    """Upsert points to ercot_history.  Returns count inserted."""
    sb = _init_supabase()
    rows = []
    for p in points:
        rows.append({
            "timestamp": p["timestamp"],
            "renewable_pct": round(p["renewable_pct"], 6),
            "low_carbon_pct": round(p["low_carbon_pct"], 6),
            "mix": p.get("mix", {}),
        })
    inserted = 0
    for i in range(0, len(rows), 500):
        chunk = rows[i : i + 500]
        sb.table("ercot_history").upsert(chunk, on_conflict="timestamp").execute()
        inserted += len(chunk)
    if not quiet:
        print(f"   ☁️  Supabase: upserted {inserted} rows")
    return inserted


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

    Outputs:
      - forecast_model.pkl     — pickled model bundle
      - training_report.json   — full metrics, feature importances, per-horizon errors
      - training_report.html   — visual report with Plotly charts
    """
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import TimeSeriesSplit
        from xgboost import XGBRegressor
    except ImportError:
        print("❌  Missing ML deps.  pip install scikit-learn xgboost numpy")
        return

    report_json_path = model_path.parent / "training_report.json"
    report_html_path = model_path.parent / "training_report.html"

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
            "mix": p.get("mix", {}),
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
    date_range_start = points[0]["dt"].strftime("%Y-%m-%d")
    date_range_end = points[-1]["dt"].strftime("%Y-%m-%d")
    span_days = (points[-1]["dt"] - points[0]["dt"]).days

    print(f"\n{'═' * 60}")
    print(f"  🧠  ML MODEL TRAINING")
    print(f"{'═' * 60}")
    print(f"  Data points     : {n:,} hourly ({len(raw_points):,} raw)")
    print(f"  Date range      : {date_range_start} → {date_range_end}  ({span_days} days)")
    print(f"  Months covered  : {months_in_data}")
    print(f"  Seasonal tier   : {tier_labels[tier]}")
    print(f"  Feature count   : {n_feat}")
    print(f"{'─' * 60}")

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

    # Also track timestamps for valid rows (for time-series charts)
    valid_dts = [points[i]["dt"] for i in range(n) if valid[i]]

    xgb_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )

    # ── Train all 4 models ────────────────────────────────────────────
    print("  Training XGBoost (renewable) …", end=" ", flush=True)
    xgb_r = XGBRegressor(**xgb_params)
    xgb_r.fit(X_v, y_r_v)
    print("✓")

    print("  Training XGBoost (low-carbon) …", end=" ", flush=True)
    xgb_lc = XGBRegressor(**xgb_params)
    xgb_lc.fit(X_v, y_lc_v)
    print("✓")

    print("  Training Ridge (renewable) …", end=" ", flush=True)
    ridge_r = Ridge(alpha=1.0)
    ridge_r.fit(X_v, y_r_v)
    print("✓")

    print("  Training Ridge (low-carbon) …", end=" ", flush=True)
    ridge_lc = Ridge(alpha=1.0)
    ridge_lc.fit(X_v, y_lc_v)
    print("✓")
    print(f"{'─' * 60}")

    # ── In-sample metrics ─────────────────────────────────────────────
    models_map = {
        "XGBoost Renewable":   (xgb_r,    y_r_v),
        "XGBoost Low-Carbon":  (xgb_lc,   y_lc_v),
        "Ridge Renewable":     (ridge_r,   y_r_v),
        "Ridge Low-Carbon":    (ridge_lc,  y_lc_v),
    }
    preds_map: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}

    for label, (model, y_true) in models_map.items():
        y_pred = model.predict(X_v)
        preds_map[label] = y_pred
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)
        metrics[label] = {"MAE": mae, "RMSE": rmse, "R²": r2}

    # Pretty-print in-sample metrics table
    print("  📈  IN-SAMPLE METRICS")
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    for label, m in metrics.items():
        print(f"  {label:<25} {m['MAE']*100:7.2f}% {m['RMSE']*100:7.2f}% {m['R²']:8.4f}")
    print(f"{'─' * 60}")

    # ── Cross-validation (time-series split) ──────────────────────────
    print("  📊  CROSS-VALIDATION (5-fold time-series split) …", flush=True)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results: dict[str, list[dict]] = {
        "XGBoost Renewable": [], "XGBoost Low-Carbon": [],
        "Ridge Renewable": [], "Ridge Low-Carbon": [],
    }
    cv_targets = {
        "XGBoost Renewable": y_r_v, "XGBoost Low-Carbon": y_lc_v,
        "Ridge Renewable": y_r_v, "Ridge Low-Carbon": y_lc_v,
    }
    holdout_preds: dict[str, "np.ndarray"] = {}
    holdout_val_idx = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_v), 1):
        X_tr, X_val = X_v[train_idx], X_v[val_idx]
        for label in cv_results:
            y_target = cv_targets[label]
            y_tr, y_val = y_target[train_idx], y_target[val_idx]
            if "XGBoost" in label:
                m = XGBRegressor(**xgb_params)
            else:
                m = Ridge(alpha=1.0)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)
            cv_results[label].append({
                "fold": fold,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "MAE": float(mean_absolute_error(y_val, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                "R²": float(r2_score(y_val, y_pred)),
            })
            # Capture last fold for out-of-sample charts
            if fold == 5:
                holdout_preds[label] = y_pred.copy()
        if fold == 5:
            holdout_val_idx = val_idx

    # Summarize CV
    cv_summary: dict[str, dict] = {}
    print(f"  {'Model':<25} {'MAE (mean±std)':>18} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─'*25} {'─'*18} {'─'*8} {'─'*8}")
    for label, folds in cv_results.items():
        maes = [f["MAE"] for f in folds]
        rmses = [f["RMSE"] for f in folds]
        r2s = [f["R²"] for f in folds]
        mean_mae, std_mae = float(np.mean(maes)), float(np.std(maes))
        mean_rmse = float(np.mean(rmses))
        mean_r2 = float(np.mean(r2s))
        cv_summary[label] = {
            "MAE_mean": mean_mae, "MAE_std": std_mae,
            "RMSE_mean": mean_rmse, "R²_mean": mean_r2,
            "folds": folds,
        }
        print(f"  {label:<25} {mean_mae*100:6.2f}% ± {std_mae*100:4.2f}% {mean_rmse*100:7.2f}% {mean_r2:8.4f}")
    print(f"{'─' * 60}")

    # ── Feature importances ───────────────────────────────────────────
    imp_r = xgb_r.feature_importances_
    imp_lc = xgb_lc.feature_importances_
    sorted_r = sorted(range(len(imp_r)), key=lambda i: imp_r[i], reverse=True)
    sorted_lc = sorted(range(len(imp_lc)), key=lambda i: imp_lc[i], reverse=True)

    feature_importance = {
        "renewable": [{"feature": feat_names[i], "importance": round(float(imp_r[i]), 4)} for i in sorted_r],
        "low_carbon": [{"feature": feat_names[i], "importance": round(float(imp_lc[i]), 4)} for i in sorted_lc],
    }

    print("  🏆  TOP FEATURE IMPORTANCES (XGBoost)")
    print(f"  {'Rank':<6} {'Renewable':<28} {'Low-Carbon':<28}")
    print(f"  {'─'*6} {'─'*28} {'─'*28}")
    for rank in range(min(10, n_feat)):
        ri = sorted_r[rank]
        li = sorted_lc[rank]
        r_txt = f"{feat_names[ri]:<20} {imp_r[ri]:.4f}"
        l_txt = f"{feat_names[li]:<20} {imp_lc[li]:.4f}"
        print(f"  {rank+1:<6} {r_txt:<28} {l_txt:<28}")
    print(f"{'─' * 60}")

    # ── Per-horizon error analysis (simulated forecast horizons) ──────
    # Walk the last 30 days and measure error at each horizon (1h–72h)
    print("  🔭  PER-HORIZON ERROR ANALYSIS …", flush=True)
    horizon_steps = [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72]
    horizon_errors: dict[int, dict] = {}

    # Use last 20% of data as test set for horizon analysis
    test_start = int(len(X_v) * 0.8)
    if test_start + max(horizon_steps) < len(X_v):
        for h in horizon_steps:
            xgb_errs_r, ridge_errs_r = [], []
            xgb_errs_lc, ridge_errs_lc = [], []
            for i in range(test_start, len(X_v) - h):
                # Predict at i for target at i+h by using lag features from i
                pred_xgb_r = float(xgb_r.predict(X_v[i:i+1])[0])
                pred_ridge_r = float(ridge_r.predict(X_v[i:i+1])[0])
                pred_xgb_lc = float(xgb_lc.predict(X_v[i:i+1])[0])
                pred_ridge_lc = float(ridge_lc.predict(X_v[i:i+1])[0])
                actual_r = float(y_r_v[i + h])
                actual_lc = float(y_lc_v[i + h])
                xgb_errs_r.append(abs(pred_xgb_r - actual_r))
                ridge_errs_r.append(abs(pred_ridge_r - actual_r))
                xgb_errs_lc.append(abs(pred_xgb_lc - actual_lc))
                ridge_errs_lc.append(abs(pred_ridge_lc - actual_lc))

            horizon_errors[h] = {
                "horizon_hours": h,
                "xgb_renewable_mae": round(float(np.mean(xgb_errs_r)), 6),
                "ridge_renewable_mae": round(float(np.mean(ridge_errs_r)), 6),
                "xgb_low_carbon_mae": round(float(np.mean(xgb_errs_lc)), 6),
                "ridge_low_carbon_mae": round(float(np.mean(ridge_errs_lc)), 6),
                "samples": len(xgb_errs_r),
            }

        print(f"  {'Horizon':>8} │ {'XGB Renew':>10} {'Ridge Renew':>12} │ {'XGB Low-C':>10} {'Ridge Low-C':>12}")
        print(f"  {'─'*8}─┼─{'─'*10}─{'─'*12}─┼─{'─'*10}─{'─'*12}")
        for h in horizon_steps:
            if h in horizon_errors:
                he = horizon_errors[h]
                print(f"  {h:>6}h  │ {he['xgb_renewable_mae']*100:9.2f}% {he['ridge_renewable_mae']*100:11.2f}% │"
                      f" {he['xgb_low_carbon_mae']*100:9.2f}% {he['ridge_low_carbon_mae']*100:11.2f}%")
    else:
        print("  ⚠️  Insufficient data for per-horizon analysis.")
        horizon_errors = {}
    print(f"{'─' * 60}")

    # ── Data distribution stats ───────────────────────────────────────
    data_stats = {
        "renewable_pct": {
            "mean": round(float(np.mean(y_r_v)), 4),
            "std": round(float(np.std(y_r_v)), 4),
            "min": round(float(np.min(y_r_v)), 4),
            "p25": round(float(np.percentile(y_r_v, 25)), 4),
            "median": round(float(np.median(y_r_v)), 4),
            "p75": round(float(np.percentile(y_r_v, 75)), 4),
            "max": round(float(np.max(y_r_v)), 4),
        },
        "low_carbon_pct": {
            "mean": round(float(np.mean(y_lc_v)), 4),
            "std": round(float(np.std(y_lc_v)), 4),
            "min": round(float(np.min(y_lc_v)), 4),
            "p25": round(float(np.percentile(y_lc_v, 25)), 4),
            "median": round(float(np.median(y_lc_v)), 4),
            "p75": round(float(np.percentile(y_lc_v, 75)), 4),
            "max": round(float(np.max(y_lc_v)), 4),
        },
    }

    print("  📊  DATA DISTRIBUTION")
    print(f"  {'Metric':<16} {'Mean':>8} {'Std':>8} {'Min':>8} {'P25':>8} {'Med':>8} {'P75':>8} {'Max':>8}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for label, st in data_stats.items():
        short = "Renewable" if "renewable" in label else "Low-Carbon"
        print(f"  {short:<16}"
              f" {st['mean']*100:7.1f}%"
              f" {st['std']*100:7.1f}%"
              f" {st['min']*100:7.1f}%"
              f" {st['p25']*100:7.1f}%"
              f" {st['median']*100:7.1f}%"
              f" {st['p75']*100:7.1f}%"
              f" {st['max']*100:7.1f}%")
    print(f"{'─' * 60}")

    # ── Hourly profile (average renewable % by hour of day) ───────────
    hourly_profile: dict[int, list[float]] = {h: [] for h in range(24)}
    for i, dt in enumerate(valid_dts):
        hourly_profile[dt.hour].append(float(y_r_v[i]))
    hourly_avg = {h: round(float(np.mean(vals)), 4) if vals else 0.0
                  for h, vals in hourly_profile.items()}

    print("  🕐  HOURLY RENEWABLE PROFILE (avg %)")
    bar_width = 30
    max_val = max(hourly_avg.values()) if hourly_avg else 1.0
    for h in range(24):
        val = hourly_avg[h]
        bar_len = int(val / max_val * bar_width) if max_val > 0 else 0
        print(f"  {h:02d}:00  {'█' * bar_len}{'░' * (bar_width - bar_len)} {val*100:5.1f}%")
    print(f"{'═' * 60}")

    # ── Save model pickle ─────────────────────────────────────────────
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
    print(f"  ✅ Model saved      → {model_path.name}")

    # ── Save training_report.json ─────────────────────────────────────
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "raw_points": len(raw_points),
            "hourly_deduplicated": n,
            "training_rows": len(X_v),
            "date_range": [date_range_start, date_range_end],
            "span_days": span_days,
            "months_covered": months_in_data,
        },
        "model": {
            "seasonal_tier": tier,
            "tier_label": tier_labels[tier],
            "feature_count": n_feat,
            "feature_names": feat_names,
            "xgb_params": xgb_params,
        },
        "in_sample_metrics": {k: {mk: round(mv, 6) for mk, mv in v.items()} for k, v in metrics.items()},
        "cross_validation": {k: {
            "MAE_mean": round(v["MAE_mean"], 6),
            "MAE_std": round(v["MAE_std"], 6),
            "RMSE_mean": round(v["RMSE_mean"], 6),
            "R²_mean": round(v["R²_mean"], 6),
            "folds": v["folds"],
        } for k, v in cv_summary.items()},
        "feature_importance": feature_importance,
        "horizon_errors": [v for v in horizon_errors.values()],
        "data_distribution": data_stats,
        "hourly_renewable_profile": hourly_avg,
    }
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Report JSON      → {report_json_path.name}")

    # ── Generate HTML report ──────────────────────────────────────────
    holdout_dts = [valid_dts[i] for i in holdout_val_idx]
    holdout_data = {
        "timestamps": holdout_dts,
        "actual_r": y_r_v[holdout_val_idx],
        "pred_r": holdout_preds["XGBoost Renewable"],
        "actual_lc": y_lc_v[holdout_val_idx],
        "pred_lc": holdout_preds["XGBoost Low-Carbon"],
    }
    _generate_training_html(report, holdout_data, report_html_path)
    print(f"  ✅ Report HTML      → {report_html_path.name}")
    print(f"{'═' * 60}\n")


def _generate_training_html(
    report: dict,
    holdout: dict,
    out_path: Path,
) -> None:
    """Generate a self-contained HTML report with Plotly charts.

    Args:
        report: Full training report dict.
        holdout: Out-of-sample (last CV fold) predictions with keys:
            timestamps, actual_r, pred_r, actual_lc, pred_lc
        out_path: Path to write the HTML file.
    """
    import numpy as np

    trained_at = report["trained_at"][:19].replace("T", " ") + " UTC"
    data_info = report["data"]
    model_info = report["model"]
    is_metrics = report["in_sample_metrics"]
    cv = report["cross_validation"]
    feat_imp = report["feature_importance"]
    horizon = report["horizon_errors"]
    dist = report["data_distribution"]
    hourly_prof = report["hourly_renewable_profile"]

    # ── Prepare chart data ────────────────────────────────────────────
    # Out-of-sample predictions from last CV fold
    h_all_dts = [dt.isoformat() for dt in holdout["timestamps"]]
    tail = min(500, len(h_all_dts))
    chart_dts = h_all_dts[-tail:]
    chart_actual_r = [round(float(v) * 100, 2) for v in holdout["actual_r"][-tail:]]
    chart_pred_r = [round(float(v) * 100, 2) for v in holdout["pred_r"][-tail:]]
    chart_actual_lc = [round(float(v) * 100, 2) for v in holdout["actual_lc"][-tail:]]
    chart_pred_lc = [round(float(v) * 100, 2) for v in holdout["pred_lc"][-tail:]]

    # Residuals (all holdout points, out-of-sample)
    residuals_r = [round(float(p - a) * 100, 3)
                   for p, a in zip(holdout["pred_r"], holdout["actual_r"])]
    residuals_lc = [round(float(p - a) * 100, 3)
                    for p, a in zip(holdout["pred_lc"], holdout["actual_lc"])]

    # Feature importance (top 15)
    fi_r = feat_imp["renewable"][:15]
    fi_lc = feat_imp["low_carbon"][:15]

    # Horizon errors
    h_hours = [h["horizon_hours"] for h in horizon]
    h_xgb_r = [round(h["xgb_renewable_mae"] * 100, 2) for h in horizon]
    h_ridge_r = [round(h["ridge_renewable_mae"] * 100, 2) for h in horizon]
    h_xgb_lc = [round(h["xgb_low_carbon_mae"] * 100, 2) for h in horizon]
    h_ridge_lc = [round(h["ridge_low_carbon_mae"] * 100, 2) for h in horizon]

    # Hourly profile
    hours_24 = list(range(24))
    hourly_vals = [round(hourly_prof.get(str(h), hourly_prof.get(h, 0)) * 100, 1) for h in hours_24]

    # CV fold data
    cv_fold_labels = [f"Fold {f['fold']}" for f in cv.get("XGBoost Renewable", {}).get("folds", [])]
    cv_xgb_r_maes = [round(f["MAE"] * 100, 2) for f in cv.get("XGBoost Renewable", {}).get("folds", [])]
    cv_ridge_r_maes = [round(f["MAE"] * 100, 2) for f in cv.get("Ridge Renewable", {}).get("folds", [])]

    # Build metrics table rows
    def _m_row(label: str, m: dict) -> str:
        return (f"<tr><td>{label}</td>"
                f"<td>{m['MAE']*100:.2f}%</td>"
                f"<td>{m['RMSE']*100:.2f}%</td>"
                f"<td>{m['R²']:.4f}</td></tr>")

    def _cv_row(label: str, s: dict) -> str:
        return (f"<tr><td>{label}</td>"
                f"<td>{s['MAE_mean']*100:.2f}% ± {s['MAE_std']*100:.2f}%</td>"
                f"<td>{s['RMSE_mean']*100:.2f}%</td>"
                f"<td>{s['R²_mean']:.4f}</td></tr>")

    def _dist_row(label: str, d: dict) -> str:
        return (f"<tr><td>{label}</td>"
                f"<td>{d['mean']*100:.1f}%</td><td>{d['std']*100:.1f}%</td>"
                f"<td>{d['min']*100:.1f}%</td><td>{d['p25']*100:.1f}%</td>"
                f"<td>{d['median']*100:.1f}%</td><td>{d['p75']*100:.1f}%</td>"
                f"<td>{d['max']*100:.1f}%</td></tr>")

    is_rows = "\n".join(_m_row(k, v) for k, v in is_metrics.items())
    cv_rows = "\n".join(_cv_row(k, v) for k, v in cv.items())
    dist_rows = "\n".join(_dist_row(k.replace("_", " ").title(), v) for k, v in dist.items())

    # Horizon table
    h_rows = "\n".join(
        f"<tr><td>{h['horizon_hours']}h</td>"
        f"<td>{h['xgb_renewable_mae']*100:.2f}%</td>"
        f"<td>{h['ridge_renewable_mae']*100:.2f}%</td>"
        f"<td>{h['xgb_low_carbon_mae']*100:.2f}%</td>"
        f"<td>{h['ridge_low_carbon_mae']*100:.2f}%</td>"
        f"<td>{h['samples']:,}</td></tr>"
        for h in horizon
    )


    # ── Build Plotly chart data as Python dicts (no JS spread) ─────────
    import copy
    _dark = {
        "paper_bgcolor": "#1a1d27", "plot_bgcolor": "#1a1d27",
        "font": {"color": "#e1e4ed", "family": "Inter, system-ui, sans-serif"},
        "xaxis": {"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a"},
        "yaxis": {"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a"},
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "legend": {"orientation": "h", "y": -0.15},
    }
    pc = {"responsive": True}

    def _ly(**kw):
        lo = copy.deepcopy(_dark)
        for k, v in kw.items():
            if k in lo and isinstance(lo[k], dict) and isinstance(v, dict):
                lo[k].update(v)
            else:
                lo[k] = v
        return lo

    avp_r_t = [
        {"x": chart_dts, "y": chart_actual_r, "name": "Actual", "type": "scatter",
         "mode": "lines", "line": {"color": "#4ade80", "width": 1}},
        {"x": chart_dts, "y": chart_pred_r, "name": "XGBoost (CV)", "type": "scatter",
         "mode": "lines", "line": {"color": "#f59e0b", "width": 1}},
    ]
    avp_r_l = _ly(title="Renewable % (out-of-sample)")

    avp_lc_t = [
        {"x": chart_dts, "y": chart_actual_lc, "name": "Actual", "type": "scatter",
         "mode": "lines", "line": {"color": "#60a5fa", "width": 1}},
        {"x": chart_dts, "y": chart_pred_lc, "name": "XGBoost (CV)", "type": "scatter",
         "mode": "lines", "line": {"color": "#f472b6", "width": 1}},
    ]
    avp_lc_l = _ly(title="Low-Carbon % (out-of-sample)")

    res_r_t = [{"x": residuals_r, "type": "histogram", "nbinsx": 60,
                "marker": {"color": "rgba(74,222,128,0.6)"}, "name": "Renewable"}]
    res_r_l = _ly(title="Renewable Residuals (pred\u2212actual)",
                  xaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "Error (pp)"},
                  yaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "Count"})

    res_lc_t = [{"x": residuals_lc, "type": "histogram", "nbinsx": 60,
                 "marker": {"color": "rgba(96,165,250,0.6)"}, "name": "Low-Carbon"}]
    res_lc_l = _ly(title="Low-Carbon Residuals (pred\u2212actual)",
                   xaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "Error (pp)"},
                   yaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "Count"})

    cv_t = [
        {"x": cv_fold_labels, "y": cv_xgb_r_maes, "name": "XGBoost",
         "type": "bar", "marker": {"color": "#f59e0b"}},
        {"x": cv_fold_labels, "y": cv_ridge_r_maes, "name": "Ridge",
         "type": "bar", "marker": {"color": "#60a5fa"}},
    ]
    cv_l = _ly(title="CV MAE by Fold (Renewable %)", barmode="group",
               yaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "MAE (%)"})

    hor_t = [
        {"x": h_hours, "y": h_xgb_r, "name": "XGB Renew", "type": "scatter",
         "mode": "lines+markers", "line": {"color": "#4ade80"}},
        {"x": h_hours, "y": h_ridge_r, "name": "Ridge Renew", "type": "scatter",
         "mode": "lines+markers", "line": {"color": "#60a5fa", "dash": "dot"}},
        {"x": h_hours, "y": h_xgb_lc, "name": "XGB Low-C", "type": "scatter",
         "mode": "lines+markers", "line": {"color": "#f59e0b"}},
        {"x": h_hours, "y": h_ridge_lc, "name": "Ridge Low-C", "type": "scatter",
         "mode": "lines+markers", "line": {"color": "#f472b6", "dash": "dot"}},
    ]
    hor_l = _ly(title="MAE by Forecast Horizon",
                xaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "Horizon (hours)"},
                yaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "MAE (%)"})

    fi_r_t = [{"y": [f["feature"] for f in fi_r[::-1]],
               "x": [f["importance"] for f in fi_r[::-1]],
               "type": "bar", "orientation": "h", "marker": {"color": "#4ade80"}}]
    fi_r_l = _ly(title="Renewable", margin={"l": 120, "r": 20, "t": 40, "b": 40}, height=420)

    fi_lc_t = [{"y": [f["feature"] for f in fi_lc[::-1]],
                "x": [f["importance"] for f in fi_lc[::-1]],
                "type": "bar", "orientation": "h", "marker": {"color": "#60a5fa"}}]
    fi_lc_l = _ly(title="Low-Carbon", margin={"l": 120, "r": 20, "t": 40, "b": 40}, height=420)

    hourly_min = min(hourly_vals) if hourly_vals else 0
    hourly_max = max(hourly_vals) if hourly_vals else 100
    hourly_t = [{"x": [f"{h:02d}:00" for h in hours_24], "y": hourly_vals,
                 "type": "bar", "marker": {"color": hourly_vals,
                    "colorscale": [[0, "#3b82f6"], [0.5, "#4ade80"], [1, "#f59e0b"]],
                    "cmin": hourly_min, "cmax": hourly_max,
                    "showscale": False}}]
    hourly_l = _ly(title="Avg Renewable % by Hour",
                   yaxis={"gridcolor": "#2a2d3a", "zerolinecolor": "#2a2d3a", "title": "Renewable %"})

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ERCOT ML Training Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --muted: #8b8fa3; --accent: #4ade80;
    --accent2: #60a5fa; --accent3: #f59e0b; --accent4: #f472b6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    padding: 24px; max-width: 1200px; margin: 0 auto;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  h2 {{
    font-size: 1.15rem; color: var(--accent); margin: 32px 0 12px;
    display: flex; align-items: center; gap: 8px;
  }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }}
  .grid {{ display: grid; gap: 16px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .grid-4 {{ grid-template-columns: 1fr 1fr 1fr 1fr; }}
  @media (max-width: 900px) {{ .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }}
  .kpi {{ text-align: center; padding: 16px; }}
  .kpi .value {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .kpi .label {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: rgba(74, 222, 128, 0.04); }}
  .chart {{ min-height: 350px; }}
  .tag {{
    display: inline-block; padding: 3px 10px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 600;
    background: rgba(74, 222, 128, 0.15); color: var(--accent);
  }}
</style>
</head>
<body>

<h1>🧠 ERCOT ML Training Report</h1>
<p class="subtitle">Trained {trained_at} &nbsp;·&nbsp; {data_info['training_rows']:,} rows &nbsp;·&nbsp;
   {data_info['date_range'][0]} → {data_info['date_range'][1]} ({data_info['span_days']} days)
   &nbsp;·&nbsp; <span class="tag">{model_info['tier_label']}</span></p>

<!-- KPI cards -->
<div class="grid grid-4" style="margin-bottom: 8px;">
  <div class="card kpi">
    <div class="value">{data_info['training_rows']:,}</div>
    <div class="label">Training Rows</div>
  </div>
  <div class="card kpi">
    <div class="value">{model_info['feature_count']}</div>
    <div class="label">Features</div>
  </div>
  <div class="card kpi">
    <div class="value" style="color: var(--accent2);">{is_metrics['XGBoost Renewable']['MAE']*100:.2f}%</div>
    <div class="label">XGB Renew MAE</div>
  </div>
  <div class="card kpi">
    <div class="value" style="color: var(--accent3);">{cv['XGBoost Renewable']['MAE_mean']*100:.2f}%</div>
    <div class="label">CV MAE (mean)</div>
  </div>
</div>

<!-- Actual vs Predicted (out-of-sample) -->
<h2>📈 Actual vs Predicted (out-of-sample, {tail} pts from last CV fold)</h2>
<div class="grid grid-2">
  <div class="card"><div id="chartAvP_R" class="chart"></div></div>
  <div class="card"><div id="chartAvP_LC" class="chart"></div></div>
</div>

<!-- Residual distribution (out-of-sample) -->
<h2>📊 Residual Distribution (out-of-sample)</h2>
<div class="grid grid-2">
  <div class="card"><div id="chartResid_R" class="chart"></div></div>
  <div class="card"><div id="chartResid_LC" class="chart"></div></div>
</div>

<!-- In-sample metrics -->
<h2>📋 In-Sample Metrics</h2>
<div class="card">
<table>
  <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
  <tbody>{is_rows}</tbody>
</table>
</div>

<!-- Cross-validation -->
<h2>🔄 Cross-Validation (5-fold Time Series)</h2>
<div class="grid grid-2">
  <div class="card">
    <table>
      <thead><tr><th>Model</th><th>MAE (mean±std)</th><th>RMSE</th><th>R²</th></tr></thead>
      <tbody>{cv_rows}</tbody>
    </table>
  </div>
  <div class="card"><div id="chartCV" class="chart"></div></div>
</div>

<!-- Horizon errors -->
<h2>🔭 Per-Horizon MAE</h2>
<div class="grid grid-2">
  <div class="card">
    <table>
      <thead><tr><th>Horizon</th><th>XGB Renew</th><th>Ridge Renew</th><th>XGB Low-C</th><th>Ridge Low-C</th><th>Samples</th></tr></thead>
      <tbody>{h_rows}</tbody>
    </table>
  </div>
  <div class="card"><div id="chartHorizon" class="chart"></div></div>
</div>

<!-- Feature importance -->
<h2>🏆 Feature Importance (XGBoost, top 15)</h2>
<div class="grid grid-2">
  <div class="card"><div id="chartFI_R" class="chart"></div></div>
  <div class="card"><div id="chartFI_LC" class="chart"></div></div>
</div>

<!-- Data distribution -->
<h2>📊 Data Distribution</h2>
<div class="card">
<table>
  <thead><tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>P25</th><th>Median</th><th>P75</th><th>Max</th></tr></thead>
  <tbody>{dist_rows}</tbody>
</table>
</div>

<!-- Hourly profile -->
<h2>🕐 Average Renewable % by Hour of Day</h2>
<div class="card"><div id="chartHourly" class="chart"></div></div>

<script>
Plotly.newPlot('chartAvP_R',{json.dumps(avp_r_t)},{json.dumps(avp_r_l)},{json.dumps(pc)});
Plotly.newPlot('chartAvP_LC',{json.dumps(avp_lc_t)},{json.dumps(avp_lc_l)},{json.dumps(pc)});
Plotly.newPlot('chartResid_R',{json.dumps(res_r_t)},{json.dumps(res_r_l)},{json.dumps(pc)});
Plotly.newPlot('chartResid_LC',{json.dumps(res_lc_t)},{json.dumps(res_lc_l)},{json.dumps(pc)});
Plotly.newPlot('chartCV',{json.dumps(cv_t)},{json.dumps(cv_l)},{json.dumps(pc)});
Plotly.newPlot('chartHorizon',{json.dumps(hor_t)},{json.dumps(hor_l)},{json.dumps(pc)});
Plotly.newPlot('chartFI_R',{json.dumps(fi_r_t)},{json.dumps(fi_r_l)},{json.dumps(pc)});
Plotly.newPlot('chartFI_LC',{json.dumps(fi_lc_t)},{json.dumps(fi_lc_l)},{json.dumps(pc)});
Plotly.newPlot('chartHourly',{json.dumps(hourly_t)},{json.dumps(hourly_l)},{json.dumps(pc)});
</script>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html)


# ── EIA API key resolution ────────────────────────────────────────────────────

def _resolve_eia_key(cli_key: str | None) -> str:
    """EIA API key lookup: CLI flag → env var → .env file → DEMO_KEY."""
    key = cli_key or os.environ.get("EIA_API_KEY")
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
            "  python3 backfill.py --supabase                   # fetch + push to Supabase\n"
            "  python3 backfill.py --hours 504 --supabase --retrain  # all-in-one\n"
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
    parser.add_argument(
        "--supabase", action="store_true",
        help="Also upsert fetched points to Supabase ercot_history table.",
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
            if args.supabase and pts:
                _sb_upsert_batch(pts, quiet=False)
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
                if args.supabase and pts:
                    _sb_upsert_batch(pts, quiet=False)
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
