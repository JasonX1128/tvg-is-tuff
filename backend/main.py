"""
Carbon-Aware Trigger API
Notifies developers via webhooks when the Texas (ERCOT) grid reaches a
specified renewable energy threshold.

Shift your compute to when the wind blows. 🌬️
"""

import asyncio
import csv
import io
import json
import logging
import math
import os
import pickle
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path
from pydantic import BaseModel, HttpUrl, Field
from supabase import create_client, Client as SupabaseClient

load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    import gridstatus

    ERCOT = gridstatus.Ercot()
except Exception:
    ERCOT = None

# ── Supabase client ────────────────────────────────────────────────────────
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
_supabase: SupabaseClient | None = None
if _SUPABASE_URL and _SUPABASE_KEY:
    try:
        _supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    except Exception as _exc:
        logging.getLogger(__name__).warning("Could not init Supabase client: %s", _exc)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

subscribers: dict[str, dict] = {}

fuel_mix_cache: dict = {
    "data": None,
    "renewable_pct": None,
    "timestamp": None,
    "mix": None,
}

# Rolling history for sparkline / time-series chart (last 288 = 24 h @ 5 min)
history: deque[dict] = deque(maxlen=288)

# Hourly profile: stores historical averages keyed by (day_of_week, hour).
# Each value is {"renewable_sum": float, "low_carbon_sum": float, "count": int}.
hourly_profile: dict[tuple[int, int], dict] = {}

# Path to trained ML forecast models (XGBoost + Ridge, created by forecast_experiment2.py).
FORECAST_MODEL_PATH = Path(__file__).resolve().parent / "forecast_model.pkl"

# ML models — loaded at startup, None if not available.
_ml_models: dict | None = None
_ml_seasonal_tier: int = 0  # seasonal feature tier stored in model pickle

# Notification log (last 200 events)
notification_log: deque[dict] = deque(maxlen=200)

# Shared HTTP client
_http_client: httpx.AsyncClient | None = None

POLL_INTERVAL_SECONDS = 300  # 5 minutes
RENEWABLE_FUELS = {"Wind", "Solar"}
LOW_CARBON_FUELS = {"Wind", "Solar", "Nuclear", "Hydro"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SubscribeRequest(BaseModel):
    callback_url: HttpUrl
    threshold: float = Field(..., gt=0.0, le=1.0, description="Fraction 0–1")
    trigger_on_drop: bool = False
    label: str = Field(
        default="",
        max_length=120,
        description="Optional human-friendly label for this subscription",
    )


class SubscribeResponse(BaseModel):
    id: str
    callback_url: str
    threshold: float
    trigger_on_drop: bool
    label: str
    created_at: str


class FuelMixResponse(BaseModel):
    timestamp: Optional[str] = None
    renewable_pct: Optional[float] = None
    low_carbon_pct: Optional[float] = None
    mix: Optional[dict] = None


class HistoryPoint(BaseModel):
    timestamp: str
    renewable_pct: float
    low_carbon_pct: float


class StatsResponse(BaseModel):
    current_renewable_pct: Optional[float] = None
    current_low_carbon_pct: Optional[float] = None
    avg_renewable_pct_24h: Optional[float] = None
    peak_renewable_pct_24h: Optional[float] = None
    peak_renewable_time: Optional[str] = None
    total_subscribers: int
    total_notifications_sent: int
    last_poll: Optional[str] = None
    green_score: Optional[str] = None
    co2_intensity_gco2_kwh: Optional[float] = None


class NotificationLogEntry(BaseModel):
    subscriber_id: str
    label: str
    callback_url: str
    threshold: float
    trigger_on_drop: bool
    renewable_pct: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    notified_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FUEL_COLORS = {
    "Wind": "#22c55e",
    "Solar": "#facc15",
    "Nuclear": "#a78bfa",
    "Hydro": "#38bdf8",
    "Natural Gas": "#f97316",
    "Coal and Lignite": "#78716c",
    "Power Storage": "#e879f9",
    "Other": "#64748b",
}

# Approximate lifecycle emission factors (gCO2eq/kWh) — IPCC 2014 medians
CO2_FACTORS: dict[str, float] = {
    "Wind": 11,
    "Solar": 45,
    "Nuclear": 12,
    "Hydro": 24,
    "Natural Gas": 490,
    "Coal and Lignite": 820,
    "Power Storage": 0,
    "Other": 400,
}


def _compute_mix(row, columns) -> dict:
    """Build a dict of fuel -> MW from a DataFrame row."""
    return {col: float(row[col]) for col in columns if col not in ("Time", "Interval Start", "Interval End")}


def _pcts(mix: dict) -> tuple[float, float]:
    """Return (renewable_pct, low_carbon_pct) from a mix dict."""
    total = sum(v for v in mix.values() if v > 0)  # ignore negative (storage discharge)
    if total <= 0:
        return 0.0, 0.0
    renew = sum(mix.get(f, 0.0) for f in RENEWABLE_FUELS)
    low_c = sum(mix.get(f, 0.0) for f in LOW_CARBON_FUELS)
    return renew / total, low_c / total


def _green_score(renewable_pct: float) -> str:
    """Return a letter grade A–F based on renewable percentage."""
    if renewable_pct >= 0.50:
        return "A"
    if renewable_pct >= 0.40:
        return "B"
    if renewable_pct >= 0.30:
        return "C"
    if renewable_pct >= 0.20:
        return "D"
    return "F"


def _co2_intensity(mix: dict) -> float:
    """Estimate grid carbon intensity in gCO2eq/kWh from current fuel mix."""
    total_mw = sum(max(v, 0) for v in mix.values())
    if total_mw <= 0:
        return 0.0
    weighted = sum(
        max(mix.get(fuel, 0), 0) * factor
        for fuel, factor in CO2_FACTORS.items()
    )
    return weighted / total_mw


def _fetch_fuel_mix() -> dict:
    """Fetch the latest ERCOT fuel mix (blocking)."""
    if ERCOT is None:
        raise RuntimeError("gridstatus is not available")

    df: pd.DataFrame = ERCOT.get_fuel_mix("latest")
    row = df.iloc[-1]
    cols = [c for c in df.columns if c not in ("Time", "Interval Start", "Interval End")]
    mix = {col: float(row[col]) for col in cols}

    renewable_pct, low_carbon_pct = _pcts(mix)

    ts = row.get("Time") or row.get("Interval Start")
    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    return {
        "timestamp": ts_str,
        "renewable_pct": renewable_pct,
        "low_carbon_pct": low_carbon_pct,
        "mix": mix,
    }


def _update_hourly_profile(ts_str: str, r_pct: float, lc_pct: float) -> None:
    """Accumulate a data point into the hourly_profile running average."""
    try:
        dt = datetime.fromisoformat(ts_str)
    except Exception:
        return
    key = (dt.weekday(), dt.hour)  # (0=Mon … 6=Sun, 0-23)
    bucket = hourly_profile.setdefault(key, {"renewable_sum": 0.0, "low_carbon_sum": 0.0, "count": 0})
    bucket["renewable_sum"] += r_pct
    bucket["low_carbon_sum"] += lc_pct
    bucket["count"] += 1


# ── Supabase helpers ───────────────────────────────────────────────────────────

def _sb_load_profile() -> None:
    """Load hourly profile from Supabase via the get_hourly_profile() RPC.

    Falls back silently if Supabase is not configured or the RPC doesn't
    exist yet — the app will just use live ERCOT data for the profile.
    """
    if _supabase is None:
        logger.info("Supabase not configured — hourly profile will seed from ERCOT.")
        return
    try:
        resp = _supabase.rpc("get_hourly_profile").execute()
        for row in resp.data:
            key = (int(row["dow"]), int(row["hr"]))
            cnt = int(row["cnt"])
            hourly_profile[key] = {
                "renewable_sum": row["avg_renewable"] * cnt,
                "low_carbon_sum": row["avg_low_carbon"] * cnt,
                "count": cnt,
            }
        total = sum(b["count"] for b in hourly_profile.values())
        logger.info("Loaded hourly profile from Supabase: %d points across %d slots",
                    total, len(hourly_profile))
    except Exception as exc:
        logger.warning("Could not load profile from Supabase: %s", exc)


def _sb_load_history(limit: int = 288) -> list[dict]:
    """Load recent history rows from Supabase (newest first → reversed)."""
    if _supabase is None:
        return []
    try:
        resp = (
            _supabase.table("ercot_history")
            .select("timestamp, renewable_pct, low_carbon_pct")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        # Reverse so oldest first (deque append order)
        rows = list(reversed(resp.data))
        logger.info("Loaded %d history points from Supabase", len(rows))
        return rows
    except Exception as exc:
        logger.warning("Could not load history from Supabase: %s", exc)
        return []


def _sb_upsert(points: list[dict]) -> int:
    """Upsert data points to ercot_history.  Returns count inserted.

    Each point must have: timestamp, renewable_pct, low_carbon_pct, mix.
    Duplicates (by timestamp) are silently updated.
    """
    if _supabase is None or not points:
        return 0
    try:
        rows = []
        for p in points:
            rows.append({
                "timestamp": p["timestamp"],
                "renewable_pct": round(p["renewable_pct"], 6),
                "low_carbon_pct": round(p["low_carbon_pct"], 6),
                "mix": p.get("mix", {}),
            })
        # Batch in chunks of 500 (Supabase row limit per request)
        inserted = 0
        for i in range(0, len(rows), 500):
            chunk = rows[i : i + 500]
            _supabase.table("ercot_history").upsert(
                chunk, on_conflict="timestamp"
            ).execute()
            inserted += len(chunk)
        return inserted
    except Exception as exc:
        logger.warning("Supabase upsert failed: %s", exc)
        return 0


def _get_hourly_avg(day_of_week: int, hour: int) -> tuple[float, float] | None:
    """Return (avg_renewable, avg_low_carbon) for a given day-of-week + hour.

    Falls back to same-hour across all days if the exact weekday isn't available,
    ensuring diurnal patterns are captured even with limited history.
    """
    # Try exact (weekday, hour) first
    key = (day_of_week, hour)
    b = hourly_profile.get(key)
    if b and b["count"] > 0:
        return b["renewable_sum"] / b["count"], b["low_carbon_sum"] / b["count"]
    # Fallback: average across all weekdays for this hour
    r_sum = lc_sum = cnt = 0
    for (_, h), bucket in hourly_profile.items():
        if h == hour and bucket["count"] > 0:
            r_sum += bucket["renewable_sum"]
            lc_sum += bucket["low_carbon_sum"]
            cnt += bucket["count"]
    if cnt > 0:
        return r_sum / cnt, lc_sum / cnt
    return None


def _fetch_yesterday_history() -> list[dict]:
    """Fetch yesterday's full fuel mix history (blocking)."""
    if ERCOT is None:
        raise RuntimeError("gridstatus is not available")
    from datetime import date, timedelta
    yesterday = date.today() - timedelta(days=1)
    df: pd.DataFrame = ERCOT.get_fuel_mix(yesterday)
    cols = [c for c in df.columns if c not in ("Time", "Interval Start", "Interval End")]
    points = []
    for _, row in df.iterrows():
        mix = {col: float(row[col]) for col in cols}
        r_pct, lc_pct = _pcts(mix)
        ts = row.get("Time") or row.get("Interval Start")
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        points.append({"timestamp": ts_str, "renewable_pct": r_pct, "low_carbon_pct": lc_pct, "mix": mix})
    return points


def _fetch_today_history() -> list[dict]:
    """Fetch today's full fuel mix history for the chart (blocking)."""
    if ERCOT is None:
        raise RuntimeError("gridstatus is not available")

    df: pd.DataFrame = ERCOT.get_fuel_mix("today")
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


async def _notify_subscriber(sub: dict, renewable_pct: float, mix: dict) -> None:
    """Send an HTTP POST to the subscriber's callback URL and log the result."""
    payload = {
        "event": "threshold_crossed",
        "renewable_pct": renewable_pct,
        "threshold": sub["threshold"],
        "trigger_on_drop": sub["trigger_on_drop"],
        "mix": mix,
        "notified_at": datetime.now(timezone.utc).isoformat(),
    }
    log_entry = {
        "subscriber_id": sub["id"],
        "label": sub.get("label", ""),
        "callback_url": sub["callback_url"],
        "threshold": sub["threshold"],
        "trigger_on_drop": sub["trigger_on_drop"],
        "renewable_pct": renewable_pct,
        "status_code": None,
        "error": None,
        "notified_at": payload["notified_at"],
    }
    client = _http_client or httpx.AsyncClient(timeout=10.0)
    try:
        resp = await client.post(str(sub["callback_url"]), json=payload)
        log_entry["status_code"] = resp.status_code
        logger.info("Notified %s (%s) → HTTP %s", sub["id"], sub.get("label", ""), resp.status_code)
    except Exception as exc:
        log_entry["error"] = str(exc)
        logger.warning("Failed to notify %s: %s", sub["id"], exc)

    notification_log.append(log_entry)


async def _poll_and_notify() -> None:
    """Background loop: poll ERCOT every POLL_INTERVAL_SECONDS, fire webhooks."""
    prev_condition: dict[str, bool] = {}

    # Load hourly profile from Supabase (replaces local JSON cache)
    await asyncio.to_thread(_sb_load_profile)

    # Load trained ML forecast models (XGBoost + Ridge)
    _load_forecast_models()

    # ── Seed history from Supabase first (persists across restarts) ────
    sb_pts = await asyncio.to_thread(_sb_load_history, 288)
    for p in sb_pts:
        history.append({
            "timestamp": p["timestamp"],
            "renewable_pct": p["renewable_pct"],
            "low_carbon_pct": p["low_carbon_pct"],
        })
    if sb_pts:
        fuel_mix_cache.update(sb_pts[-1])
    logger.info("Loaded %d history points from Supabase", len(sb_pts))

    # ── Supplement with live ERCOT data (today + yesterday if thin) ────
    try:
        logger.info("Seeding history with today's ERCOT data …")
        points = await asyncio.to_thread(_fetch_today_history)
        for p in points:
            history.append({
                "timestamp": p["timestamp"],
                "renewable_pct": p["renewable_pct"],
                "low_carbon_pct": p["low_carbon_pct"],
            })
            _update_hourly_profile(p["timestamp"], p["renewable_pct"], p["low_carbon_pct"])
        if points:
            fuel_mix_cache.update(points[-1])
        # Persist today's points to Supabase
        await asyncio.to_thread(_sb_upsert, points)
        logger.info("Seeded %d today points (total history: %d)", len(points), len(history))
    except Exception as exc:
        logger.error("Failed to seed today's history: %s", exc)

    # If we still have thin history, backfill from yesterday via ERCOT
    if len(history) < 24:
        try:
            logger.info("Fetching yesterday's ERCOT data for history …")
            yesterday_pts = await asyncio.to_thread(_fetch_yesterday_history)
            today_snapshot = list(history)
            history.clear()
            for p in yesterday_pts:
                history.append({
                    "timestamp": p["timestamp"],
                    "renewable_pct": p["renewable_pct"],
                    "low_carbon_pct": p["low_carbon_pct"],
                })
                _update_hourly_profile(p["timestamp"], p["renewable_pct"], p["low_carbon_pct"])
            for p in today_snapshot:
                history.append(p)
            # Persist yesterday's points to Supabase
            await asyncio.to_thread(_sb_upsert, yesterday_pts)
            logger.info("Back-filled %d yesterday + %d today points",
                         len(yesterday_pts), len(today_snapshot))
        except Exception as exc:
            logger.warning("Failed to fetch yesterday's data: %s", exc)

    while True:
        try:
            logger.info("Polling ERCOT fuel mix …")
            data = await asyncio.to_thread(_fetch_fuel_mix)
            fuel_mix_cache.update(data)

            history.append({
                "timestamp": data["timestamp"],
                "renewable_pct": data["renewable_pct"],
                "low_carbon_pct": data["low_carbon_pct"],
            })
            _update_hourly_profile(data["timestamp"], data["renewable_pct"], data["low_carbon_pct"])

            # Persist to Supabase
            await asyncio.to_thread(_sb_upsert, [data])

            renewable_pct = data["renewable_pct"]
            mix = data["mix"]

            for sub in list(subscribers.values()):
                threshold = sub["threshold"]
                trigger_on_drop = sub["trigger_on_drop"]
                sub_id = sub["id"]

                condition_met = (
                    (not trigger_on_drop and renewable_pct >= threshold)
                    or (trigger_on_drop and renewable_pct < threshold)
                )

                if condition_met and not prev_condition.get(sub_id, False):
                    await _notify_subscriber(sub, renewable_pct, mix)

                prev_condition[sub_id] = condition_met

        except Exception as exc:
            logger.error("Polling error: %s", exc)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=10.0)
    task = asyncio.create_task(_poll_and_notify())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await _http_client.aclose()


app = FastAPI(
    title="Carbon-Aware Trigger API",
    description=(
        "Webhooks that fire when Texas (ERCOT) renewable energy crosses "
        "your configured threshold. Shift your compute to when the wind blows! ⚡🌬️"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/subscribe", response_model=SubscribeResponse, status_code=201)
async def subscribe(req: SubscribeRequest):
    """Register a webhook to fire when the renewable threshold is crossed."""
    sub_id = str(uuid.uuid4())
    sub = {
        "id": sub_id,
        "callback_url": str(req.callback_url),
        "threshold": req.threshold,
        "trigger_on_drop": req.trigger_on_drop,
        "label": req.label or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    subscribers[sub_id] = sub
    logger.info("New subscriber %s '%s' (threshold=%.0f%%)", sub_id, req.label, req.threshold * 100)
    return sub


@app.delete("/subscribe/{sub_id}", status_code=204)
async def unsubscribe(sub_id: str):
    """Remove a webhook subscription."""
    if sub_id not in subscribers:
        raise HTTPException(status_code=404, detail="Subscriber not found.")
    del subscribers[sub_id]
    logger.info("Removed subscriber %s", sub_id)


@app.get("/subscribers", response_model=list[SubscribeResponse])
async def list_subscribers():
    """List all active webhook subscriptions."""
    return list(subscribers.values())


@app.get("/fuel-mix", response_model=FuelMixResponse)
async def get_fuel_mix():
    """Return the most recently fetched ERCOT fuel mix and renewable percentage."""
    if fuel_mix_cache.get("mix") is None:
        try:
            data = await asyncio.to_thread(_fetch_fuel_mix)
            fuel_mix_cache.update(data)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Unable to fetch ERCOT data: {exc}")

    return {
        "timestamp": fuel_mix_cache.get("timestamp"),
        "renewable_pct": fuel_mix_cache.get("renewable_pct"),
        "low_carbon_pct": fuel_mix_cache.get("low_carbon_pct"),
        "mix": fuel_mix_cache.get("mix"),
    }


@app.get("/history", response_model=list[HistoryPoint])
async def get_history(limit: int = Query(default=288, le=288, ge=1)):
    """Return recent renewable % history (up to 24 h at 5-min intervals)."""
    items = list(history)
    return items[-limit:]


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Dashboard statistics."""
    pts = list(history)
    avg_r = None
    peak_r = None
    peak_t = None
    if pts:
        avg_r = sum(p["renewable_pct"] for p in pts) / len(pts)
        best = max(pts, key=lambda p: p["renewable_pct"])
        peak_r = best["renewable_pct"]
        peak_t = best["timestamp"]

    current_r = fuel_mix_cache.get("renewable_pct")
    mix = fuel_mix_cache.get("mix")

    return {
        "current_renewable_pct": current_r,
        "current_low_carbon_pct": fuel_mix_cache.get("low_carbon_pct"),
        "avg_renewable_pct_24h": avg_r,
        "peak_renewable_pct_24h": peak_r,
        "peak_renewable_time": peak_t,
        "total_subscribers": len(subscribers),
        "total_notifications_sent": len(notification_log),
        "last_poll": fuel_mix_cache.get("timestamp"),
        "green_score": _green_score(current_r) if current_r is not None else None,
        "co2_intensity_gco2_kwh": round(_co2_intensity(mix), 1) if mix else None,
    }


@app.get("/notifications", response_model=list[NotificationLogEntry])
async def get_notifications(limit: int = Query(default=50, le=200, ge=1)):
    """Return recent webhook notification log."""
    items = list(notification_log)
    return items[-limit:][::-1]  # newest first


@app.get("/fuel-colors")
async def get_fuel_colors():
    """Return the canonical fuel → color mapping for chart consistency."""
    return FUEL_COLORS


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ercot_available": ERCOT is not None,
        "cache_populated": fuel_mix_cache.get("mix") is not None,
        "subscribers": len(subscribers),
        "history_points": len(history),
    }


# ── ML forecast helpers ────────────────────────────────────────────────────

def _load_forecast_models() -> None:
    """Load the trained XGBoost + Ridge models from disk (if available)."""
    global _ml_models, _ml_seasonal_tier
    try:
        if not FORECAST_MODEL_PATH.exists():
            logger.info("No forecast model at %s — will use profile fallback.", FORECAST_MODEL_PATH)
            return
        with open(FORECAST_MODEL_PATH, "rb") as f:
            _ml_models = pickle.load(f)
        _ml_seasonal_tier = _ml_models.get("seasonal_tier", 0)
        tier_labels = {0: "none", 1: "sin/cos month", 2: "full seasonal"}
        logger.info(
            "Loaded ML forecast models (trained on %d points at %s, seasonal: %s)",
            _ml_models.get("train_points", 0),
            _ml_models.get("trained_at", "?"),
            tier_labels.get(_ml_seasonal_tier, "?"),
        )
    except Exception as exc:
        logger.warning("Could not load forecast models: %s", exc)
        _ml_models = None


def _build_ml_features(
    future_dt: datetime,
    t0: datetime,
    recent_r: list[float],
    tier: int = 0,
) -> list[float]:
    """Build the feature vector expected by XGBoost / Ridge.

    The number of features depends on the seasonal tier stored in the
    trained model pickle:

      Tier 0 (< 6 months data): 19 base features — no seasonal.
      Tier 1 (6–11 months):     21 features — + sin/cos month.
      Tier 2 (≥ 12 months):     26 features — + doy, month×hour, season flags.

    Base features (always):
      hour, sin/cos_hour, dow, sin/cos_dow, is_weekend,
      lag_{1,2,3,6,12,24}, roll_mean_{6,12,24}, roll_std_24,
      hour_x_weekend, days_since_start
    """
    hr = future_dt.hour
    dow = future_dt.weekday()
    is_wknd = 1.0 if dow >= 5 else 0.0
    n = len(recent_r)

    feats: list[float] = [
        hr,                                           # hour
        math.sin(2 * math.pi * hr / 24),              # sin_hour
        math.cos(2 * math.pi * hr / 24),              # cos_hour
        dow,                                          # dow
        math.sin(2 * math.pi * dow / 7),              # sin_dow
        math.cos(2 * math.pi * dow / 7),              # cos_dow
        is_wknd,                                      # is_weekend
        # lags
        recent_r[-1]  if n >= 1  else 0.0,            # lag_1
        recent_r[-2]  if n >= 2  else 0.0,            # lag_2
        recent_r[-3]  if n >= 3  else 0.0,            # lag_3
        recent_r[-6]  if n >= 6  else 0.0,            # lag_6
        recent_r[-12] if n >= 12 else 0.0,            # lag_12
        recent_r[-24] if n >= 24 else 0.0,            # lag_24
        # rolling
        float(sum(recent_r[-6:])  / min(n, 6))  if n >= 1 else 0.0,
        float(sum(recent_r[-12:]) / min(n, 12)) if n >= 1 else 0.0,
        float(sum(recent_r[-24:]) / min(n, 24)) if n >= 1 else 0.0,
        float(_std(recent_r[-24:]))              if n >= 2 else 0.0,
        # interaction / trend
        hr * is_wknd,                                 # hour_x_weekend
        (future_dt - t0).total_seconds() / 86400,     # days_since_start
    ]

    # ── Conditional seasonal features ──────────────────────────────────
    if tier >= 1:
        mon = future_dt.month
        feats.append(math.sin(2 * math.pi * mon / 12))   # sin_month
        feats.append(math.cos(2 * math.pi * mon / 12))   # cos_month

    if tier >= 2:
        doy = future_dt.timetuple().tm_yday
        mon = future_dt.month
        feats.append(math.sin(2 * math.pi * doy / 365.25))  # sin_doy
        feats.append(math.cos(2 * math.pi * doy / 365.25))  # cos_doy
        feats.append(float(mon * hr))                         # month_x_hour
        feats.append(1.0 if mon in (6, 7, 8) else 0.0)       # is_summer
        feats.append(1.0 if mon in (12, 1, 2) else 0.0)      # is_winter

    return feats


def _std(vals: list[float]) -> float:
    """Population standard deviation."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


@app.get("/forecast")
async def get_forecast(hours: int = Query(default=24, ge=1, le=72)):
    """Predict renewable % and low-carbon % for the next *hours* hours.

    Uses a hybrid ML approach (XGBoost for short-term ≤ 6 h, Ridge for
    long-term) trained on 90 days of EIA hourly ERCOT generation data,
    with temporal features (hour, day-of-week, month, cyclical encodings)
    and lagged / rolling features from the live data feed.

    Falls back to historical weekday×hour bucket averages + trend blend
    when trained models are unavailable.
    """
    pts = list(history)
    if len(pts) < 6:
        raise HTTPException(status_code=503, detail="Not enough history for forecast.")

    try:
        base_dt = datetime.fromisoformat(pts[-1]["timestamp"])
    except Exception:
        base_dt = datetime.now(timezone.utc)

    # ── ML path ────────────────────────────────────────────────────────
    if _ml_models is not None:
        import numpy as np

        xgb_r = _ml_models["xgb_renewable"]
        xgb_lc = _ml_models["xgb_low_carbon"]
        ridge_r = _ml_models["ridge_renewable"]
        ridge_lc = _ml_models["ridge_low_carbon"]

        # Build a buffer of recent hourly renewable values for lag features.
        # History is 5-min; sample every 12th point to get hourly.
        hourly_r: list[float] = []
        step = max(1, len(pts) // 24)  # aim for ~24 hourly samples
        for j in range(0, len(pts), step):
            hourly_r.append(pts[j]["renewable_pct"])
        # Always include the latest point
        hourly_r.append(pts[-1]["renewable_pct"])

        # Pre-build a "direct" lag buffer: for future hours we substitute
        # the historical hourly-profile average instead of feeding the
        # model's own predictions back.  This avoids autoregressive
        # variance collapse (smooth predictions → smoother lags → ever-
        # narrower forecast range).  The profile comes from 2 years of
        # real ERCOT data so it preserves diurnal solar/wind patterns.
        n_actual = len(hourly_r)
        direct_r = list(hourly_r)  # copy actuals we already have
        for i in range(1, hours + 1):
            fut = base_dt + timedelta(hours=i)
            prof = _get_hourly_avg(fut.weekday(), fut.hour)
            direct_r.append(prof[0] if prof else direct_r[-1])

        # Reference t0 for the "days_since_start" feature — approximate
        t0 = base_dt - timedelta(days=90)

        forecast_pts: list[dict] = []
        for i in range(1, hours + 1):  # hourly resolution for ML
            future_dt = base_dt + timedelta(hours=i)

            # Use the direct lag buffer (actuals + profile) for feature
            # construction so the model sees realistic variability in its
            # lag/rolling inputs, matching training conditions.
            feats = _build_ml_features(future_dt, t0, direct_r[:n_actual + i], _ml_seasonal_tier)
            x = np.array(feats, dtype=np.float64).reshape(1, -1)

            # Hybrid blend: XGBoost dominates short-term, Ridge long-term.
            # Smooth sigmoid-style crossover centred at 6 h.
            xgb_weight = 1.0 / (1.0 + math.exp((i - 6) / 2.0))
            ridge_weight = 1.0 - xgb_weight

            pred_r = float(
                xgb_weight * xgb_r.predict(x)[0]
                + ridge_weight * ridge_r.predict(x)[0]
            )
            pred_lc = float(
                xgb_weight * xgb_lc.predict(x)[0]
                + ridge_weight * ridge_lc.predict(x)[0]
            )

            pred_r = max(0.0, min(1.0, pred_r))
            pred_lc = max(0.0, min(1.0, pred_lc))

            forecast_pts.append({
                "timestamp": future_dt.isoformat(),
                "renewable_pct": round(pred_r, 5),
                "low_carbon_pct": round(pred_lc, 5),
            })

        return {
            "method": "ml_hybrid_xgboost_ridge",
            "model_trained_at": _ml_models.get("trained_at", None),
            "model_train_points": _ml_models.get("train_points", 0),
            "seasonal_tier": _ml_seasonal_tier,
            "forecast_hours": hours,
            "points": forecast_pts,
        }

    # ── Fallback: bucket-average + trend blend ─────────────────────────
    window = pts[-24:] if len(pts) >= 24 else pts
    weights = [1.05 ** i for i in range(len(window))]
    w_sum = sum(weights)
    trend_r = sum(p["renewable_pct"] * w for p, w in zip(window, weights)) / w_sum
    trend_lc = sum(p["low_carbon_pct"] * w for p, w in zip(window, weights)) / w_sum

    forecast_pts = []
    n_points = hours * 12  # 5-min intervals

    for i in range(1, n_points + 1):
        future_dt = base_dt + timedelta(minutes=5 * i)
        profile = _get_hourly_avg(future_dt.weekday(), future_dt.hour)
        if profile:
            hist_r, hist_lc = profile
            blend_hist = min(0.7 + 0.02 * i, 0.9)
            r = hist_r * blend_hist + trend_r * (1.0 - blend_hist)
            lc = hist_lc * blend_hist + trend_lc * (1.0 - blend_hist)
        else:
            r, lc = trend_r, trend_lc
        forecast_pts.append({
            "timestamp": future_dt.isoformat(),
            "renewable_pct": round(max(0.0, min(1.0, r)), 5),
            "low_carbon_pct": round(max(0.0, min(1.0, lc)), 5),
        })

    return {
        "method": "historical_hourly_profile_blend",
        "profile_hours_available": len(hourly_profile),
        "based_on_points": len(window),
        "forecast_hours": hours,
        "points": forecast_pts,
    }


@app.get("/history/export")
async def export_history():
    """Download history as CSV."""
    pts = list(history)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "renewable_pct", "low_carbon_pct"])
    writer.writeheader()
    for p in pts:
        writer.writerow(p)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ercot_history.csv"},
    )


@app.get("/share")
async def share_card():
    """Generate a shareable status summary."""
    r = fuel_mix_cache.get("renewable_pct")
    lc = fuel_mix_cache.get("low_carbon_pct")
    mix = fuel_mix_cache.get("mix")
    if r is None:
        raise HTTPException(status_code=503, detail="No data yet.")
    co2 = round(_co2_intensity(mix), 1) if mix else None
    score = _green_score(r)
    pts = list(history)
    avg_r = sum(p["renewable_pct"] for p in pts) / len(pts) if pts else None
    peak = max((p["renewable_pct"] for p in pts), default=None)
    return {
        "title": "ERCOT Grid Status",
        "renewable_pct": round(r * 100, 1),
        "low_carbon_pct": round(lc * 100, 1) if lc else None,
        "green_score": score,
        "co2_intensity_gco2_kwh": co2,
        "avg_24h": round(avg_r * 100, 1) if avg_r else None,
        "peak_24h": round(peak * 100, 1) if peak else None,
        "timestamp": fuel_mix_cache.get("timestamp"),
        "share_text": (
            f"\u26a1 ERCOT Grid Report Card: {score}\n"
            f"\U0001f33f Renewable: {round(r*100,1)}%\n"
            f"\U0001f30d Low Carbon: {round(lc*100,1) if lc else '?'}%\n"
            f"\U0001f4a8 CO\u2082: {co2} gCO\u2082/kWh\n"
            f"\U0001f4c8 24h Peak: {round(peak*100,1) if peak else '?'}%\n"
            f"via Carbon-Aware Trigger"
        ),
    }


@app.get("/co2")
async def get_co2():
    """Estimated grid carbon intensity (gCO2eq/kWh) based on current fuel mix."""
    mix = fuel_mix_cache.get("mix")
    if not mix:
        raise HTTPException(status_code=503, detail="No fuel mix data yet.")
    intensity = _co2_intensity(mix)
    breakdown = {}
    total_mw = sum(max(v, 0) for v in mix.values())
    if total_mw > 0:
        for fuel, factor in CO2_FACTORS.items():
            mw = max(mix.get(fuel, 0), 0)
            breakdown[fuel] = {
                "mw": round(mw, 1),
                "share_pct": round(mw / total_mw, 4),
                "emission_factor_gco2_kwh": factor,
                "contribution_gco2_kwh": round(mw * factor / total_mw, 1),
            }
    return {
        "co2_intensity_gco2_kwh": round(intensity, 1),
        "breakdown": breakdown,
        "timestamp": fuel_mix_cache.get("timestamp"),
    }


@app.get("/stream")
async def stream_updates():
    """Server-Sent Events stream — pushes fuel mix updates in real time."""
    import json as _json

    async def event_generator():
        last_ts = None
        while True:
            ts = fuel_mix_cache.get("timestamp")
            if ts and ts != last_ts:
                last_ts = ts
                r = fuel_mix_cache.get("renewable_pct")
                payload = {
                    "timestamp": ts,
                    "renewable_pct": r,
                    "low_carbon_pct": fuel_mix_cache.get("low_carbon_pct"),
                    "mix": fuel_mix_cache.get("mix"),
                    "green_score": _green_score(r) if r is not None else None,
                    "co2_intensity_gco2_kwh": round(
                        _co2_intensity(fuel_mix_cache.get("mix", {})), 1
                    ),
                }
                yield f"data: {_json.dumps(payload)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Serve the single-file frontend (must be last — catch-all)
# ---------------------------------------------------------------------------

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    """Serve the frontend SPA."""
    index = _frontend_dir / "index.html"
    if index.is_file():
        return HTMLResponse(content=index.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>Frontend not found — place index.html in /frontend</h1>",
        status_code=404,
    )
