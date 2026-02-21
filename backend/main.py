"""
Carbon-Aware Trigger API
Notifies developers via webhooks when the Texas (ERCOT) grid reaches a
specified renewable energy threshold.

Shift your compute to when the wind blows. 🌬️
"""

import asyncio
import logging
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

try:
    import gridstatus

    ERCOT = gridstatus.Ercot()
except Exception:
    ERCOT = None

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

    # Seed history with today's data on first run
    try:
        logger.info("Seeding history with today's ERCOT data …")
        points = await asyncio.to_thread(_fetch_today_history)
        for p in points:
            history.append({
                "timestamp": p["timestamp"],
                "renewable_pct": p["renewable_pct"],
                "low_carbon_pct": p["low_carbon_pct"],
            })
        if points:
            latest = points[-1]
            fuel_mix_cache.update(latest)
        logger.info("Seeded %d history points", len(points))
    except Exception as exc:
        logger.error("Failed to seed history: %s", exc)

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

    return {
        "current_renewable_pct": fuel_mix_cache.get("renewable_pct"),
        "current_low_carbon_pct": fuel_mix_cache.get("low_carbon_pct"),
        "avg_renewable_pct_24h": avg_r,
        "peak_renewable_pct_24h": peak_r,
        "peak_renewable_time": peak_t,
        "total_subscribers": len(subscribers),
        "total_notifications_sent": len(notification_log),
        "last_poll": fuel_mix_cache.get("timestamp"),
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
