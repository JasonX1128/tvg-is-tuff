"""
Carbon-Aware Trigger API
Notifies developers via webhooks when the Texas (ERCOT) grid reaches a
specified renewable energy threshold.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

try:
    import gridstatus

    ERCOT = gridstatus.Ercot()
except Exception:
    ERCOT = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory store for subscribers
subscribers: dict[str, dict] = {}

# Cache for the last known fuel mix
fuel_mix_cache: dict = {
    "data": None,
    "renewable_pct": None,
    "timestamp": None,
}

# Shared HTTP client for webhook delivery
_http_client: httpx.AsyncClient | None = None

POLL_INTERVAL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SubscribeRequest(BaseModel):
    callback_url: HttpUrl
    threshold: float  # 0.0 – 1.0, e.g. 0.40 means 40 %
    trigger_on_drop: bool = False  # True → ping when renewable % DROPS below threshold


class SubscribeResponse(BaseModel):
    id: str
    callback_url: str
    threshold: float
    trigger_on_drop: bool
    created_at: str


class FuelMixResponse(BaseModel):
    timestamp: Optional[str]
    renewable_pct: Optional[float]
    mix: Optional[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_fuel_mix() -> dict:
    """Fetch the latest ERCOT fuel mix and compute renewable percentage."""
    if ERCOT is None:
        raise RuntimeError("gridstatus is not available")

    df: pd.DataFrame = ERCOT.get_fuel_mix("latest")

    # get_fuel_mix("latest") returns all rows for today; use the last row
    row = df.iloc[-1]

    mix = {col: float(row[col]) for col in df.columns if col != "Time"}

    wind = mix.get("Wind", 0.0)
    solar = mix.get("Solar", 0.0)
    total = sum(mix.values())
    renewable_pct = (wind + solar) / total if total > 0 else 0.0

    ts = row["Time"]
    if hasattr(ts, "isoformat"):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    return {
        "timestamp": ts_str,
        "renewable_pct": renewable_pct,
        "mix": mix,
    }


async def _notify_subscriber(sub: dict, renewable_pct: float, mix: dict) -> None:
    """Send an HTTP POST to the subscriber's callback URL."""
    payload = {
        "event": "threshold_crossed",
        "renewable_pct": renewable_pct,
        "threshold": sub["threshold"],
        "trigger_on_drop": sub["trigger_on_drop"],
        "mix": mix,
        "notified_at": datetime.now(timezone.utc).isoformat(),
    }
    client = _http_client or httpx.AsyncClient(timeout=10.0)
    try:
        resp = await client.post(str(sub["callback_url"]), json=payload)
        logger.info(
            "Notified subscriber %s → HTTP %s", sub["id"], resp.status_code
        )
    except Exception as exc:
        logger.warning("Failed to notify subscriber %s: %s", sub["id"], exc)


async def _poll_and_notify() -> None:
    """Background task: poll ERCOT every POLL_INTERVAL_SECONDS and fire webhooks."""
    # Tracks the last "condition met" state per subscriber to avoid repeat pings.
    prev_condition: dict[str, bool] = {}

    while True:
        try:
            logger.info("Polling ERCOT fuel mix …")
            data = await asyncio.to_thread(_fetch_fuel_mix)

            fuel_mix_cache.update(data)

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

                # Only notify when the condition transitions from False → True.
                if condition_met and not prev_condition.get(sub_id, False):
                    await _notify_subscriber(sub, renewable_pct, mix)

                prev_condition[sub_id] = condition_met

        except Exception as exc:
            logger.error("Error in polling loop: %s", exc)

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
        "your configured threshold. Shift your compute to when the wind blows!"
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
    """Register a webhook to be called when the renewable threshold is crossed."""
    if not (0.0 < req.threshold <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="threshold must be between 0 (exclusive) and 1 (inclusive).",
        )

    sub_id = str(uuid.uuid4())
    sub = {
        "id": sub_id,
        "callback_url": str(req.callback_url),
        "threshold": req.threshold,
        "trigger_on_drop": req.trigger_on_drop,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    subscribers[sub_id] = sub
    logger.info("New subscriber %s (threshold=%.0f%%)", sub_id, req.threshold * 100)
    return sub


@app.delete("/subscribe/{sub_id}", status_code=204)
async def unsubscribe(sub_id: str):
    """Remove a previously registered webhook subscription."""
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
    if fuel_mix_cache["data"] is None and fuel_mix_cache["renewable_pct"] is None:
        # Attempt a fresh fetch on first request
        try:
            data = await asyncio.to_thread(_fetch_fuel_mix)
            fuel_mix_cache.update(data)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to fetch ERCOT data: {exc}",
            )

    return {
        "timestamp": fuel_mix_cache.get("timestamp"),
        "renewable_pct": fuel_mix_cache.get("renewable_pct"),
        "mix": fuel_mix_cache.get("mix"),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
