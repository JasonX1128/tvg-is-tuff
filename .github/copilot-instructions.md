# Copilot Instructions — Carbon-Aware Trigger
always use python3, not python, in all terminal commands.
## Architecture Overview
Two-component app: a **FastAPI backend** (`backend/main.py`) and a **single-file static frontend** (`frontend/index.html`, no build step).

```
frontend/index.html   ← self-contained SPA; Plotly + Inter font loaded via CDN; no bundler
backend/main.py       ← FastAPI + background polling loop + in-memory state
backend/requirements.txt
```

## Data Flow
1. **Polling loop** (`_poll_and_notify`): background `asyncio.Task` launched at lifespan start; seeds today's history via `ERCOT.get_fuel_mix("today")` then polls `ERCOT.get_fuel_mix("latest")` (blocking, run via `asyncio.to_thread`) every 300 s.
2. **Renewable %** = `(Wind + Solar) / Total` (positive values only) from the last DataFrame row.
3. **Low-carbon %** = `(Wind + Solar + Nuclear + Hydro) / Total`.
4. **Edge-triggered notifications**: `prev_condition: dict[str, bool]` tracks per-subscriber state; webhooks fire **only on False → True transitions** to avoid repeat pings.
5. **Cache**: `fuel_mix_cache` (module-level dict) is the shared state between the loop and `GET /fuel-mix`; on the first `/fuel-mix` request it also triggers an immediate fetch if the cache is empty.
6. **History**: `deque(maxlen=288)` stores up to 24 h of data points for the trend chart.
7. **Notification log**: `deque(maxlen=200)` stores recent webhook delivery attempts with status codes/errors.
8. **CO₂ intensity**: estimated via IPCC 2014 median lifecycle emission factors (gCO₂eq/kWh) per fuel type. Weighted average across the current mix.
9. **Green Score**: letter grade A–F based on renewable % (A ≥ 50%, B ≥ 40%, C ≥ 30%, D ≥ 20%, F < 20%).
10. **SSE stream** (`/stream`): yields `data: {json}\n\n` events every 5 s when the cache timestamp changes; clients auto-reconnect via `EventSource`.
11. **Frontend served by FastAPI**: `GET /` returns `frontend/index.html` via `HTMLResponse`; the frontend auto-detects same-origin vs. `file://` for the API base URL.

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/subscribe` | Register webhook (label, callback_url, threshold 0–1, trigger_on_drop) |
| DELETE | `/subscribe/{sub_id}` | Remove a subscription |
| GET | `/subscribers` | List all active subscriptions |
| GET | `/fuel-mix` | Latest ERCOT fuel mix + renewable_pct + low_carbon_pct |
| GET | `/history?limit=288` | Time-series of renewable/low-carbon % (up to 24 h) |
| GET | `/stats` | Dashboard KPIs (current %, 24h avg, peak, subscriber count, notification count) |
| GET | `/notifications?limit=50` | Recent webhook delivery log (newest first) |
| GET | `/fuel-colors` | Canonical fuel → hex color mapping |
| GET | `/co2` | Estimated grid carbon intensity (gCO₂/kWh) with per-fuel breakdown |
| GET | `/stream` | SSE (Server-Sent Events) real-time feed of fuel mix updates |
| GET | `/health` | System health check |
| GET | `/` | Serves the frontend SPA (index.html) |

## Threshold Representation
- **API / storage**: fraction `0.0–1.0` (e.g. `0.40`)
- **UI display / input**: percentage `0–100` — frontend divides by 100 on `POST /subscribe`, multiplies by 100 on render

## State & Persistence
- Subscribers are stored in the module-level `subscribers: dict[str, dict]` — **not persisted**; restarts wipe all subscriptions.
- The shared `httpx.AsyncClient` (`_http_client`) is created at lifespan startup and closed at shutdown; use it for all outbound webhook POSTs.

## Webhook Payload Shape
```json
{
  "event": "threshold_crossed",
  "renewable_pct": 0.47,
  "threshold": 0.40,
  "trigger_on_drop": false,
  "mix": { "Wind": 12000, "Solar": 4000, ... },
  "notified_at": "<ISO-8601 UTC>"
}
```

## Developer Workflows

### Run the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload   # API docs at http://localhost:8000/docs
```

### Run the frontend locally
Open `frontend/index.html` directly in a browser, **but first** change:
```js
const API = 'http://localhost:8000';  // line near bottom of index.html
```
In production, `API` is `''` (same-origin assumed).

### gridstatus unavailability
`import gridstatus` is wrapped in `try/except`; if unavailable, `ERCOT = None` and the app still starts, but `/fuel-mix` returns HTTP 503.

## Key Conventions
- New API routes go in `backend/main.py`; Pydantic models live in the same file, grouped before routes.
- Frontend fetches use the `API` constant prefix — never hardcode `localhost` in `fetch()` calls.
- CORS is currently `allow_origins=["*"]`; keep this for local dev but narrow before any production deployment.
- No test suite exists yet; validate behaviour via `/docs` (Swagger UI) and browser DevTools.
