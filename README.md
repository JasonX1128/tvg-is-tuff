# Carbon-Aware Trigger

> **Shift your compute to when the wind blows in Texas.**

A webhook-based API that notifies developers when the ERCOT (Texas) grid reaches a configurable renewable energy threshold. Schedule your ML training, backups, or other heavy jobs to run during green windows.

---

## Architecture

```
frontend/        ← Static landing page with live ERCOT fuel-mix chart
backend/
  main.py        ← FastAPI app + background polling worker
  requirements.txt
```

### How it works

1. **Subscribe** — register your `callback_url` and a renewable `threshold` (0–100 %).  
   Optionally set `trigger_on_drop=true` to be notified when renewables *drop* below the threshold.
2. **Background worker** — polls the ERCOT fuel mix every 5 minutes via [gridstatus](https://github.com/kmax12/gridstatus).
3. **Ping** — when `(Wind + Solar) / Total ≥ threshold`, your endpoint receives an HTTP POST with the current mix data.
4. **Execute** — your server starts the heavy background job.

---

## Quick start

### Backend

```bash
cd backend
python3 -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API docs at `http://localhost:8000/docs`.  
**Dashboard** at `http://localhost:8000/` — the frontend is served automatically.

### Frontend (standalone, optional)

Open `frontend/index.html` directly in a browser for file-based dev (auto-detects `localhost:8000` as API).

---

## API Reference

### `POST /subscribe`

Register a webhook.

```json
{
  "callback_url": "https://your-server.example/webhook",
  "threshold": 0.40,
  "trigger_on_drop": false
}
```

| Field | Type | Description |
|---|---|---|
| `callback_url` | `string` | URL that will receive the POST when the threshold is crossed |
| `threshold` | `float` | Renewable fraction (0.0 – 1.0), e.g. `0.40` = 40 % |
| `trigger_on_drop` | `bool` | If `true`, ping when renewables drop *below* threshold (default `false`) |

Returns the created subscription with its `id`.

### `DELETE /subscribe/{id}`

Remove a subscription.

### `GET /subscribers`

List all active subscriptions.

### `GET /fuel-mix`

Returns the latest ERCOT fuel mix, including `renewable_pct` and per-fuel-type MW values.

### `GET /health`

Health check.

### `GET /co2`

Estimated grid carbon intensity (gCO₂eq/kWh) based on current fuel mix, with per-fuel breakdown using IPCC 2014 median lifecycle emission factors.

### `GET /stream`

Server-Sent Events (SSE) endpoint. Connect via `EventSource` for real-time fuel mix updates pushed to the browser without polling.

### `GET /stats`

Dashboard KPIs including current renewable %, 24h average, 24h peak, subscriber count, notification count, **Green Score** (A–F letter grade), and **CO₂ intensity**.

---

## Features

| Feature | Description |
|---------|-------------|
| **Webhook Subscriptions** | Register endpoints, get pinged on threshold crossings |
| **SSE Live Stream** | Real-time updates via `EventSource` — no polling needed |
| **Green Score (A–F)** | At-a-glance letter grade for grid cleanliness |
| **CO₂ Intensity** | Estimated gCO₂/kWh using IPCC emission factors |
| **Carbon Savings Estimator** | Frontend widget — see how much CO₂ you save by running now |
| **Donut + Bar Charts** | Plotly-powered fuel mix visualizations |
| **Browser Notifications** | Desktop alerts when the Green Score changes |
| **Animated Counters** | Smooth number transitions on stat cards |
| **Single-Server Hosting** | FastAPI serves both the API and the frontend SPA |

---

## Webhook payload

```json
{
  "event": "threshold_crossed",
  "renewable_pct": 0.47,
  "threshold": 0.40,
  "trigger_on_drop": false,
  "mix": {
    "Wind": 12500,
    "Solar": 4200,
    "Natural Gas": 18000,
    "Nuclear": 5100,
    "Coal and Lignite": 2800,
    "Hydro": 120,
    "Power Storage": -300,
    "Other": 80
  },
  "notified_at": "2024-06-01T14:35:00+00:00"
}
```
