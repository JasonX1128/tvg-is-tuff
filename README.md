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
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs available at `http://localhost:8000/docs`.

### Frontend

Open `frontend/index.html` in a browser (or serve it statically alongside the API).

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
