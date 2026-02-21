#!/usr/bin/env python3
"""
migrate_to_supabase.py — One-time migration of ml_training_data.json → Supabase.

Uploads all 18K+ historical data points to the ercot_history table.
Safe to re-run — uses upsert (ON CONFLICT timestamp) so duplicates are
harmlessly updated.

Usage:
    python3 migrate_to_supabase.py
    python3 migrate_to_supabase.py --dry-run    # preview without writing
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).resolve().parent / ".env")

ML_DATA_PATH = Path(__file__).resolve().parent / "ml_training_data.json"
BATCH_SIZE = 500  # Supabase row limit per request


def main():
    parser = argparse.ArgumentParser(description="Migrate local JSON to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--file", default=str(ML_DATA_PATH), help="Path to ml_training_data.json")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("❌ Set SUPABASE_URL and SUPABASE_KEY in backend/.env")
        sys.exit(1)

    # Load local data
    data_path = Path(args.file)
    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        ml_data = json.load(f)

    points = ml_data.get("points", [])
    print(f"📂 Loaded {len(points)} points from {data_path.name}")

    if not points:
        print("⚠️  No points to migrate.")
        return

    # Validate a sample
    sample = points[0]
    required = {"timestamp", "renewable_pct", "low_carbon_pct", "mix"}
    missing = required - set(sample.keys())
    if missing:
        print(f"❌ Points are missing required fields: {missing}")
        sys.exit(1)

    print(f"   First: {sample['timestamp']}")
    print(f"   Last:  {points[-1]['timestamp']}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Batches: {(len(points) + BATCH_SIZE - 1) // BATCH_SIZE}")

    if args.dry_run:
        print("\n🔍 Dry run — no data written.")
        return

    # Connect to Supabase
    sb = create_client(url, key)
    print()

    uploaded = 0
    errors = 0
    total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(points), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        chunk = points[i : i + BATCH_SIZE]

        rows = []
        for p in chunk:
            rows.append({
                "timestamp": p["timestamp"],
                "renewable_pct": round(p["renewable_pct"], 6),
                "low_carbon_pct": round(p["low_carbon_pct"], 6),
                "mix": p.get("mix", {}),
            })

        try:
            sb.table("ercot_history").upsert(rows, on_conflict="timestamp").execute()
            uploaded += len(rows)
            print(f"  ⏳ Batch {batch_num}/{total_batches}: {len(rows)} rows → ✅ ({uploaded} total)")
        except Exception as exc:
            errors += 1
            print(f"  ⏳ Batch {batch_num}/{total_batches}: ❌ {str(exc)[:120]}")

    print(f"\n{'─' * 50}")
    print(f"✅ Uploaded: {uploaded} rows")
    if errors:
        print(f"❌ Failed batches: {errors}")
    print(f"🎉 Migration complete!")


if __name__ == "__main__":
    main()
