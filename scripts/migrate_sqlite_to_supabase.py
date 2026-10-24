#!/usr/bin/env python3
"""One-shot: copy parking.db tables into Supabase via PostgREST."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "parking.db"
BATCH = 500

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_ANON_KEY", "")


def to_ts(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    # SQLite stores local wall time without TZ; treat as HKT.
    if "T" not in value and "+" not in value and not value.endswith("Z"):
        return f"{value.replace(' ', 'T')}+08:00"
    return value


def post_rows(table: str, rows: list[dict]) -> None:
    if not rows:
        return
    body = json.dumps(rows).encode("utf-8")
    req = urllib.request.Request(
        f"{URL}/rest/v1/{table}",
        data=body,
        method="POST",
        headers={
            "apikey": KEY,
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"{table}: HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{table} insert failed ({e.code}): {detail}") from e


def batched(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def main() -> int:
    if not URL or not KEY:
        print("Set SUPABASE_URL and SUPABASE_ANON_KEY", file=sys.stderr)
        return 1
    if not DB_PATH.exists():
        print(f"Missing {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    carparks = [
        {
            "park_id": r["park_id"],
            "name": r["name"],
            "district": r["district"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "address": r["address"],
            "sub_district": r["sub_district"],
            "google_area": r["google_area"],
        }
        for r in conn.execute("SELECT * FROM carparks")
        if r["park_id"]
    ]
    print(f"carparks: {len(carparks)}")
    for batch in batched(carparks, BATCH):
        post_rows("carparks", batch)

    weather = [
        {
            "collected_at": to_ts(r["collected_at"]),
            "temperature": r["temperature"],
            "max_rainfall": r["max_rainfall"],
            "update_time": to_ts(r["update_time"]),
        }
        for r in conn.execute("SELECT * FROM weather_records")
    ]
    print(f"weather_records: {len(weather)}")
    for batch in batched(weather, BATCH):
        post_rows("weather_records", batch)

    vacancy = [
        {
            "park_id": r["park_id"],
            "vacancy": r["vacancy"],
            "vacancy_type": r["vacancy_type"],
            "lastupdate": to_ts(r["lastupdate"]),
            "collected_at": to_ts(r["collected_at"]),
        }
        for r in conn.execute(
            "SELECT park_id, vacancy, vacancy_type, lastupdate, collected_at "
            "FROM vacancy_records"
        )
    ]
    print(f"vacancy_records: {len(vacancy)}")
    for i, batch in enumerate(batched(vacancy, BATCH), start=1):
        post_rows("vacancy_records", batch)
        if i % 10 == 0:
            print(f"  … {min(i * BATCH, len(vacancy))}/{len(vacancy)}")

    conn.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
