#!/usr/bin/env python3
"""持續採集運輸署空位 + 天文台天氣，寫入 Supabase。"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from google_areas import reverse_geocode_location_label  # noqa: E402
from parking_db import (  # noqa: E402
    get_carpark_rows,
    get_write_client,
    insert_vacancy_records,
    insert_weather_record,
    list_carparks_missing_google_area,
    list_park_ids,
    now_hk_iso,
    parse_api_timestamp,
    update_carpark_google_area,
    upsert_carparks,
    uses_service_role,
)

CARPARKS_STAMP_FILE = ROOT / ".carparks_last_update"
VACANCY_URL = "https://api.data.gov.hk/v1/carpark-info-vacancy"
WEATHER_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"


def ensure_google_areas(*, only_missing: bool = True, pause_sec: float = 0.05) -> int:
    client = get_write_client()
    if only_missing:
        rows = list_carparks_missing_google_area(client)
    else:
        rows = [
            r
            for r in (
                client.table("carparks")
                .select("park_id,latitude,longitude")
                .not_.is_("latitude", "null")
                .not_.is_("longitude", "null")
                .order("park_id")
                .execute()
                .data
                or []
            )
        ]
    updated = 0
    for i, row in enumerate(rows):
        try:
            label = reverse_geocode_location_label(
                float(row["latitude"]), float(row["longitude"])
            )
        except Exception:
            continue
        if not label:
            continue
        update_carpark_google_area(str(row["park_id"]), label, client)
        updated += 1
        if pause_sec > 0 and i + 1 < len(rows):
            time.sleep(pause_sec)
    return updated


def update_carparks() -> int:
    response = requests.get(
        VACANCY_URL, params={"data": "info", "lang": "zh_TW"}, timeout=30
    )
    if response.status_code != 200:
        raise RuntimeError(f"carparks API HTTP {response.status_code}")
    results = response.json().get("results") or []
    if not results:
        raise RuntimeError("carparks API 回傳空結果")

    client = get_write_client()
    park_ids = [
        str(cp.get("park_Id") or "").strip()
        for cp in results
        if str(cp.get("park_Id") or "").strip()
    ]
    existing = get_carpark_rows(park_ids, client)

    payload = []
    for cp in results:
        park_id = str(cp.get("park_Id") or "").strip()
        if not park_id:
            continue
        lat = cp.get("latitude")
        lon = cp.get("longitude")
        old = existing.get(park_id)
        clear_area = False
        google_area = None
        if old is None:
            clear_area = True
        elif old.get("latitude") != lat or old.get("longitude") != lon:
            clear_area = True
        else:
            google_area = old.get("google_area")
        row = {
            "park_id": park_id,
            "name": cp.get("name"),
            "district": cp.get("district"),
            "latitude": lat,
            "longitude": lon,
            "address": cp.get("displayAddress"),
            "google_area": google_area,
            "_clear_google_area": clear_area,
        }
        payload.append(row)

    n = upsert_carparks(payload, client)
    try:
        ensure_google_areas(only_missing=True)
    except Exception as e:
        print("google_area 更新略過:", e)
    return n


def maybe_update_carparks_monthly(target_ids: set[str]) -> set[str]:
    now_ym = datetime.now().strftime("%Y-%m")
    if CARPARKS_STAMP_FILE.exists():
        if CARPARKS_STAMP_FILE.read_text(encoding="utf-8").strip() == now_ym:
            return target_ids
    print(f"開始每月更新 carparks（{now_ym}）...")
    try:
        n = update_carparks()
        CARPARKS_STAMP_FILE.write_text(now_ym, encoding="utf-8")
        target_ids = list_park_ids()
        print(f"carparks 已更新：{n} 筆，目標車場 {len(target_ids)} 個")
    except Exception as e:
        print("carparks 每月更新失敗:", e)
    return target_ids


def collect_vacancy(target_ids: set[str]) -> None:
    try:
        response = requests.get(
            VACANCY_URL,
            params={
                "data": "vacancy",
                "lang": "zh_TW",
                "vehicleTypes": "privateCar",
            },
            timeout=30,
        )
        if response.status_code != 200:
            print(f"空位請求失敗 HTTP {response.status_code}")
            return

        collected_at = now_hk_iso()
        records = []
        for item in response.json().get("results") or []:
            park_id = str(item.get("park_Id"))
            if park_id not in target_ids:
                continue
            private = item.get("privateCar") or []
            if not private:
                continue
            v = private[0]
            records.append(
                {
                    "park_id": park_id,
                    "vacancy": v.get("vacancy"),
                    "vacancy_type": v.get("vacancy_type"),
                    "lastupdate": parse_api_timestamp(v.get("lastupdate")),
                    "collected_at": collected_at,
                }
            )

        if not records:
            print("沒有新資料")
            return

        insert_vacancy_records(records)
        print(f"成功寫入 Supabase {len(records)} 筆空位 ({collected_at})")
    except Exception as e:
        print("空位收集失敗:", e)


def collect_weather() -> None:
    try:
        resp = requests.get(
            WEATHER_URL,
            params={"dataType": "rhrread", "lang": "en"},
            timeout=30,
        )
        weather = resp.json()
        hko_temp = None
        for t in weather.get("temperature", {}).get("data", []):
            if "Observatory" in t.get("place", ""):
                hko_temp = t.get("value")
                break
        rainfall_list = weather.get("rainfall", {}).get("data", [])
        max_rain = max((r.get("max", 0) for r in rainfall_list), default=0)
        collected_at = now_hk_iso()
        insert_weather_record(
            {
                "collected_at": collected_at,
                "temperature": hko_temp,
                "max_rainfall": max_rain,
                "update_time": parse_api_timestamp(weather.get("updateTime")),
            }
        )
        print(f"天氣已寫入 Supabase ({collected_at})")
    except Exception as e:
        print("天氣收集失敗:", e)


def main() -> None:
    get_write_client()
    mode = "service_role" if uses_service_role() else "anon+RPC"
    target_ids = list_park_ids()
    if not target_ids:
        print("carparks 為空，先跑一次 update_carparks()…")
        update_carparks()
        target_ids = list_park_ids()
    print(
        f"開始持續收集 → Supabase（{mode}；每 10 分鐘；"
        f"目標車場 {len(target_ids)}；Ctrl+C 停止）"
    )
    while True:
        try:
            target_ids = maybe_update_carparks_monthly(target_ids)
            collect_vacancy(target_ids)
            collect_weather()
        except Exception as e:
            # 唔好因單次網絡／API 錯誤就退出迴圈
            print("本輪採集異常（稍後重試）:", e)
        time.sleep(600)


if __name__ == "__main__":
    main()
