"""用 Google Maps reverse geocode 為停車場填寫短地區名（快取至 DB）。"""

from __future__ import annotations

import os
import sqlite3
import time

import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

_REVERSE_GEOCODE_SKIP_NAMES = {
    "香港",
    "Hong Kong",
    "HK",
    "中華人民共和國",
    "China",
    "People's Republic of China",
}
_REVERSE_GEOCODE_TYPE_PRIORITY = (
    "neighborhood",
    "sublocality_level_2",
    "sublocality_level_1",
    "sublocality",
    "locality",
    "administrative_area_level_2",
    "administrative_area_level_1",
)


def label_from_google_address_components(results: list) -> str | None:
    """從 Google reverse geocode 結果抽出短地區名。"""
    components: list[tuple[str, set[str]]] = []
    for result in results or []:
        for comp in result.get("address_components") or []:
            name = (comp.get("long_name") or comp.get("short_name") or "").strip()
            if not name or name in _REVERSE_GEOCODE_SKIP_NAMES:
                continue
            types = set(comp.get("types") or [])
            components.append((name, types))

    for preferred in _REVERSE_GEOCODE_TYPE_PRIORITY:
        for name, types in components:
            if preferred in types:
                return name

    top = (results or [{}])[0]
    formatted = (top.get("formatted_address") or "").strip()
    if not formatted:
        return None
    short = formatted.replace(", Hong Kong", "").replace("香港", "").strip(" ,")
    return short or formatted


def reverse_geocode_location_label(lat: float, lon: float) -> str | None:
    """
    用 Google Geocoding reverse geocode 把 GPS 轉成地區名。
    無 API key 或 API 異常時 raise RuntimeError。
    """
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("missing_api_key")

    params = {
        "latlng": f"{lat},{lon}",
        "language": "zh-HK",
        "key": GOOGLE_MAPS_API_KEY,
    }
    response = requests.get(GOOGLE_GEOCODE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if status == "ZERO_RESULTS":
        return None
    if status == "OVER_QUERY_LIMIT":
        raise RuntimeError("geocode_status:OVER_QUERY_LIMIT")
    if status != "OK" or not data.get("results"):
        raise RuntimeError(f"geocode_status:{status}")
    return label_from_google_address_components(data["results"])


def ensure_carpark_google_areas(
    db_path: str = "parking.db",
    *,
    only_missing: bool = True,
    limit: int | None = None,
    pause_sec: float = 0.05,
) -> int:
    """
    為車場寫入 google_area（Google reverse geocode 短名）。
    回傳成功更新列數。無 API key 時 raise RuntimeError。
    """
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("missing_api_key")

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(carparks)")}
        if "google_area" not in cols:
            conn.execute("ALTER TABLE carparks ADD COLUMN google_area TEXT")
            conn.commit()

        if only_missing:
            where = (
                "latitude IS NOT NULL AND longitude IS NOT NULL "
                "AND (google_area IS NULL OR TRIM(google_area) = '')"
            )
        else:
            where = "latitude IS NOT NULL AND longitude IS NOT NULL"

        sql = (
            f"SELECT park_id, latitude, longitude FROM carparks WHERE {where} "
            "ORDER BY park_id"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()

        updated = 0
        for i, (park_id, lat, lon) in enumerate(rows):
            try:
                label = reverse_geocode_location_label(float(lat), float(lon))
            except RuntimeError as exc:
                if "OVER_QUERY_LIMIT" in str(exc):
                    break
                continue
            except requests.RequestException:
                continue
            if not label:
                continue
            conn.execute(
                "UPDATE carparks SET google_area = ? WHERE park_id = ?",
                (label, park_id),
            )
            updated += 1
            if pause_sec > 0 and i + 1 < len(rows):
                time.sleep(pause_sec)
        conn.commit()
        return updated
    finally:
        conn.close()
