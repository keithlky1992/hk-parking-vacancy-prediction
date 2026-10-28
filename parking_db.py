"""Supabase access for parking vacancy data (read: anon; write: service role or ingest RPCs)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

HK_TZ = ZoneInfo("Asia/Hong_Kong")
PAGE_SIZE = 1000


def get_secret(name: str) -> str:
    """Read from env (.env / Streamlit Cloud) or st.secrets."""
    val = (os.getenv(name) or "").strip()
    if val:
        return val
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""


def _supabase_url() -> str:
    return get_secret("SUPABASE_URL").rstrip("/")


def _supabase_anon_key() -> str:
    return get_secret("SUPABASE_ANON_KEY")


def _supabase_service_role_key() -> str:
    return get_secret("SUPABASE_SERVICE_ROLE_KEY")


def _require_url_and_key(key: str, *, what: str) -> tuple[str, str]:
    url = _supabase_url()
    if not url or not key:
        raise RuntimeError(
            f"缺少 Supabase 設定（{what}）。請在 .env 或 Streamlit Secrets 設定 "
            "SUPABASE_URL 以及對應的 key。"
        )
    return url, key


def get_read_client() -> Client:
    url, key = _require_url_and_key(_supabase_anon_key(), what="讀取 / anon")
    return create_client(url, key)


def get_write_client() -> Client:
    """Prefer service_role; fall back to anon + SECURITY DEFINER RPCs."""
    service = _supabase_service_role_key()
    if service:
        url, key = _require_url_and_key(service, what="寫入 / service_role")
        return create_client(url, key)
    return get_read_client()


def uses_service_role() -> bool:
    return bool(_supabase_service_role_key())


def _fetch_all(
    client: Client,
    table: str,
    columns: str,
    *,
    order_col: str = "id",
    filters: Optional[list[tuple[str, str, Any]]] = None,
) -> list[dict[str, Any]]:
    """filters: [(op, column, value), ...]；op 支援 eq / gte / lte / in_。"""
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        q = client.table(table).select(columns)
        for op, col, val in filters or []:
            if op == "eq":
                q = q.eq(col, val)
            elif op == "gte":
                q = q.gte(col, val)
            elif op == "lte":
                q = q.lte(col, val)
            elif op == "in_":
                q = q.in_(col, list(val))
            else:
                raise ValueError(f"unsupported filter op: {op}")
        resp = q.order(order_col).range(start, start + PAGE_SIZE - 1).execute()
        page = resp.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def load_carparks_frame(client: Optional[Client] = None) -> pd.DataFrame:
    """只載入車場靜態資料（快；畀地圖／附近篩選用）。"""
    client = client or get_read_client()
    carparks = _fetch_all(
        client,
        "carparks",
        "park_id,name,district,google_area,latitude,longitude",
        order_col="park_id",
    )
    if not carparks:
        return pd.DataFrame(
            columns=[
                "park_id",
                "name",
                "district",
                "google_area",
                "latitude",
                "longitude",
            ]
        )
    return pd.DataFrame(carparks)


def to_hk_naive(series: pd.Series) -> pd.Series:
    """timestamptz / ISO → 香港時間、無 tz（對齊舊 SQLite 行為）。"""
    # 庫內可能混有 "…+00" 與完整 ISO（含微秒）；用 mixed 避免整批 coerce 成 NaT
    ts = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    return ts.dt.tz_convert(HK_TZ).dt.tz_localize(None)


def now_hk_iso() -> str:
    return datetime.now(HK_TZ).isoformat()


def parse_api_timestamp(value: Any) -> Optional[str]:
    """政府 API / 本機字串 → ISO timestamptz（當香港時間）。"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=HK_TZ)
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    if "T" in text or text.endswith("Z") or "+" in text[10:]:
        return text
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=HK_TZ)
        return dt.isoformat()
    except ValueError:
        return text


def load_vacancy_frame(
    client: Optional[Client] = None,
    *,
    park_ids: Optional[Iterable[str]] = None,
    since: Optional[datetime | str] = None,
) -> pd.DataFrame:
    """
    等同舊 SQL：
    SELECT v.park_id, v.vacancy, v.collected_at, c.name, c.district,
           c.google_area, c.latitude, c.longitude
    FROM vacancy_records v LEFT JOIN carparks c ...

    park_ids：只拉指定車場（附近查詢快很多）。
    since：只拉 collected_at >= since 的紀錄。
    """
    client = client or get_read_client()
    id_list = [str(p) for p in park_ids] if park_ids is not None else None
    empty = pd.DataFrame(
        columns=[
            "park_id",
            "vacancy",
            "collected_at",
            "name",
            "district",
            "google_area",
            "latitude",
            "longitude",
        ]
    )

    if id_list is not None and not id_list:
        return empty

    if id_list is not None:
        parks_rows: list[dict[str, Any]] = []
        for i in range(0, len(id_list), PAGE_SIZE):
            batch = id_list[i : i + PAGE_SIZE]
            parks_rows.extend(
                _fetch_all(
                    client,
                    "carparks",
                    "park_id,name,district,google_area,latitude,longitude",
                    order_col="park_id",
                    filters=[("in_", "park_id", batch)],
                )
            )
    else:
        parks_rows = _fetch_all(
            client,
            "carparks",
            "park_id,name,district,google_area,latitude,longitude",
            order_col="park_id",
        )

    since_iso: Optional[str] = None
    if since is not None:
        if isinstance(since, datetime):
            since_iso = since.isoformat()
        else:
            since_iso = str(since)

    vacancies: list[dict[str, Any]] = []
    if id_list is not None:
        for i in range(0, len(id_list), PAGE_SIZE):
            batch = id_list[i : i + PAGE_SIZE]
            filters: list[tuple[str, str, Any]] = [("in_", "park_id", batch)]
            if since_iso:
                filters.append(("gte", "collected_at", since_iso))
            vacancies.extend(
                _fetch_all(
                    client,
                    "vacancy_records",
                    "id,park_id,vacancy,collected_at",
                    order_col="id",
                    filters=filters,
                )
            )
    else:
        filters = []
        if since_iso:
            filters.append(("gte", "collected_at", since_iso))
        vacancies = _fetch_all(
            client,
            "vacancy_records",
            "id,park_id,vacancy,collected_at",
            order_col="id",
            filters=filters or None,
        )

    if not vacancies:
        return empty

    parks_df = pd.DataFrame(parks_rows)
    vac_df = pd.DataFrame(vacancies).drop(columns=["id"], errors="ignore")
    df = vac_df.merge(parks_df, on="park_id", how="left")
    df["collected_at"] = to_hk_naive(df["collected_at"])
    return df


def list_park_ids(client: Optional[Client] = None) -> set[str]:
    client = client or get_read_client()
    rows = _fetch_all(client, "carparks", "park_id", order_col="park_id")
    return {str(r["park_id"]) for r in rows if r.get("park_id")}


def get_carpark_rows(
    park_ids: Iterable[str], client: Optional[Client] = None
) -> dict[str, dict[str, Any]]:
    client = client or get_write_client()
    ids = [str(p) for p in park_ids]
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(ids), PAGE_SIZE):
        batch = ids[i : i + PAGE_SIZE]
        resp = (
            client.table("carparks")
            .select("park_id,latitude,longitude,google_area")
            .in_("park_id", batch)
            .execute()
        )
        for row in resp.data or []:
            out[str(row["park_id"])] = row
    return out


def upsert_carparks(rows: Iterable[dict[str, Any]], client: Optional[Client] = None) -> int:
    client = client or get_write_client()
    payload = list(rows)
    if not payload:
        return 0

    if uses_service_role():
        clean = []
        for row in payload:
            item = {
                k: v
                for k, v in row.items()
                if not k.startswith("_")
            }
            if row.get("_clear_google_area"):
                item["google_area"] = None
            clean.append(item)
        for i in range(0, len(clean), PAGE_SIZE):
            client.table("carparks").upsert(
                clean[i : i + PAGE_SIZE], on_conflict="park_id"
            ).execute()
        return len(clean)

    for row in payload:
        client.rpc(
            "upsert_carpark",
            {
                "p_park_id": row["park_id"],
                "p_name": row.get("name"),
                "p_district": row.get("district"),
                "p_latitude": row.get("latitude"),
                "p_longitude": row.get("longitude"),
                "p_address": row.get("address"),
                "p_google_area": row.get("google_area"),
                "p_clear_google_area": bool(row.get("_clear_google_area")),
            },
        ).execute()
    return len(payload)


def update_carpark_google_area(
    park_id: str, label: str, client: Optional[Client] = None
) -> None:
    client = client or get_write_client()
    if uses_service_role():
        (
            client.table("carparks")
            .update({"google_area": label})
            .eq("park_id", park_id)
            .execute()
        )
        return
    client.rpc(
        "set_carpark_google_area",
        {"p_park_id": park_id, "p_label": label},
    ).execute()


def list_carparks_missing_google_area(
    client: Optional[Client] = None,
    *,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """未填 google_area，或只有非中文名（需重抓優先中文）嘅車場。"""
    from google_areas import needs_google_area_refresh

    client = client or get_write_client()
    q = (
        client.table("carparks")
        .select("park_id,latitude,longitude,google_area")
        .not_.is_("latitude", "null")
        .not_.is_("longitude", "null")
        .order("park_id")
    )
    resp = q.execute()
    rows = [
        {"park_id": r["park_id"], "latitude": r["latitude"], "longitude": r["longitude"]}
        for r in (resp.data or [])
        if needs_google_area_refresh(r.get("google_area"))
    ]
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def insert_vacancy_records(
    records: list[dict[str, Any]], client: Optional[Client] = None
) -> int:
    client = client or get_write_client()
    if not records:
        return 0
    if uses_service_role():
        for i in range(0, len(records), PAGE_SIZE):
            client.table("vacancy_records").insert(records[i : i + PAGE_SIZE]).execute()
        return len(records)

    total = 0
    for i in range(0, len(records), PAGE_SIZE):
        batch = records[i : i + PAGE_SIZE]
        resp = client.rpc("ingest_vacancy_records", {"rows": batch}).execute()
        total += int(resp.data or len(batch))
    return total


def insert_weather_record(record: dict[str, Any], client: Optional[Client] = None) -> None:
    client = client or get_write_client()
    if uses_service_role():
        client.table("weather_records").insert(record).execute()
        return
    client.rpc(
        "ingest_weather_record",
        {
            "p_collected_at": record.get("collected_at"),
            "p_temperature": record.get("temperature"),
            "p_max_rainfall": record.get("max_rainfall"),
            "p_update_time": record.get("update_time"),
        },
    ).execute()
