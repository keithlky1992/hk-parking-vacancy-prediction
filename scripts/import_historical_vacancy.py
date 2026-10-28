#!/usr/bin/env python3
"""
從 DATA.GOV.HK historical archive 匯入 TD 停車場空位歷史。

來源：https://resource.data.one.gov.hk/td/carpark/vacancy_all.json
（約每 15 分鐘一版；可用月度／日度 ZIP）

例子：
  # 只統計、唔寫庫（建議先跑）
  python scripts/import_historical_vacancy.py --start 2025-10-01 --end 2025-10-31 --dry-run

  # 試跑一日並寫入
  python scripts/import_historical_vacancy.py --start 2025-10-28 --end 2025-10-28 --write

  # 匯入一整個月（已存在同區間會先刪再寫）
  python scripts/import_historical_vacancy.py --month 2025-10 --write --replace
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from parking_db import (  # noqa: E402
    delete_vacancy_records_between,
    insert_vacancy_records,
    list_park_ids,
    parse_api_timestamp,
    uses_service_role,
)

HK_TZ = ZoneInfo("Asia/Hong_Kong")
TD_VACANCY_URL = "https://resource.data.one.gov.hk/td/carpark/vacancy_all.json"
ARCHIVE_LIST = "https://app.data.gov.hk/v1/historical-archive/list-file-versions"
ARCHIVE_GET = "https://app.data.gov.hk/v1/historical-archive/get-file"
CACHE_DIR = ROOT / "data" / "historical"
TS_IN_NAME = re.compile(r"(?<!\d)(\d{8})-(\d{4})(?!\d)")
BATCH_SIZE = 1000


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def parse_ymd(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def month_bounds(yyyy_mm: str) -> tuple[date, date]:
    y, m = map(int, yyyy_mm.split("-"))
    return date(y, m, 1), date(y, m, monthrange(y, m)[1])


def daterange(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def list_archive_packages(day: date) -> list[dict]:
    """回傳該日可下載嘅 ZIP（優先 period=M，否則 period=D）。"""
    params = {
        "url": TD_VACANCY_URL,
        "start": _ymd(day),
        "end": _ymd(day),
    }
    try:
        resp = requests.get(ARCHIVE_LIST, params=params, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  警告：無法列出 {_ymd(day)} 封存（{e}）")
        return []
    files = resp.json().get("data-files") or []
    monthly = [f for f in files if f.get("period") == "M"]
    daily = [f for f in files if f.get("period") == "D"]
    return monthly or daily


def packages_for_range(start: date, end: date) -> list[dict]:
    """
    以每月 1 號探測月度 ZIP；若無則改用日度 ZIP。
    同一 filename 只保留一次。
    """
    seen: set[str] = set()
    out: list[dict] = []
    # Probe each month start in range + each day that might need daily zips
    probe_days: list[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        probe_days.append(date(y, m, 1))
        m += 1
        if m == 13:
            y, m = y + 1, 1

    monthly_covered: set[tuple[int, int]] = set()
    for day in probe_days:
        for pkg in list_archive_packages(day):
            if pkg.get("period") != "M":
                continue
            key = str(pkg.get("filename") or pkg.get("timestamp"))
            if key in seen:
                continue
            seen.add(key)
            out.append(pkg)
            # timestamp for monthly is YYYYMM01
            ts = str(pkg.get("timestamp") or "")
            if len(ts) >= 6:
                monthly_covered.add((int(ts[:4]), int(ts[4:6])))

    for day in daterange(start, end):
        if (day.year, day.month) in monthly_covered:
            continue
        for pkg in list_archive_packages(day):
            if pkg.get("period") != "D":
                continue
            key = str(pkg.get("filename") or pkg.get("timestamp"))
            if key in seen:
                continue
            seen.add(key)
            out.append(pkg)
    return out


def cache_path_for(pkg: dict) -> Path:
    name = str(pkg.get("filename") or f"{pkg.get('timestamp')}.zip")
    # filename 可能好長；用 timestamp + period 做短名
    period = pkg.get("period") or "X"
    ts = pkg.get("timestamp") or "unknown"
    safe = re.sub(r"[^\w.\-]+", "_", f"{period}_{ts}_{name}")[-180:]
    return CACHE_DIR / safe


def download_package(pkg: dict, *, force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(pkg)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    ts = str(pkg.get("timestamp") or "")
    if not ts:
        raise ValueError(f"package 缺 timestamp: {pkg}")
    params = {"url": TD_VACANCY_URL, "time": ts, "format": "zip"}
    print(f"下載 ZIP {ts} ({pkg.get('period')}) …")
    resp = requests.get(ARCHIVE_GET, params=params, timeout=300)
    resp.raise_for_status()
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "zip" not in ctype and not resp.content.startswith(b"PK"):
        raise RuntimeError(
            f"下載失敗（非 zip）：HTTP {resp.status_code} {ctype} {resp.text[:200]}"
        )
    path.write_bytes(resp.content)
    print(f"  已存 {path} ({len(resp.content)} bytes)")
    return path


def snapshot_ts_from_name(name: str) -> datetime | None:
    m = TS_IN_NAME.search(name)
    if not m:
        return None
    day, hm = m.group(1), m.group(2)
    return datetime.strptime(f"{day}{hm}", "%Y%m%d%H%M").replace(tzinfo=HK_TZ)


def iter_zip_snapshots(
    zip_path: Path,
) -> Iterator[tuple[datetime, dict]]:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            ts = snapshot_ts_from_name(name)
            if ts is None:
                continue
            try:
                raw = zf.read(name).decode("utf-8-sig")
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as e:
                print(f"  略過損毀快照 {name[-60:]}: {e}")
                continue
            if not isinstance(payload, dict):
                print(f"  略過非物件 JSON {name[-60:]}")
                continue
            yield ts, payload


def extract_private_car_hourly(
    payload: dict,
    *,
    collected_at: datetime,
    allowed_park_ids: set[str] | None,
) -> list[dict]:
    """TD vacancy_all → vacancy_records rows（只取 type=P、HOURLY）。"""
    rows: list[dict] = []
    collected_iso = collected_at.isoformat()
    for park in payload.get("car_park") or []:
        park_id = str(park.get("park_id") or "").strip()
        if not park_id:
            continue
        if allowed_park_ids is not None and park_id not in allowed_park_ids:
            continue
        for vt in park.get("vehicle_type") or []:
            if vt.get("type") != "P":
                continue
            picked = None
            for sc in vt.get("service_category") or []:
                if sc.get("category") == "HOURLY":
                    picked = sc
                    break
            if picked is None and (vt.get("service_category") or []):
                picked = vt["service_category"][0]
            if not picked:
                continue
            vacancy = picked.get("vacancy")
            try:
                vacancy_i = int(vacancy)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "park_id": park_id,
                    "vacancy": vacancy_i,
                    "vacancy_type": picked.get("vacancy_type"),
                    "lastupdate": parse_api_timestamp(picked.get("lastupdate")),
                    "collected_at": collected_iso,
                }
            )
            break
    return rows


def filter_snapshots(
    snapshots: Iterable[tuple[datetime, dict]],
    start: date,
    end: date,
) -> Iterator[tuple[datetime, dict]]:
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=HK_TZ)
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=HK_TZ)
    for ts, payload in snapshots:
        if start_dt <= ts < end_dt:
            yield ts, payload


def run_import(
    *,
    start: date,
    end: date,
    write: bool,
    replace: bool,
    max_files: int | None,
    force_download: bool,
) -> None:
    if write and not uses_service_role():
        raise SystemExit("寫入需要 SUPABASE_SERVICE_ROLE_KEY（.env）。")

    # 一律對齊 carparks，避免 FK 失敗；dry-run 都顯示覆蓋率
    allowed_set = set(list_park_ids())
    print(f"carparks 白名單：{len(allowed_set)} 個 park_id")
    print(f"範圍：{start} → {end}（香港時間）")

    packages = packages_for_range(start, end)
    if not packages:
        raise SystemExit("搵唔到歷史 ZIP，請檢查日期範圍。")
    print(f"將使用 {len(packages)} 個 ZIP 包")

    if write and replace:
        start_iso = datetime.combine(start, datetime.min.time(), tzinfo=HK_TZ).isoformat()
        end_iso = datetime.combine(
            end + timedelta(days=1), datetime.min.time(), tzinfo=HK_TZ
        ).isoformat()
        deleted = delete_vacancy_records_between(start_iso, end_iso)
        print(f"--replace：已刪除區間內 {deleted} 筆")

    stats = {
        "snapshots": 0,
        "rows": 0,
        "valid": 0,
        "skipped_unknown_park": 0,
        "written": 0,
        "parks_seen": set(),
        "parks_matched": set(),
    }
    pending: list[dict] = []

    def flush() -> None:
        if not write or not pending:
            pending.clear()
            return
        n = insert_vacancy_records(pending)
        stats["written"] += n
        pending.clear()
        print(f"  已寫入累計 {stats['written']} 筆 …")

    stop = False
    for pkg in packages:
        if stop:
            break
        path = download_package(pkg, force=force_download)
        for ts, payload in filter_snapshots(iter_zip_snapshots(path), start, end):
            if max_files is not None and stats["snapshots"] >= max_files:
                stop = True
                break
            stats["snapshots"] += 1
            all_parks = {
                str(p.get("park_id") or "").strip()
                for p in (payload.get("car_park") or [])
                if p.get("park_id")
            }
            stats["parks_seen"].update(all_parks)
            matched = all_parks & allowed_set
            stats["parks_matched"].update(matched)
            stats["skipped_unknown_park"] += len(all_parks - allowed_set)

            rows = extract_private_car_hourly(
                payload, collected_at=ts, allowed_park_ids=allowed_set
            )
            stats["rows"] += len(rows)
            stats["valid"] += sum(1 for r in rows if r["vacancy"] >= 0)
            if write:
                pending.extend(rows)
                if len(pending) >= BATCH_SIZE:
                    flush()
            if stats["snapshots"] % 50 == 0:
                print(
                    f"  已處理 {stats['snapshots']} 個快照，"
                    f"列 {stats['rows']}（有效 {stats['valid']}）"
                )

    flush()

    print("---")
    print(f"快照：{stats['snapshots']}")
    print(f"可寫入列（白名單私家車）：{stats['rows']}（有效 vacancy≥0：{stats['valid']}）")
    print(
        f"封存出現過嘅 park_id：{len(stats['parks_seen'])}；"
        f"對到 carparks：{len(stats['parks_matched'])}"
    )
    print(f"因唔喺 carparks 而跳過嘅 park 次數（累加）：{stats['skipped_unknown_park']}")
    if write:
        print(f"實際寫入：{stats['written']}")
    else:
        print("dry-run：無寫入。加 --write 先會入庫。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="匯入 TD 歷史空位到 Supabase")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--month", help="YYYY-MM，整月")
    g.add_argument("--start", help="YYYY-MM-DD（需同時提供 --end）")
    p.add_argument("--end", help="YYYY-MM-DD（含當日）")
    p.add_argument("--dry-run", action="store_true", help="只統計（預設）")
    p.add_argument("--write", action="store_true", help="寫入 Supabase")
    p.add_argument(
        "--replace",
        action="store_true",
        help="寫入前刪除該日期區間內已有 vacancy_records",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多處理幾個快照（試跑用）",
    )
    p.add_argument("--force-download", action="store_true", help="忽略本地 ZIP 快取")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.month:
        start, end = month_bounds(args.month)
    else:
        if not args.end:
            raise SystemExit("用 --start 時必須同時提供 --end")
        start, end = parse_ymd(args.start), parse_ymd(args.end)
    if end < start:
        raise SystemExit("--end 不可早於 --start")

    write = bool(args.write)
    if args.dry_run:
        write = False

    run_import(
        start=start,
        end=end,
        write=write,
        replace=bool(args.replace) and write,
        max_files=args.max_files,
        force_download=bool(args.force_download),
    )


if __name__ == "__main__":
    main()
