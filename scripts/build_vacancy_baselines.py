#!/usr/bin/env python3
"""
方案 1：用 DATA.GOV.HK 過去一年歷史 ZIP，本機累加平均後寫入摘要。

預設輸出：
  data/vacancy_baselines.parquet
  data/vacancy_baselines_meta.json

可選 --write-supabase（需 DB 可寫）覆寫 vacancy_baselines 表。

例子：
  python scripts/build_vacancy_baselines.py
  python scripts/build_vacancy_baselines.py --days 365 --write-supabase
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

load_dotenv(ROOT / ".env")

from import_historical_vacancy import (  # noqa: E402
    HK_TZ,
    download_package,
    extract_private_car_hourly,
    filter_snapshots,
    iter_zip_snapshots,
    packages_for_range,
)
from parking_db import (  # noqa: E402
    get_write_client,
    list_park_ids,
    uses_service_role,
)

OUT_PARQUET = ROOT / "data" / "vacancy_baselines.parquet"
OUT_META = ROOT / "data" / "vacancy_baselines_meta.json"

# PK encoding（同 app / DB 約定）
# slot_dow:      (park, hh, dow, -1)
# slot_weekend:  (park, hh, -1, is_weekend)
# slot:          (park, hh, -1, -1)
# park:          (park, -1, -1, -1)


def _half_hour(ts: datetime) -> int:
    return ts.hour * 2 + (1 if ts.minute >= 30 else 0)


def accumulate_year(
    *,
    days: int,
    allowed: set[str],
    max_files: int | None,
) -> tuple[dict[tuple, list[float]], datetime, datetime, int]:
    """回傳 {key: [sum, count]}, window_start, window_end, snapshot_count。"""
    end_dt = datetime.now(HK_TZ)
    start_dt = end_dt - timedelta(days=days)
    start_d, end_d = start_dt.date(), end_dt.date()
    print(f"窗口：{start_dt.isoformat()} → {end_dt.isoformat()}（{days} 日）")

    packages = packages_for_range(start_d, end_d)
    if not packages:
        raise SystemExit("搵唔到歷史 ZIP")
    print(f"ZIP 包：{len(packages)} 個")

    sums: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    snapshots = 0

    def add(key: tuple, value: float) -> None:
        slot = sums[key]
        slot[0] += value
        slot[1] += 1.0

    for pkg in packages:
        path = download_package(pkg)
        for ts, payload in filter_snapshots(iter_zip_snapshots(path), start_d, end_d):
            if ts < start_dt or ts > end_dt:
                continue
            if max_files is not None and snapshots >= max_files:
                break
            snapshots += 1
            hh = _half_hour(ts)
            dow = ts.weekday()
            is_we = 1 if dow >= 5 else 0
            rows = extract_private_car_hourly(
                payload, collected_at=ts, allowed_park_ids=allowed
            )
            for row in rows:
                v = int(row["vacancy"])
                if v < 0:
                    continue
                pid = row["park_id"]
                add((pid, hh, dow, -1), v)
                add((pid, hh, -1, is_we), v)
                add((pid, hh, -1, -1), v)
                add((pid, -1, -1, -1), v)
            if snapshots % 200 == 0:
                print(f"  已處理 {snapshots} 個快照，keys={len(sums)}")
        if max_files is not None and snapshots >= max_files:
            break

    print(f"完成：{snapshots} 個快照，{len(sums)} 個聚合 key")
    return sums, start_dt, end_dt, snapshots


def sums_to_frame(
    sums: dict[tuple, list[float]],
    *,
    window_start: datetime,
    window_end: datetime,
) -> pd.DataFrame:
    now = datetime.now(HK_TZ)
    rows = []
    for (park_id, hh, dow, is_we), (total, n) in sums.items():
        if n <= 0:
            continue
        rows.append(
            {
                "park_id": park_id,
                "half_hour": int(hh),
                "dow": int(dow),
                "is_weekend": int(is_we),
                "mean": float(total / n),
                "n": int(n),
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "updated_at": now.isoformat(),
            }
        )
    return pd.DataFrame(rows)


def write_local(df: pd.DataFrame, meta: dict) -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫 {OUT_PARQUET}（{len(df)} 列）")
    print(f"已寫 {OUT_META}")


def write_supabase(df: pd.DataFrame) -> None:
    if not uses_service_role():
        raise SystemExit("--write-supabase 需要 SUPABASE_SERVICE_ROLE_KEY")
    client = get_write_client()
    records = df.to_dict(orient="records")
    print(f"寫入 Supabase vacancy_baselines：{len(records)} 列…")
    # 全表覆寫
    client.table("vacancy_baselines").delete().neq("park_id", "").execute()
    batch = 500
    for i in range(0, len(records), batch):
        chunk = records[i : i + batch]
        client.table("vacancy_baselines").upsert(chunk).execute()
        if (i // batch) % 20 == 0:
            print(f"  upsert {min(i + batch, len(records))}/{len(records)}")
    print("Supabase 寫入完成")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="建立過去一年 vacancy baseline 摘要")
    p.add_argument("--days", type=int, default=365, help="往前幾多日（預設 365）")
    p.add_argument("--max-files", type=int, default=None, help="最多快照數（試跑）")
    p.add_argument("--write-supabase", action="store_true")
    p.add_argument("--skip-local", action="store_true")
    args = p.parse_args(argv)

    allowed = set(list_park_ids())
    print(f"carparks 白名單：{len(allowed)}")
    sums, w_start, w_end, snapshots = accumulate_year(
        days=args.days, allowed=allowed, max_files=args.max_files
    )
    df = sums_to_frame(sums, window_start=w_start, window_end=w_end)
    if df.empty:
        raise SystemExit("無有效聚合結果")

    meta = {
        "days": args.days,
        "window_start": w_start.isoformat(),
        "window_end": w_end.isoformat(),
        "snapshots": snapshots,
        "rows": len(df),
        "parks": int(df["park_id"].nunique()),
    }
    if not args.skip_local:
        write_local(df, meta)
    if args.write_supabase:
        write_supabase(df)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
