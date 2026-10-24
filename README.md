# HK Parking Vacancy Prediction

Capstone project: Predict parking vacancy 30–60 minutes ahead for car parks in Kwun Tong and Sha Tin districts.

## Research Question

Can we predict whether a car park will have private-car vacancies (classification) and the number of vacancies (regression) 30–60 minutes ahead, using official real-time data + time features, and outperform a same-weekday-same-hour historical average baseline?

## Current Status

- Data collection running (every ~10 min) via `collect.ipynb`
- Local SQLite `parking.db` **and** mirrored Supabase Postgres (`carparks`, `vacancy_records`, `weather_records`)
- ~587 car parks territory-wide; historical vacancy snapshots from mid-Oct 2026
- Baseline and Gradient Boosting models implemented
- Streamlit app「咪再兜」with recommendation query
- Report outline drafted

## Supabase

Project: `keithlky1992's Project` (`onxvxtaigqimuishfddp`, region `ap-south-1`).

| Table | Role |
|-------|------|
| `carparks` | Static park metadata (PK `park_id`) |
| `vacancy_records` | Time-series vacancy snapshots |
| `weather_records` | HKO weather snapshots |

RLS: public **SELECT** for `anon` / `authenticated`; writes require **service role**.

`.env` keys:

```
SUPABASE_URL=https://onxvxtaigqimuishfddp.supabase.co
SUPABASE_ANON_KEY=...                 # app.py 讀取
SUPABASE_SERVICE_ROLE_KEY=...         # 採集寫入（Dashboard → Settings → API）
```

```bash
# App（讀雲端）
streamlit run app.py

# 採集（寫雲端，每 10 分鐘）
python scripts/collect_to_supabase.py
# 或跑 collect.ipynb

# 一次性由本機 SQLite 匯入（已做過可略）
python scripts/migrate_sqlite_to_supabase.py
```

## Key Files

- `app.py`: Streamlit「咪再兜」（讀 Supabase）
- `parking_db.py`: Supabase 讀寫共用層
- `scripts/collect_to_supabase.py` / `collect.ipynb`: 每 10 分鐘採集寫入雲端
- `scripts/migrate_sqlite_to_supabase.py`: 一次性 SQLite → Supabase
- `explore_api.ipynb`: 建模／EDA
- `parking.db`: 本機備份（可選）
- `report_outline.md`: 報告大綱

## Next Steps

- Scheduled cloud collection（Edge Function / Cron，唔靠本機常開）
- Persist trained models（Supabase Storage）
- Complete final report and presentation
