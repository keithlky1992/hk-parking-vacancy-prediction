import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_mic_recorder import mic_recorder
from streamlit_js_eval import get_geolocation
from dotenv import load_dotenv
import asyncio
import base64
import html
import io
import os
import json
import re
import wave
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo
from pyflac import StreamEncoder

from google_areas import reverse_geocode_location_label
from parking_db import get_secret, load_vacancy_frame

WEEKDAY_CN = {0: "週一", 1: "週二", 2: "週三", 3: "週四", 4: "週五", 5: "週六", 6: "週日"}
WEEKDAY_SHORT = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}
DAY_TO_NUM = {"週一": 0, "週二": 1, "週三": 2, "週四": 3, "週五": 4, "週六": 5, "週日": 6}

# 「邊兩個／哪三個」等中文數字
CN_NUM = {
    "一": 1,
    "壹": 1,
    "二": 2,
    "兩": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
NUM_CN = {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
DEFAULT_TOP_N = 10
# 暫時隱藏建議／資料答覆書面文字；朗讀掣／自動播仍然保留。改 True 即可恢復顯示
SHOW_WRITTEN_SUGGESTION = False
# 朗讀自然結束時由 JS 點擊；放頁頂並以 CSS 收起，唔插入結果區
SPEECH_ENDED_SENTINEL_LABEL = "SPEECH_ENDED"

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
LOGO_ICON_PATH = ASSETS_DIR / "mi_zai_dau_icon.png"
FAVICON_PATH = ASSETS_DIR / "mi_zai_dau_favicon.png"

st.set_page_config(
    page_title="咪再兜",
    page_icon=str(FAVICON_PATH if FAVICON_PATH.exists() else LOGO_ICON_PATH),
    layout="centered",
    initial_sidebar_state="collapsed",
)

DEEPSEEK_API_KEY = get_secret("DEEPSEEK_API_KEY") or None
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# Google STT 需要 FLAC；不用 speech_recognition 內建的 x86_64 flac-mac。
GOOGLE_STT_URL = "http://www.google.com/speech-api/v2/recognize"
GOOGLE_STT_KEY = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"

# Google Maps Geocoding：用地名定位，再篩附近停車場。
# 港九密度高，「附近」用較細半徑，避免 3km 內遠區大車場（如 AIRSIDE）蓋過目標地點門口車場。
GOOGLE_MAPS_API_KEY = get_secret("GOOGLE_MAPS_API_KEY") or None
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
NEARBY_RADIUS_KM = 1.2
NEARBY_RADIUS_FALLBACK_KM = 2.0
# 地標「附近」排序：空位 ÷ (1+距離)^冪，較近車場優先
NEARBY_DISTANCE_SCORE_POWER = 2.0
# 極近且名稱實質相同 → 視為同一停車場（運輸署常有兩套 park_id）
EQUIVALENT_CARPARK_MAX_KM = 0.06

# 政府停車場 API：基本資料（含收費規則）
CARPARK_INFO_VACANCY_URL = "https://api.data.gov.hk/v1/carpark-info-vacancy"
DOW_TO_EN = {
    0: "MON",
    1: "TUE",
    2: "WED",
    3: "THU",
    4: "FRI",
    5: "SAT",
    6: "SUN",
}
DAY_CN_TO_EN = {name: DOW_TO_EN[num] for num, name in WEEKDAY_CN.items()}
DAYNIGHT_FEE_LABEL = {
    "day-park": "日泊",
    "night-park": "夜泊",
    "6-hours-park": "6小時",
    "12-hours-park": "12小時",
    "24-hours-park": "24小時",
}


def _pcm_from_audio(audio: dict) -> tuple[np.ndarray, int]:
    """從 mic_recorder 結果取出 mono int16 PCM 與 sample rate。"""
    raw = audio["bytes"]
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
        samples = np.frombuffer(frames, dtype=dtype)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if samples.dtype != np.int16:
            # 正規化到 int16
            samples = samples.astype(np.float64)
            max_abs = np.max(np.abs(samples)) or 1.0
            samples = (samples / max_abs * 32767).astype(np.int16)
        return samples.astype(np.int16, copy=False), int(rate)

    rate = int(audio.get("sample_rate") or 16000)
    width = int(audio.get("sample_width") or 2)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype)
    if samples.dtype != np.int16:
        samples = samples.astype(np.float64)
        max_abs = np.max(np.abs(samples)) or 1.0
        samples = (samples / max_abs * 32767).astype(np.int16)
    return samples.astype(np.int16, copy=False), rate


def _pcm_to_flac(samples: np.ndarray, sample_rate: int) -> bytes:
    """用 pyflac（arm64 可用）把 PCM 編成 FLAC。"""
    chunks: list[bytes] = []

    def write_callback(data, num_bytes, num_samples, current_frame):
        chunks.append(bytes(data[:num_bytes]))

    encoder = StreamEncoder(sample_rate=sample_rate, write_callback=write_callback)
    encoder.process(samples)
    encoder.finish()
    return b"".join(chunks)


def _parse_google_stt_response(response_text: str) -> str:
    for line in response_text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        results = payload.get("result") or []
        if not results:
            continue
        alts = results[0].get("alternative") or []
        if alts and alts[0].get("transcript"):
            return alts[0]["transcript"].strip()
    return ""


def transcribe_audio(audio: dict) -> str:
    """將 mic_recorder 音訊轉成文字。"""
    samples, rate = _pcm_from_audio(audio)
    if samples.size == 0:
        return ""
    if rate < 8000:
        # Google STT 要求至少 8kHz；簡單重複取樣拉高
        factor = int(np.ceil(8000 / rate))
        samples = np.repeat(samples, factor)
        rate = rate * factor

    flac_bytes = _pcm_to_flac(samples, rate)
    last_error = None
    for lang in ("yue-Hant-HK", "zh-HK", "zh-TW"):
        params = urlencode(
            {
                "client": "chromium",
                "lang": lang,
                "key": GOOGLE_STT_KEY,
                "pFilter": 0,
            }
        )
        try:
            response = requests.post(
                f"{GOOGLE_STT_URL}?{params}",
                data=flac_bytes,
                headers={"Content-Type": f"audio/x-flac; rate={rate}"},
                timeout=30,
            )
            response.raise_for_status()
            text = _parse_google_stt_response(response.text)
            if text:
                return text
        except requests.RequestException as e:
            last_error = e
            continue
    if last_error:
        raise RuntimeError(str(last_error)) from last_error
    return ""


# 天文台自動氣象站座標（來源：hko.gov.hk/tc/cis/stn.htm），英文名對應 1 分鐘 CSV。
HKO_AWS_STATIONS = {
    "Chek Lap Kok": {"name_tc": "赤鱲角", "lat": 22.309444, "lon": 113.921944},
    "Cheung Chau": {"name_tc": "長洲", "lat": 22.201111, "lon": 114.026667},
    "Clear Water Bay": {"name_tc": "清水灣", "lat": 22.263333, "lon": 114.299722},
    "Happy Valley": {"name_tc": "跑馬地", "lat": 22.270556, "lon": 114.183611},
    "HK Observatory": {"name_tc": "香港天文台", "lat": 22.301944, "lon": 114.174167},
    "HK Park": {"name_tc": "香港公園", "lat": 22.278333, "lon": 114.162222},
    "Kai Tak Runway Park": {"name_tc": "啟德跑道公園", "lat": 22.305, "lon": 114.216944},
    "Kau Sai Chau": {"name_tc": "滘西洲", "lat": 22.370278, "lon": 114.3125},
    "King's Park": {"name_tc": "京士柏", "lat": 22.311944, "lon": 114.172778},
    "Kowloon City": {"name_tc": "九龍城", "lat": 22.335, "lon": 114.184722},
    "Kwun Tong": {"name_tc": "觀塘", "lat": 22.318611, "lon": 114.224722},
    "Lau Fau Shan": {"name_tc": "流浮山", "lat": 22.468889, "lon": 113.983611},
    "Ngong Ping": {"name_tc": "昂坪", "lat": 22.258611, "lon": 113.912778},
    "Pak Tam Chung": {"name_tc": "北潭涌", "lat": 22.402778, "lon": 114.323056},
    "Peng Chau": {"name_tc": "坪洲", "lat": 22.291111, "lon": 114.043333},
    "Sai Kung": {"name_tc": "西貢", "lat": 22.375556, "lon": 114.274444},
    "Sha Tin": {"name_tc": "沙田", "lat": 22.4025, "lon": 114.21},
    "Sham Shui Po": {"name_tc": "深水埗", "lat": 22.335833, "lon": 114.136944},
    "Shau Kei Wan": {"name_tc": "筲箕灣", "lat": 22.281667, "lon": 114.236111},
    "Shek Kong": {"name_tc": "石崗", "lat": 22.436111, "lon": 114.084722},
    "Sheung Shui": {"name_tc": "上水", "lat": 22.501944, "lon": 114.111111},
    "Stanley": {"name_tc": "赤柱", "lat": 22.214167, "lon": 114.218611},
    "Ta Kwu Ling": {"name_tc": "打鼓嶺", "lat": 22.528611, "lon": 114.156667},
    "Tai Lung": {"name_tc": "大隴", "lat": 22.484722, "lon": 114.1175},
    "Tai Mei Tuk": {"name_tc": "大美督", "lat": 22.475278, "lon": 114.2375},
    "Tai Mo Shan": {"name_tc": "大帽山", "lat": 22.410556, "lon": 114.124444},
    "Tai Po": {"name_tc": "大埔", "lat": 22.448333, "lon": 114.177222},
    "Tate's Cairn": {"name_tc": "大老山", "lat": 22.357778, "lon": 114.217778},
    "The Peak": {"name_tc": "山頂", "lat": 22.264167, "lon": 114.155},
    "Tseung Kwan O": {"name_tc": "將軍澳", "lat": 22.315833, "lon": 114.255556},
    "Tsing Yi": {"name_tc": "青衣", "lat": 22.344167, "lon": 114.11},
    "Tsuen Wan Ho Koon": {"name_tc": "荃灣可觀", "lat": 22.383611, "lon": 114.107778},
    "Tsuen Wan Shing Mun Valley": {"name_tc": "荃灣城門谷", "lat": 22.375556, "lon": 114.126667},
    "Tuen Mun": {"name_tc": "屯門", "lat": 22.390556, "lon": 113.976667},
    "Waglan Island": {"name_tc": "橫瀾島", "lat": 22.183056, "lon": 114.302778},
    "Wetland Park": {"name_tc": "濕地公園", "lat": 22.466667, "lon": 114.008889},
    "Wong Chuk Hang": {"name_tc": "黃竹坑", "lat": 22.247778, "lon": 114.173611},
    "Wong Tai Sin": {"name_tc": "黃大仙", "lat": 22.339444, "lon": 114.205278},
    "Yuen Long Park": {"name_tc": "元朗公園", "lat": 22.440833, "lon": 114.018333},
}
HKO_TEMP_CSV_URL = (
    "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/"
    "latest_1min_temperature.csv"
)
HKO_HUMIDITY_CSV_URL = (
    "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/"
    "latest_1min_humidity.csv"
)
# 距離最近測站超過此距離（公里）視為非香港，不使用天文台數據
HKO_MAX_STATION_DISTANCE_KM = 80.0


def hong_kong_now() -> datetime:
    """目前香港時間（naive，方便與業務邏輯比較）。"""
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).replace(tzinfo=None)


def format_hong_kong_now() -> str:
    """香港時間（Asia/Hong_Kong），例如：2026年10月21日（二）7:54:05pm。"""
    now = hong_kong_now()
    hour12 = now.hour % 12 or 12
    suffix = "am" if now.hour < 12 else "pm"
    return (
        f"{now.year}年{now.month}月{now.day}日"
        f"（{WEEKDAY_SHORT[now.weekday()]}）"
        f"{hour12}:{now.minute:02d}:{now.second:02d}{suffix}"
    )


def font_size_for_content(
    text: str,
    *,
    base: str = "1.15rem",
    medium: str = "1.0rem",
    small: str = "0.9rem",
) -> str:
    """依文字長度選字級：短內容用大字，長內容（如完整日期時間）用較小字。"""
    n = len((text or "").strip())
    if n <= 8:
        return base
    if n <= 16:
        return medium
    return small


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """兩點球面距離（公里）。"""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def geocode_place(place: str) -> dict | None:
    """
    用 Google Geocoding 搜尋香港地點。
    成功回傳 {lat, lon, formatted_address}；找不到回 None。
    無 API key 或 API 異常時 raise RuntimeError。
    """
    text = (place or "").strip()
    if not text:
        return None
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("missing_api_key")

    params = {
        "address": text,
        "region": "hk",
        "language": "zh-HK",
        "components": "country:HK",
        "key": GOOGLE_MAPS_API_KEY,
    }
    response = requests.get(GOOGLE_GEOCODE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if status == "ZERO_RESULTS":
        return None
    if status != "OK" or not data.get("results"):
        raise RuntimeError(f"geocode_status:{status}")

    top = data["results"][0]
    loc = top.get("geometry", {}).get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")
    if lat is None or lng is None:
        return None
    return {
        "lat": float(lat),
        "lon": float(lng),
        "formatted_address": (top.get("formatted_address") or text).strip(),
    }


def _is_all_hong_kong(district) -> bool:
    text = str(district or "").strip()
    return text in ("", "全部", "全港", "香港", "null", "None")


def attach_geocode_to_intent(intent: dict) -> str | None:
    """
    非全港地點一律用 Google Geocoding 定位，寫入 geo_lat / geo_lon / geo_label。
    篩選停車場只靠此座標＋距離半徑；保留 district 原文作顯示／除錯。
    成功回 None；失敗回傳給使用者看的錯誤訊息。
    """
    district = intent.get("district") or "全港"
    if _is_all_hong_kong(district):
        return None

    place = str(district).strip()
    try:
        result = geocode_place(place)
    except RuntimeError as exc:
        msg = str(exc)
        if msg == "missing_api_key":
            return (
                "尚未設定 Google Maps API Key。"
                "請在專案根目錄 .env 加入 GOOGLE_MAPS_API_KEY=你的金鑰後重新啟動。"
            )
        return (
            f"地點搜尋暫時無法使用（{msg}）。"
            "請稍後再試，或重新輸入查詢。"
        )
    except requests.RequestException:
        return "地點搜尋連線失敗，請稍後再試或重新輸入查詢。"

    if result is None:
        return (
            f"找不到「{place}」這個地方。"
            "請重新輸入更清楚的地名（例如商場、街道或地區名稱）。"
        )

    intent["geo_lat"] = result["lat"]
    intent["geo_lon"] = result["lon"]
    intent["geo_label"] = result["formatted_address"]
    return None


def _parse_hko_latest_csv(text: str) -> dict[str, float]:
    """解析天文台 latest_1min CSV → {英文站名: 數值}。"""
    readings: dict[str, float] = {}
    for i, line in enumerate(text.splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        station = parts[1].strip()
        raw = parts[2].strip()
        try:
            readings[station] = float(raw)
        except ValueError:
            continue
    return readings


def _nearest_hko_station(
    lat: float, lon: float, candidates: list[str] | None = None
) -> tuple[str, float] | None:
    """回傳距離最近的天文台測站英文名與距離（公里）。"""
    names = candidates if candidates is not None else list(HKO_AWS_STATIONS)
    best_name = None
    best_dist = float("inf")
    for name in names:
        info = HKO_AWS_STATIONS.get(name)
        if not info:
            continue
        dist = _haversine_km(lat, lon, info["lat"], info["lon"])
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_name is None:
        return None
    return best_name, best_dist


def resolve_user_district(lat: float, lon: float) -> str | None:
    """只用 Google Maps reverse geocode 把 GPS 轉成地區名。"""
    try:
        return reverse_geocode_location_label(lat, lon)
    except Exception:
        return None


# 地區名變動較慢；天氣對齊天文台 1 分鐘更新
LOCATION_CACHE_TTL_SEC = 10 * 60
WEATHER_CACHE_TTL_SEC = 60
HEADER_AUTO_REFRESH = timedelta(seconds=WEATHER_CACHE_TTL_SEC)


@st.cache_data(ttl=LOCATION_CACHE_TTL_SEC)
def fetch_user_location_label(lat: float, lon: float) -> str | None:
    """Google reverse geocode → 地區名（優先中文）。"""
    # cache key 含版本，避免選名邏輯更新後仍用舊英文結果
    _ = "zh-first-v1"
    return resolve_user_district(lat, lon)


def _format_temp_humidity(temp: float, humidity: float) -> str:
    """對齊天文台 App：氣溫 + 相對濕度。"""
    return f"{temp:.1f}°C｜相對濕度 {humidity:.0f}%"


@st.cache_data(ttl=WEATHER_CACHE_TTL_SEC)
def fetch_user_weather(lat: float, lon: float) -> str | None:
    """天文台 1 分鐘氣溫／相對濕度（對齊所在地區，類似天文台 App）。

    部分測站只有氣溫、冇濕度（例如黃大仙）。優先揀附近同時有兩者嘅站；
    否則分別用最近氣溫站 + 最近濕度站，確保地區天氣仍顯示完整。
    """
    try:
        temp_resp = requests.get(HKO_TEMP_CSV_URL, timeout=10)
        temp_resp.raise_for_status()
        temps = _parse_hko_latest_csv(temp_resp.text)

        hum_resp = requests.get(HKO_HUMIDITY_CSV_URL, timeout=10)
        hum_resp.raise_for_status()
        hums = _parse_hko_latest_csv(hum_resp.text)
    except Exception:
        return None

    both_candidates = [
        n for n in temps if n in HKO_AWS_STATIONS and n in hums
    ]
    nearest_both = _nearest_hko_station(lat, lon, both_candidates)
    if nearest_both is not None and nearest_both[1] <= HKO_MAX_STATION_DISTANCE_KM:
        station_en, _ = nearest_both
        return _format_temp_humidity(temps[station_en], hums[station_en])

    # 後備：氣溫、濕度分開揀最近有讀數嘅站（仍限香港範圍）
    nearest_temp = _nearest_hko_station(
        lat, lon, [n for n in temps if n in HKO_AWS_STATIONS]
    )
    nearest_hum = _nearest_hko_station(
        lat, lon, [n for n in hums if n in HKO_AWS_STATIONS]
    )
    if (
        nearest_temp is None
        or nearest_hum is None
        or nearest_temp[1] > HKO_MAX_STATION_DISTANCE_KM
        or nearest_hum[1] > HKO_MAX_STATION_DISTANCE_KM
    ):
        return None

    return _format_temp_humidity(
        temps[nearest_temp[0]], hums[nearest_hum[0]]
    )


def fetch_location_and_weather(lat: float, lon: float) -> dict:
    """地區 + 天氣（分開 cache，避免為咗天氣每分鐘打 Google Geocode）。"""
    return {
        "location": fetch_user_location_label(lat, lon),
        "weather": fetch_user_weather(lat, lon),
    }


def resolve_user_geolocation() -> None:
    """向瀏覽器要 GPS；有新座標就更新，等緊回應時保留舊值。"""
    geo = get_geolocation()
    if geo is None:
        return
    if isinstance(geo, dict) and "error" in geo:
        # 已有座標就唔因單次失敗清掉（自動刷新期間短暫失敗）
        if not st.session_state.get("user_geo_coords"):
            st.session_state.user_geo_error = True
        return
    if isinstance(geo, dict) and geo.get("coords"):
        lat = float(geo["coords"]["latitude"])
        lon = float(geo["coords"]["longitude"])
        # 約 100m 精度，減少 GPS 微抖導致的 cache miss
        st.session_state.user_geo_coords = (round(lat, 3), round(lon, 3))
        st.session_state.pop("user_geo_error", None)


@st.fragment(run_every=HEADER_AUTO_REFRESH)
def render_header_context_panel() -> None:
    """標題下資訊卡：時間由前端 JS 每秒改文字；地區／天氣每 60 秒自動刷新。"""
    resolve_user_geolocation()

    hk_time = format_hong_kong_now()
    location_label = "定位中…"
    weather_label = "定位中…"

    if st.session_state.get("user_geo_error"):
        location_label = "無法取得地區"
        weather_label = "無法取得位置"
    elif st.session_state.get("user_geo_coords"):
        lat, lon = st.session_state.user_geo_coords
        info = fetch_location_and_weather(lat, lon)
        location_label = info["location"] or "無法取得地區"
        weather_label = info["weather"] or "暫時無法取得"

    rows = [
        ("香港時間", hk_time, "mzd-hk-time"),
        ("你目前所在的地區", location_label, "mzd-location"),
        ("天氣", weather_label, "mzd-weather"),
    ]
    cells = []
    for label, value, value_class in rows:
        class_attr = f' class="{html.escape(value_class)}"' if value_class else ""
        cells.append(
            f'<div style="min-width:0;flex:1 1 0;">'
            f'<div style="font-size:0.85rem;line-height:1.2;opacity:0.65;margin-bottom:0.15rem;">'
            f"{html.escape(label)}</div>"
            f'<div{class_attr} style="font-size:1.05rem;font-weight:600;line-height:1.35;'
            f'word-break:break-word;">{html.escape(value)}</div></div>'
        )
    cells = "".join(cells)
    st.markdown(
        f'<div style="display:flex;gap:1.25rem;align-items:flex-start;'
        f'flex-wrap:wrap;padding:0.7rem 0.9rem;margin:0.15rem 0 0.85rem 0;'
        f'border:1px solid rgba(128,128,128,0.28);border-radius:0.6rem;">'
        f"{cells}</div>",
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    """顯示「咪再兜」品牌 logo 與副標；撳 logo 或名稱可重新整理頁面。"""
    if not LOGO_ICON_PATH.exists():
        st.title("咪再兜")
        st.caption("香港停車場空位預測｜AI Agent 查詢")
        return

    icon_b64 = base64.b64encode(LOGO_ICON_PATH.read_bytes()).decode("ascii")
    # 用 components.html 令 click／reload 喺 Streamlit iframe 內可靠運作
    components.html(
        f"""
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    overflow: hidden;
  }}
  .mzd-brand {{
    display: flex;
    align-items: center;
    gap: 0.95rem;
    margin: 0 0 2px 0;
    padding: 0.95rem 1.1rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, #0b0f14 0%, #121a22 58%, #0a3531 100%);
    border: 1px solid rgba(0, 229, 192, 0.28);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
    font-family: "Source Sans Pro", "Segoe UI", sans-serif;
    box-sizing: border-box;
  }}
  .mzd-home {{
    display: flex;
    align-items: center;
    gap: 0.95rem;
    min-width: 0;
    text-decoration: none;
    color: inherit;
    cursor: pointer;
    border-radius: 0.55rem;
  }}
  .mzd-home:focus-visible {{
    outline: 2px solid rgba(0, 229, 192, 0.7);
    outline-offset: 3px;
  }}
  .mzd-home:hover .mzd-name {{
    color: #ffffff;
  }}
  .mzd-home:hover .mzd-mi {{
    color: #5ff0d4;
  }}
  .mzd-home:hover .mzd-mark {{
    filter: drop-shadow(0 4px 12px rgba(0, 229, 192, 0.32));
    transform: scale(1.03);
  }}
  .mzd-mark {{
    width: 76px;
    height: 76px;
    flex-shrink: 0;
    display: block;
    transition: transform 0.15s ease, filter 0.15s ease;
    filter: drop-shadow(0 4px 10px rgba(0, 229, 192, 0.18));
  }}
  .mzd-text {{
    min-width: 0;
  }}
  .mzd-name {{
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #f4f7f8;
    line-height: 1.15;
    transition: color 0.15s ease;
  }}
  .mzd-mi {{
    color: #00e5c0;
    transition: color 0.15s ease;
  }}
  .mzd-tag {{
    margin-top: 0.28rem;
    font-size: 0.86rem;
    color: rgba(244, 247, 248, 0.68);
    letter-spacing: 0.02em;
  }}
  @media (max-width: 480px) {{
    .mzd-brand {{
      gap: 0.75rem;
      padding: 0.8rem 0.9rem;
    }}
    .mzd-home {{
      gap: 0.75rem;
    }}
    .mzd-mark {{
      width: 62px;
      height: 62px;
    }}
    .mzd-name {{
      font-size: 1.55rem;
      letter-spacing: 0.05em;
    }}
    .mzd-tag {{
      font-size: 0.78rem;
    }}
  }}
</style>
<div class="mzd-brand">
  <a class="mzd-home" href="#" id="mzd-home" title="重新整理頁面" aria-label="重新整理咪再兜">
    <img class="mzd-mark" src="data:image/png;base64,{icon_b64}" alt="咪再兜" />
    <div class="mzd-text">
      <div class="mzd-name"><span class="mzd-mi">咪</span>再兜</div>
      <div class="mzd-tag">香港停車場空位預測｜AI Agent 查詢</div>
    </div>
  </a>
</div>
<script>
  document.getElementById("mzd-home").addEventListener("click", function (e) {{
    e.preventDefault();
    try {{
      (window.top || window.parent || window).location.reload();
    }} catch (err) {{
      window.location.reload();
    }}
  }});
</script>
""",
        height=112,
        scrolling=False,
    )


def inject_layout_css() -> None:
    """統一間距與對齊，令標題／查詢列／結果區更整齊。"""
    st.markdown(
        """
<style>
  .block-container {
    padding-top: 3.6rem;
    padding-bottom: 2rem;
    max-width: 880px;
  }
  div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
  }
  h1 {
    margin: 0 0 0.25rem 0 !important;
    padding: 0.25rem 0 0 0 !important;
    font-size: 1.75rem !important;
    line-height: 1.4 !important;
    overflow: visible !important;
  }
  [data-testid="stHeadingWithActionElements"],
  [data-testid="stMarkdownContainer"] h1 {
    overflow: visible !important;
  }
  [data-testid="stCaption"] {
    margin-bottom: 0.35rem !important;
  }
  /* get_geolocation 的 iframe 會撐開空隙；資訊卡每 60s 刷新仍會插入，一律收起。 */
  iframe[title*="streamlit_js_eval"] {
    position: fixed !important;
    left: -10000px !important;
    top: 0 !important;
    width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }
  [data-testid="stElementContainer"]:has(iframe[title*="streamlit_js_eval"]) {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    border: none !important;
  }
  div[data-testid="stTextInput"] {
    margin-bottom: 0 !important;
  }
  div[data-testid="column"] > div,
  div[data-testid="stColumn"] > div {
    width: 100%;
  }
  /* 查詢掣同語音掣：同一高度／圓角／字級／字型 */
  div[data-testid="stFormSubmitButton"] {
    width: 100%;
  }
  div[data-testid="stFormSubmitButton"] button {
    height: 2.5rem !important;
    min-height: 2.5rem !important;
    width: 100% !important;
    padding: 0 0.6rem !important;
    border-radius: 0.5rem !important;
    line-height: 1 !important;
    box-sizing: border-box !important;
    font-size: 1rem !important;
    font-family: "Source Sans Pro", "Segoe UI", sans-serif !important;
    font-weight: 400 !important;
  }
  div[data-testid="stFormSubmitButton"] button p,
  div[data-testid="stFormSubmitButton"] button span {
    font-size: 1rem !important;
    font-family: inherit !important;
    font-weight: inherit !important;
    line-height: 1 !important;
  }
  /* 同語音掣（mic_recorder）：outline 預設 + 珊瑚紅 hover（寫死色碼，避免 theme 變數缺失） */
  div[data-testid="stFormSubmitButton"] button,
  button.mzd-mic-like,
  div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #0e1117 !important;
    border: 1px solid rgba(250, 250, 250, 0.2) !important;
    color: rgba(250, 250, 250, 0.95) !important;
  }
  div[data-testid="stFormSubmitButton"] button:hover,
  div[data-testid="stFormSubmitButton"] button:focus-visible,
  button.mzd-mic-like:hover,
  button.mzd-mic-like:focus-visible,
  div[data-testid="stButton"] button[kind="secondary"]:hover,
  div[data-testid="stButton"] button[kind="secondary"]:focus-visible {
    border-color: #ff4b4b !important;
    color: #ff4b4b !important;
    background-color: #0e1117 !important;
  }
  div[data-testid="stFormSubmitButton"] button:hover p,
  div[data-testid="stFormSubmitButton"] button:hover span,
  div[data-testid="stFormSubmitButton"] button:focus-visible p,
  div[data-testid="stFormSubmitButton"] button:focus-visible span,
  button.mzd-mic-like:hover p,
  button.mzd-mic-like:hover span,
  button.mzd-mic-like:focus-visible p,
  button.mzd-mic-like:focus-visible span,
  div[data-testid="stButton"] button[kind="secondary"]:hover p,
  div[data-testid="stButton"] button[kind="secondary"]:hover span,
  div[data-testid="stButton"] button[kind="secondary"]:focus-visible p,
  div[data-testid="stButton"] button[kind="secondary"]:focus-visible span {
    color: #ff4b4b !important;
  }
  div[data-testid="stFormSubmitButton"] button:focus,
  button.mzd-mic-like:focus,
  div[data-testid="stButton"] button[kind="secondary"]:focus {
    box-shadow: none !important;
  }
  iframe[title*="mic_recorder"] {
    height: 2.5rem !important;
    min-height: 2.5rem !important;
    width: 100% !important;
    display: block !important;
  }
  div[data-testid="stTextInput"] input {
    height: 2.5rem !important;
    min-height: 2.5rem !important;
    border-radius: 0.5rem !important;
    box-sizing: border-box !important;
  }
  /* Alert／語音掣：唔加額外 margin（只留 VerticalBlock 預設 gap＝margin a） */
  div[data-testid="stAlert"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
  }
  div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]),
  div[data-testid="stHorizontalBlock"]:has(.mzd-speech-left-slot),
  div[data-testid="stElementContainer"]:has(
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"])
  ),
  div[data-testid="stElementContainer"]:has(
    div[data-testid="stHorizontalBlock"]:has(.mzd-speech-left-slot)
  ),
  div[data-testid="stElementContainer"]:has(div[data-testid="stButton"] button[kind="secondary"]) {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
  }
  div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"])
    [data-testid="stElementContainer"],
  div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"])
    div[data-testid="stButton"],
  div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"])
    div[data-testid="column"] > div,
  div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"])
    div[data-testid="stColumn"] > div,
  div[data-testid="stHorizontalBlock"]:has(.mzd-speech-left-slot)
    [data-testid="stElementContainer"],
  div[data-testid="stHorizontalBlock"]:has(.mzd-speech-left-slot)
    div[data-testid="column"] > div,
  div[data-testid="stHorizontalBlock"]:has(.mzd-speech-left-slot)
    div[data-testid="stColumn"] > div {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
  }
  /* 手機 columns stack 時，空左欄唔好撐高 */
  @media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"])
      > div[data-testid="column"]:not(:has(button)),
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"])
      > div[data-testid="stColumn"]:not(:has(button)),
    div[data-testid="stHorizontalBlock"]:has(.mzd-speech-left-slot)
      > div[data-testid="column"]:has(.mzd-speech-left-slot),
    div[data-testid="stHorizontalBlock"]:has(.mzd-speech-left-slot)
      > div[data-testid="stColumn"]:has(.mzd-speech-left-slot) {
      display: none !important;
    }
  }
  /* SPEECH_ENDED sentinel：錨點 + 下一粒掣，完全移出文件流 */
  [data-testid="stElementContainer"]:has(.mzd-speech-ended-anchor),
  [data-testid="stElementContainer"]:has(.mzd-speech-ended-anchor)
    + [data-testid="stElementContainer"] {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    border: none !important;
  }
  /* Streamlit 1.51 會把 height=0 當 falsy 回落 ~150px；用 height=1 再收起。
     排除 mic_recorder，避免語音掣被一併藏起。 */
  iframe[height="0"]:not([title*="mic_recorder"]),
  iframe[height="1"]:not([title*="mic_recorder"]) {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }
  [data-testid="stElementContainer"]:has(iframe[height="0"]:not([title*="mic_recorder"])),
  [data-testid="stElementContainer"]:has(iframe[height="1"]:not([title*="mic_recorder"])) {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    border: none !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
    # mic 對齊、常駐朗讀控制器、查詢／停止掣即時 cancel
    components.html(
        """
<script>
(function () {
  var root = window.parent;
  var doc = root.document;

  function ensureSpeechController() {
    if (root.__hkParkingSpeech) return root.__hkParkingSpeech;

    function getSynth() {
      return root.speechSynthesis || null;
    }
    function getUtteranceCtor() {
      return root.SpeechSynthesisUtterance || null;
    }
    function pickVoice(voices) {
      var list = voices || [];
      return list.find(function (v) { return (v.lang || "").toLowerCase().startsWith("zh-hk"); })
        || list.find(function (v) { return (v.lang || "").toLowerCase().includes("hk"); })
        || list.find(function (v) { return (v.lang || "").toLowerCase().startsWith("zh"); })
        || null;
    }
    function clickSpeechEndedSentinel() {
      try {
        if (typeof root.__hkParkingSetSpeechSpeakingUi === "function") {
          root.__hkParkingSetSpeechSpeakingUi(false);
        }
        doc.querySelectorAll("button").forEach(function (btn) {
          if ((btn.textContent || "").trim() === "SPEECH_ENDED") btn.click();
        });
      } catch (e) {}
    }

    var ctrl = {
      _token: 0,
      _cancelled: false,
      _lastText: "",
      _speakTimer: null,
      _failTimer: null,
      // iOS／部分手機瀏覽器：第一次 speak 必須喺使用者手勢內同步觸發，之後先可以自動播
      unlock: function () {
        var synth = getSynth();
        var Utterance = getUtteranceCtor();
        if (!synth || !Utterance) return;
        try { synth.getVoices(); } catch (e) {}
        if (root.__hkParkingSpeechUnlocked) return;
        try {
          var warm = new Utterance("");
          warm.volume = 0;
          warm.lang = "zh-HK";
          synth.speak(warm);
          root.__hkParkingSpeechUnlocked = true;
        } catch (e) {}
      },
      stop: function () {
        this._cancelled = true;
        this._token += 1;
        if (this._speakTimer) {
          root.clearTimeout(this._speakTimer);
          this._speakTimer = null;
        }
        if (this._failTimer) {
          root.clearTimeout(this._failTimer);
          this._failTimer = null;
        }
        var synth = getSynth();
        if (!synth) return;
        try { synth.resume(); } catch (e) {}
        try { synth.cancel(); } catch (e) {}
        try { synth.cancel(); } catch (e) {}
      },
      speak: function (text, opts) {
        var self = this;
        var synth = getSynth();
        var Utterance = getUtteranceCtor();
        if (!synth || !Utterance || !text) return;
        opts = opts || {};
        // immediate：用戶剛撳掣；可縮短 delay，但仍要留極短間隔令 cancel 生效
        var startDelay = opts.immediate ? 40 : 150;

        // 取消任何未開始嘅排程，再開新一次（同一函式內原子完成，避免同 stop iframe 搶跑）
        if (self._speakTimer) {
          root.clearTimeout(self._speakTimer);
          self._speakTimer = null;
        }
        if (self._failTimer) {
          root.clearTimeout(self._failTimer);
          self._failTimer = null;
        }
        self._cancelled = false;
        self._token += 1;
        var token = self._token;
        self._lastText = text;
        root.__hkParkingSpeechLastText = text;
        if (typeof root.__hkParkingSetSpeechSpeakingUi === "function") {
          root.__hkParkingSetSpeechSpeakingUi(true);
        }

        try { synth.resume(); } catch (e) {}
        try { synth.cancel(); } catch (e) {}

        self._speakTimer = root.setTimeout(function () {
          self._speakTimer = null;
          if (token !== self._token || self._cancelled) return;

          var u = new Utterance(text);
          u.lang = "zh-HK";
          u.rate = 0.95;
          u.onend = function () {
            if (token !== self._token) return;
            if (self._cancelled) {
              self._cancelled = false;
              return;
            }
            root.setTimeout(function () {
              if (token !== self._token || self._cancelled) return;
              if (synth.speaking || synth.pending) return;
              clickSpeechEndedSentinel();
            }, 80);
          };
          u.onerror = u.onend;

          function start() {
            if (token !== self._token || self._cancelled) return;
            var voice = pickVoice(synth.getVoices());
            if (voice) u.voice = voice;
            try { synth.resume(); } catch (e) {}
            synth.speak(u);
          }

          var voices = synth.getVoices();
          if (voices && voices.length) {
            start();
          } else {
            var started = false;
            synth.onvoiceschanged = function () {
              if (started) return;
              started = true;
              start();
            };
            root.setTimeout(function () {
              if (!started) start();
            }, 250);
          }

          // 若仍未開始，再試一次（唔會 hardReset 取消自己）
          root.setTimeout(function () {
            if (token !== self._token || self._cancelled) return;
            if (!synth.speaking && !synth.pending) {
              try { synth.resume(); } catch (e) {}
              synth.speak(u);
            }
          }, 400);

          // 手機常見：未解鎖時 speak 會靜默失敗 → 還原掣面俾用戶手動播
          self._failTimer = root.setTimeout(function () {
            self._failTimer = null;
            if (token !== self._token || self._cancelled) return;
            if (synth.speaking || synth.pending) return;
            clickSpeechEndedSentinel();
          }, 700);
        }, startDelay);
      }
    };
    root.__hkParkingSpeech = ctrl;
    return ctrl;
  }

  var speech = ensureSpeechController();

  function setSpeechSpeakingUi(speaking) {
    // 而家只用單一 toggle 掣，唔再疊兩個掣；呢度只同步狀態旗標
    root.__hkParkingSpeechSpeaking = !!speaking;
  }
  root.__hkParkingSetSpeechSpeakingUi = setSpeechSpeakingUi;

  // 極短靜音 WAV：喺「查詢／語音」手勢內 play 一次，之後結果頁可用同一 Audio 自動播（iOS／手機）
  var SILENT_WAV = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAA==";

  function ensureUnlockedMediaAudio() {
    if (!root.__hkParkingMediaAudio) {
      root.__hkParkingMediaAudio = new root.Audio();
      root.__hkParkingMediaAudio.preload = "auto";
      try { root.__hkParkingMediaAudio.playsInline = true; } catch (e) {}
      try { root.__hkParkingMediaAudio.setAttribute("playsinline", "true"); } catch (e) {}
      try { root.__hkParkingMediaAudio.setAttribute("webkit-playsinline", "true"); } catch (e) {}
    }
    return root.__hkParkingMediaAudio;
  }

  function stopUnlockedMedia() {
    var a = root.__hkParkingMediaAudio;
    if (!a) return;
    try { a.pause(); } catch (e) {}
    try { a.currentTime = 0; } catch (e) {}
    a.onended = null;
    a.onerror = null;
  }

  function unlockMediaNow() {
    speech.unlock();
    try {
      var a = ensureUnlockedMediaAudio();
      // iOS：必須非 muted 解鎖，否則之後只能自動播靜音
      a.muted = false;
      a.volume = 1;
      a.src = SILENT_WAV;
      var finish = function () {
        try { a.pause(); a.currentTime = 0; } catch (e) {}
        root.__hkParkingMediaUnlocked = true;
      };
      var p = a.play();
      if (p && typeof p.then === "function") {
        p.then(finish).catch(function () {
          // 即使 promise 失敗，手勢內呼叫過 play() 有時仍算解鎖
          root.__hkParkingMediaUnlocked = true;
        });
      } else {
        finish();
      }
    } catch (e) {}
  }

  function playUnlockedMedia(src, cbs) {
    cbs = cbs || {};
    if (!src) {
      if (typeof cbs.onError === "function") cbs.onError();
      return false;
    }
    try {
      var a = ensureUnlockedMediaAudio();
      stopUnlockedMedia();
      a.muted = false;
      a.volume = 1;
      a.src = src;
      a.onended = function () {
        if (typeof cbs.onEnded === "function") cbs.onEnded();
      };
      a.onerror = function () {
        if (typeof cbs.onError === "function") cbs.onError();
      };
      var p = a.play();
      if (p && typeof p.then === "function") {
        p.then(function () {
          if (typeof cbs.onStart === "function") cbs.onStart();
        }).catch(function () {
          if (typeof cbs.onError === "function") cbs.onError();
        });
      } else if (typeof cbs.onStart === "function") {
        cbs.onStart();
      }
      return true;
    } catch (e) {
      if (typeof cbs.onError === "function") cbs.onError();
      return false;
    }
  }

  root.__hkParkingUnlockMedia = unlockMediaNow;
  root.__hkParkingPlayUnlockedMedia = playUnlockedMedia;
  root.__hkParkingStopUnlockedMedia = stopUnlockedMedia;

  function unlockSpeechNow() {
    unlockMediaNow();
  }

  // 停建議播音；唔動解鎖用 Audio，避免 pointerdown→click 連續兩下打掉 silent unlock
  function stopSuggestionPlaybackOnly() {
    speech.stop();
    setSpeechSpeakingUi(false);
    try { broadcastStopSpeechPlayers(); } catch (e) {}
  }

  function stopSpeechNow() {
    stopSuggestionPlaybackOnly();
    try { stopUnlockedMedia(); } catch (e) {}
  }

  function toggleSuggestionSpeechNow(ev) {
    // 單一掣：以掣面文字為準（避免狀態旗標同畫面唔同步）
    if (speech._toggleLock) return;
    speech._toggleLock = true;
    root.setTimeout(function () { speech._toggleLock = false; }, 400);
    var btn = ev && (ev.currentTarget || ev.target);
    var label = ((btn && btn.textContent) || "").replace(/\\s+/g, "");
    if (label.indexOf("停止播放") !== -1) {
      stopSpeechNow();
      return;
    }
    // 必須喺 pointerdown／click 手勢內先 unlock，iOS 先肯出聲
    unlockSpeechNow();
    var text = root.__hkParkingSpeechLastText || (speech._lastText || "");
    if (!text) return;
    speech.speak(text, { immediate: true });
    setSpeechSpeakingUi(true);
  }

  function collapseIframeContainer(iframe) {
    iframe.style.position = "fixed";
    iframe.style.left = "-10000px";
    iframe.style.top = "0";
    iframe.style.width = "0";
    iframe.style.height = "0";
    iframe.style.minHeight = "0";
    iframe.style.border = "none";
    iframe.style.opacity = "0";
    iframe.style.pointerEvents = "none";
    var wrap = iframe.closest('[data-testid="stElementContainer"]');
    if (wrap) {
      wrap.style.position = "absolute";
      wrap.style.width = "0";
      wrap.style.height = "0";
      wrap.style.minHeight = "0";
      wrap.style.margin = "0";
      wrap.style.padding = "0";
      wrap.style.overflow = "hidden";
      wrap.style.border = "none";
    }
  }

  function collapseJsEvalIframes() {
    doc.querySelectorAll('iframe[title*="streamlit_js_eval"]').forEach(collapseIframeContainer);
  }

  // Streamlit 1.51：height=0 會被當成 falsy 而用預設 ~150px，撐開「播放」下方空白。
  // 標記用 height=1 的 speech sync／speak／stop iframe，一律收起。
  // 唔好收起 mic_recorder（語音掣）；佢可能短暫 height=0／未 setFrameHeight。
  function isMicRecorderIframe(iframe) {
    var title = iframe.title || "";
    return title.indexOf("mic_recorder") !== -1;
  }
  function collapseHiddenSpeechIframes() {
    doc.querySelectorAll("iframe").forEach(function (iframe) {
      if (isMicRecorderIframe(iframe)) return;
      var srcdoc = "";
      try { srcdoc = iframe.srcdoc || iframe.getAttribute("srcdoc") || ""; } catch (e) {}
      // 可見嘅「播放」player 唔好收起
      if (srcdoc.indexOf("speech-player-ui") !== -1) return;
      var marker = srcdoc.indexOf("sync-speech-text:") !== -1
        || srcdoc.indexOf("speak-nonce:") !== -1
        || srcdoc.indexOf("stop-nonce:") !== -1
        || srcdoc.indexOf("__hkParkingSpeech") !== -1;
      var hAttr = parseInt(iframe.getAttribute("height") || "-1", 10);
      var hStyle = parseInt(iframe.style.height || "-1", 10);
      var tiny = (hAttr >= 0 && hAttr <= 1) || (hStyle >= 0 && hStyle <= 1);
      if (marker || tiny) collapseIframeContainer(iframe);
    });
  }

  function broadcastStopSpeechPlayers() {
    doc.querySelectorAll("iframe").forEach(function (iframe) {
      try {
        iframe.contentWindow && iframe.contentWindow.postMessage(
          { type: "hk-parking-stop-speech" },
          "*"
        );
      } catch (e) {}
    });
  }

  function micVoiceIconSvg() {
    // Google／Cursor 風格：直立收音咪（非舞台／唱歌咪）
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="1.1em" height="1.1em" fill="currentColor" aria-hidden="true" style="flex-shrink:0;display:block;"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>';
  }
  function micStopIconSvg() {
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="1.05em" height="1.05em" fill="currentColor" aria-hidden="true" style="flex-shrink:0;display:block;"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>';
  }

  function restoreMicIframeVisibility(iframe) {
    // 若先前被 collapseHiddenSpeechIframes 誤收起，還原可見性
    iframe.style.position = "";
    iframe.style.left = "";
    iframe.style.top = "";
    iframe.style.opacity = "";
    iframe.style.pointerEvents = "";
    iframe.style.border = "";
    iframe.style.height = "2.5rem";
    iframe.style.minHeight = "2.5rem";
    iframe.style.width = "100%";
    var wrap = iframe.closest('[data-testid="stElementContainer"]');
    if (wrap) {
      wrap.style.position = "";
      wrap.style.width = "";
      wrap.style.height = "";
      wrap.style.minHeight = "";
      wrap.style.margin = "";
      wrap.style.padding = "";
      wrap.style.overflow = "";
      wrap.style.border = "";
    }
  }

  function styleMicButtons() {
    doc.querySelectorAll('iframe[title*="mic_recorder"]').forEach(function (iframe) {
      restoreMicIframeVisibility(iframe);
      if (!iframe.dataset.stopSpeechBound) {
        iframe.dataset.stopSpeechBound = "1";
        // 語音掣手勢內解鎖 Media／TTS（先停舊播，再 unlock），之後結果頁先可自動播
        iframe.addEventListener("pointerdown", function () {
          stopSuggestionPlaybackOnly();
          unlockSpeechNow();
        }, true);
        iframe.addEventListener("click", function () {
          stopSuggestionPlaybackOnly();
          unlockSpeechNow();
        }, true);
      }
      try {
        var btn = iframe.contentDocument && iframe.contentDocument.querySelector("button");
        if (!btn) return;
        btn.style.height = "2.5rem";
        btn.style.minHeight = "2.5rem";
        btn.style.width = "100%";
        btn.style.padding = "0 0.6rem";
        btn.style.margin = "0";
        btn.style.borderRadius = "0.5rem";
        btn.style.boxSizing = "border-box";
        btn.style.display = "flex";
        btn.style.alignItems = "center";
        btn.style.justifyContent = "center";
        btn.style.gap = "0.35rem";
        btn.style.fontSize = "1rem";
        btn.style.fontFamily = '"Source Sans Pro", "Segoe UI", sans-serif';
        btn.style.fontWeight = "400";
        btn.style.lineHeight = "1";
        btn.style.whiteSpace = "nowrap";

        var label = (btn.textContent || "").replace(/\\s+/g, "");
        var mode = label.indexOf("停止") !== -1 ? "stop" : "voice";
        if (btn.dataset.iconMode !== mode || !btn.querySelector("svg")) {
          btn.dataset.iconMode = mode;
          var icon = mode === "stop" ? micStopIconSvg() : micVoiceIconSvg();
          var text = mode === "stop" ? "停止" : "語音";
          btn.innerHTML = icon + "<span>" + text + "</span>";
        }

        if (!btn.dataset.stopSpeechBound) {
          btn.dataset.stopSpeechBound = "1";
          btn.addEventListener("click", function () {
            stopSuggestionPlaybackOnly();
            unlockSpeechNow();
          }, true);
          btn.addEventListener("pointerdown", function () {
            stopSuggestionPlaybackOnly();
            unlockSpeechNow();
          }, true);
        }
      } catch (e) {}
    });
  }

  var MZD_HOVER_RED = "#ff4b4b";
  var MZD_BTN_BORDER = "rgba(250, 250, 250, 0.2)";
  var MZD_BTN_TEXT = "rgba(250, 250, 250, 0.95)";
  var MZD_BTN_BG = "#0e1117";

  function setMicLikeColors(btn, hovered) {
    var border = hovered ? MZD_HOVER_RED : MZD_BTN_BORDER;
    var color = hovered ? MZD_HOVER_RED : MZD_BTN_TEXT;
    btn.style.setProperty("border-color", border, "important");
    btn.style.setProperty("color", color, "important");
    btn.style.setProperty("background-color", MZD_BTN_BG, "important");
    btn.style.setProperty("background-image", "none", "important");
    btn.querySelectorAll("p, span, svg").forEach(function (el) {
      el.style.setProperty("color", color, "important");
      el.style.setProperty("fill", color, "important");
    });
  }

  function bindMicLikeHover(btn) {
    if (btn.dataset.mzdHoverBound) return;
    btn.dataset.mzdHoverBound = "1";
    btn.classList.add("mzd-mic-like");
    setMicLikeColors(btn, false);
    btn.addEventListener("mouseenter", function () { setMicLikeColors(btn, true); });
    btn.addEventListener("mouseleave", function () { setMicLikeColors(btn, false); });
    btn.addEventListener("focus", function () { setMicLikeColors(btn, true); });
    btn.addEventListener("blur", function () { setMicLikeColors(btn, false); });
  }

  function isMicLikeLabel(label) {
    return label.indexOf("查詢") !== -1
      || label.indexOf("停止播放") !== -1
      || label.indexOf("播放") !== -1
      || label.indexOf("再聽一次") !== -1;
  }

  function styleMicLikeActionButtons() {
    doc.querySelectorAll('div[data-testid="stFormSubmitButton"] button').forEach(bindMicLikeHover);
    doc.querySelectorAll("button").forEach(function (btn) {
      var label = (btn.textContent || "").replace(/\\s+/g, "");
      if (!isMicLikeLabel(label)) return;
      bindMicLikeHover(btn);
    });
  }

  function bindQueryStopSpeech() {
    doc.querySelectorAll('div[data-testid="stFormSubmitButton"] button').forEach(function (btn) {
      if (btn.dataset.stopSpeechBound) return;
      btn.dataset.stopSpeechBound = "1";
      // 查詢掣手勢內解鎖 Media／TTS（先停舊播，再 unlock），結果頁自動朗讀先可喺 iOS／手機出聲
      btn.addEventListener("pointerdown", function () {
        stopSuggestionPlaybackOnly();
        unlockSpeechNow();
      }, true);
      btn.addEventListener("click", function () {
        stopSuggestionPlaybackOnly();
        unlockSpeechNow();
      }, true);
    });
  }

  function isSpeechToggleLabel(label) {
    return label.indexOf("停止播放") !== -1
      || label.indexOf("播放") !== -1
      || label.indexOf("再聽一次") !== -1;
  }

  function bindSuggestionSpeechButtons() {
    doc.querySelectorAll("button").forEach(function (btn) {
      var label = (btn.textContent || "").replace(/\\s+/g, "");
      if (!isSpeechToggleLabel(label)) return;
      bindMicLikeHover(btn);
      if (btn.dataset.speechToggleBound) return;
      btn.dataset.speechToggleBound = "1";
      // 前端即時 stop／replay；唔 preventDefault，等 Streamlit 更新掣文字
      btn.addEventListener("pointerdown", toggleSuggestionSpeechNow, true);
      // iOS 部分情況 pointerdown 唔夠；click 再補一次 unlock＋播
      btn.addEventListener("click", toggleSuggestionSpeechNow, true);
    });
  }

  function hideSpeechEndedSentinel() {
    doc.querySelectorAll("button").forEach(function (btn) {
      if ((btn.textContent || "").trim() !== "SPEECH_ENDED") return;
      var el = btn.closest('[data-testid="stElementContainer"]')
        || btn.closest('div[data-testid="stButton"]')
        || btn.parentElement;
      if (el) {
        el.style.position = "absolute";
        el.style.width = "0";
        el.style.height = "0";
        el.style.minHeight = "0";
        el.style.margin = "0";
        el.style.padding = "0";
        el.style.overflow = "hidden";
        el.style.opacity = "0";
        el.style.pointerEvents = "none";
        el.style.border = "none";
      }
      btn.style.position = "fixed";
      btn.style.left = "-10000px";
      btn.style.top = "0";
      btn.style.width = "1px";
      btn.style.height = "1px";
      btn.style.opacity = "0";
    });
  }

  // 與 format_hong_kong_now() 一致：只改 .mzd-hk-time 文字，唔重建 DOM
  var MZD_WEEKDAY = { Sun: "日", Mon: "一", Tue: "二", Wed: "三", Thu: "四", Fri: "五", Sat: "六" };
  function formatHongKongNow() {
    var parts = {};
    new Intl.DateTimeFormat("en-US", {
      timeZone: "Asia/Hong_Kong",
      year: "numeric",
      month: "numeric",
      day: "numeric",
      weekday: "short",
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
      hour12: true
    }).formatToParts(new Date()).forEach(function (p) {
      if (p.type !== "literal") parts[p.type] = p.value;
    });
    var wd = MZD_WEEKDAY[parts.weekday] || parts.weekday || "";
    var hour = String(parseInt(parts.hour, 10) || 0);
    var suffix = (parts.dayPeriod || "").toLowerCase().replace(/\\./g, "");
    return parts.year + "年" + parts.month + "月" + parts.day + "日（" + wd + "）"
      + hour + ":" + parts.minute + ":" + parts.second + suffix;
  }
  function tickHkClock() {
    var el = doc.querySelector(".mzd-hk-time");
    if (!el) return;
    var text = formatHongKongNow();
    if (el.textContent !== text) el.textContent = text;
  }

  styleMicButtons();
  styleMicLikeActionButtons();
  collapseJsEvalIframes();
  collapseHiddenSpeechIframes();
  bindQueryStopSpeech();
  bindSuggestionSpeechButtons();
  hideSpeechEndedSentinel();
  tickHkClock();
  setInterval(function () {
    styleMicButtons();
    styleMicLikeActionButtons();
    collapseJsEvalIframes();
    collapseHiddenSpeechIframes();
    bindQueryStopSpeech();
    bindSuggestionSpeechButtons();
    hideSpeechEndedSentinel();
    tickHkClock();
  }, 400);
})();
</script>
        """,
        height=1,
    )


def apply_pending_input(widget_key: str, pending_key: str) -> None:
    """必須在對應 text_input 建立之前呼叫；把語音結果寫入 widget state。"""
    if pending_key not in st.session_state:
        return
    # 只喺呢度改 widget key，避免 StreamlitAPIException
    st.session_state[widget_key] = st.session_state.pop(pending_key)


def handle_voice_transcription(
    audio: dict | None,
    *,
    last_id_key: str,
    pending_key: str,
    auto_submit_key: str,
    feedback_key: str,
) -> None:
    """辨識後只寫入 pending_key（唔好直接改 text_input 的 key），再 rerun。"""
    if not pending_key.startswith("_pending_"):
        raise ValueError(f"pending_key 必須係暫存鍵，收到：{pending_key}")
    if not audio or audio.get("id") == st.session_state.get(last_id_key):
        return
    st.session_state[last_id_key] = audio["id"]
    with st.spinner("正在辨識語音..."):
        try:
            voice_text = transcribe_audio(audio)
            service_error = None
        except Exception as e:
            voice_text = ""
            service_error = str(e)

    # 成功／空白／錯誤都更新；空字串會清空輸入框
    st.session_state[pending_key] = (voice_text or "").strip()
    if service_error:
        st.session_state[feedback_key] = (
            "error",
            f"語音辨識服務暫時無法使用：{service_error}",
        )
    elif st.session_state[pending_key]:
        st.session_state.pop(feedback_key, None)
        st.session_state[auto_submit_key] = True
    else:
        st.session_state[feedback_key] = (
            "warning",
            "未能辨識語音，請再說一次或改用文字輸入。",
        )
    st.rerun()


def render_voice_feedback(feedback_key: str) -> None:
    """全寬顯示語音回饋（避免擠喺窄欄）。"""
    feedback = st.session_state.pop(feedback_key, None)
    if not feedback:
        return
    level, message = feedback
    if level == "error":
        st.error(message)
    else:
        st.warning(message)


inject_layout_css()

# SPEECH_ENDED sentinel 放頁頂並以 CSS 收起，避免插入結果區撐開 margin b／c。
# 唔好喺呢度 st.rerun()：撳掣本身已觸發重跑；若再 rerun 會中斷腳本，
# 結果表（同標題／查詢列）未畫就消失。只改 session 狀態，讓本輪繼續畫完。
if "_suggestion_speaking" not in st.session_state:
    st.session_state._suggestion_speaking = False
st.markdown(
    '<span class="mzd-speech-ended-anchor" aria-hidden="true"></span>',
    unsafe_allow_html=True,
)
if st.button(SPEECH_ENDED_SENTINEL_LABEL, key="_suggestion_speech_ended"):
    st.session_state._suggestion_speaking = False
    st.session_state.pop("_pending_speak_text", None)

render_brand_header()
render_header_context_panel()

# 語音結果必須喺 text_input(key="user_query") 之前套用
apply_pending_input("user_query", "_pending_user_query")
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# 文字輸入 + 查詢 + 語音同一列（Enter 或按「查詢」皆可送出）
st.caption("輸入查詢（例如：天水圍今晚八點邊度較易泊？）")
form_col, voice_col = st.columns([6.2, 1], vertical_alignment="bottom")
with form_col:
    with st.form("query_form", border=False):
        input_col, submit_col = st.columns([5.2, 1], vertical_alignment="bottom")
        with input_col:
            user_query = st.text_input(
                "輸入查詢",
                placeholder="輸入你的問題...",
                label_visibility="collapsed",
                key="user_query",
            )
        with submit_col:
            submitted = st.form_submit_button(
                "🔍 查詢",
                use_container_width=True,
                type="secondary",
            )
with voice_col:
    audio = mic_recorder(
        start_prompt="語音",
        stop_prompt="停止",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="voice_input",
    )
# 只寫 _pending_user_query，唔會喺 widget 建立後改 user_query
handle_voice_transcription(
    audio,
    last_id_key="_last_voice_audio_id",
    pending_key="_pending_user_query",
    auto_submit_key="auto_submit_query",
    feedback_key="_voice_feedback",
)
render_voice_feedback("_voice_feedback")



@st.cache_data(ttl=600, show_spinner="")
def load_data():
    """從 Supabase 載入空位歷史（香港時間；cache 10 分鐘）。"""
    df = load_vacancy_frame()
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    # 半小時槽位：0=12:00am … 30=3:00pm、31=3:30pm … 47=11:30pm
    df["half_hour"] = (
        df["collected_at"].dt.hour * 2
        + (df["collected_at"].dt.minute >= 30).astype(int)
    )
    df["dow"] = df["collected_at"].dt.dayofweek
    # 結果表「地區」：Google reverse geocode 短名；未填則退回十八區
    df["sub_district"] = df["google_area"].fillna(df["district"])
    return df


@st.cache_data(ttl=6 * 3600)
def load_carpark_fee_rules() -> dict[str, dict]:
    """
    從政府 API 取得私家車收費規則。
    回傳 park_id -> {hourlyCharges, dayNightParks}；多數車場可能無資料。
    """
    try:
        response = requests.get(
            CARPARK_INFO_VACANCY_URL,
            params={
                "data": "info",
                "lang": "zh_TW",
                "vehicleTypes": "privateCar",
            },
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    except Exception:
        return {}

    out: dict[str, dict] = {}
    for item in results:
        park_id = str(item.get("park_Id") or "").strip()
        if not park_id:
            continue
        private_car = item.get("privateCar") or {}
        hourly = private_car.get("hourlyCharges") or []
        day_night = private_car.get("dayNightParks") or []
        if hourly or day_night:
            out[park_id] = {
                "hourlyCharges": hourly,
                "dayNightParks": day_night,
            }
    return out


def _minutes_from_hhmm(value: str | None) -> int | None:
    """將 '08:00'／'24:00' 轉成分鐘數（0–1440）。"""
    if not value or not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) < 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if hour == 24 and minute == 0:
        return 24 * 60
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour * 60 + minute
    return None


def _time_in_period(now_m: int, start_m: int, end_m: int) -> bool:
    """判斷 now_m 是否落在 [start, end)（支援通宵時段）。"""
    if start_m == end_m:
        return True
    if start_m < end_m:
        return start_m <= now_m < end_m
    return now_m >= start_m or now_m < end_m


def _fee_rule_applies(rule: dict, weekday_en: str, now_m: int) -> bool:
    weekdays = rule.get("weekdays") or []
    if weekday_en not in weekdays:
        return False
    start_m = _minutes_from_hhmm(rule.get("periodStart") or "00:00")
    end_m = _minutes_from_hhmm(rule.get("periodEnd") or "24:00")
    if start_m is None or end_m is None:
        return False
    return _time_in_period(now_m, start_m, end_m)


def _fee_period_length(rule: dict) -> int:
    start_m = _minutes_from_hhmm(rule.get("periodStart") or "00:00")
    end_m = _minutes_from_hhmm(rule.get("periodEnd") or "24:00")
    if start_m is None or end_m is None:
        return 24 * 60
    if start_m < end_m:
        return end_m - start_m
    if start_m == end_m:
        return 24 * 60
    return (24 * 60 - start_m) + end_m


def _format_hkd(price) -> str | None:
    try:
        amount = float(price)
    except (TypeError, ValueError):
        return None
    if amount == int(amount):
        return f"${int(amount)}"
    return f"${amount:.1f}"


def format_carpark_fee(
    park_id: str,
    day_cn: str,
    hour: int,
    minute: int,
    fee_map: dict[str, dict] | None = None,
) -> str:
    """
    依查詢星期／時間，取出適用時租（無則日夜泊套餐；再無則 —）。
    """
    rules = (fee_map if fee_map is not None else load_carpark_fee_rules()).get(
        str(park_id)
    )
    if not rules:
        return "未提供"

    weekday_en = DAY_CN_TO_EN.get(day_cn) or DOW_TO_EN.get(
        DAY_TO_NUM.get(day_cn, hong_kong_now().weekday()), "MON"
    )
    now_m = int(hour) * 60 + int(minute)

    hourly_matches = [
        rule
        for rule in (rules.get("hourlyCharges") or [])
        if _fee_rule_applies(rule, weekday_en, now_m)
    ]
    if hourly_matches:
        hourly_matches.sort(key=_fee_period_length)
        rule = hourly_matches[0]
        price_str = _format_hkd(rule.get("price"))
        if not price_str:
            return "未提供"
        unit = "半小時" if rule.get("type") == "half-hourly" else "小時"
        return f"{price_str}／{unit}"

    day_night_matches = [
        rule
        for rule in (rules.get("dayNightParks") or [])
        if _fee_rule_applies(rule, weekday_en, now_m)
    ]
    if day_night_matches:
        day_night_matches.sort(key=_fee_period_length)
        rule = day_night_matches[0]
        price_str = _format_hkd(rule.get("price"))
        if not price_str:
            return "未提供"
        unit = DAYNIGHT_FEE_LABEL.get(rule.get("type") or "", "套餐")
        return f"{price_str}／{unit}"

    return "未提供"


df = load_data()


def _normalize_park_name(name: str) -> str:
    """正規化名稱以便比對重複車場（忽略大小寫、空白、停車場／車場後綴）。"""
    s = (name or "").strip().lower()
    s = re.sub(r"[\s\-_/·．.]+", "", s)
    s = s.replace("停車場", "").replace("車場", "")
    s = re.sub(r"parking|carpark", "", s, flags=re.I)
    return s


def _prefer_canonical_park_id(park_id: str) -> tuple:
    """排序用：純數字 id 優先於 tdc*，其餘按字串。"""
    pid = str(park_id)
    return (0 if pid.isdigit() else 1, pid)


def dedupe_equivalent_carparks(
    parks: pd.DataFrame,
    *,
    id_col: str = "park_id",
    name_col: str = "name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    max_distance_km: float = EQUIVALENT_CARPARK_MAX_KM,
    prefer_higher_col: str | None = None,
) -> pd.DataFrame:
    """
    合併極近且正規化名稱相同的重複紀錄，只保留一筆。
    prefer_higher_col 有值時保留該欄較高者；否則偏好較「標準」的 park_id。
    """
    if parks.empty or lat_col not in parks.columns or lon_col not in parks.columns:
        return parks.copy()

    work = parks.dropna(subset=[lat_col, lon_col]).copy()
    if work.empty:
        return parks.iloc[0:0].copy()

    work["_norm_name"] = work[name_col].map(_normalize_park_name)
    work = work.reset_index(drop=True)
    n = len(work)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    lats = work[lat_col].astype(float).tolist()
    lons = work[lon_col].astype(float).tolist()
    norms = work["_norm_name"].tolist()
    for i in range(n):
        if not norms[i]:
            continue
        for j in range(i + 1, n):
            if norms[i] != norms[j]:
                continue
            if _haversine_km(lats[i], lons[i], lats[j], lons[j]) <= max_distance_km:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep_indices: list[int] = []
    for members in clusters.values():
        if len(members) == 1:
            keep_indices.append(members[0])
            continue
        group = work.iloc[members]
        if prefer_higher_col and prefer_higher_col in group.columns:
            best = group.sort_values(
                prefer_higher_col, ascending=False, kind="mergesort"
            ).iloc[0]
            keep_indices.append(int(best.name))
        else:
            best_idx = min(
                members,
                key=lambda idx: _prefer_canonical_park_id(work.iloc[idx][id_col]),
            )
            keep_indices.append(best_idx)

    out = work.iloc[sorted(keep_indices)].drop(columns=["_norm_name"])
    return out.reset_index(drop=True)


def carparks_near_coords(
    lat: float,
    lon: float,
    *,
    radius_km: float = NEARBY_RADIUS_KM,
    fallback_km: float = NEARBY_RADIUS_FALLBACK_KM,
) -> tuple[pd.DataFrame, float | None]:
    """
    依座標篩選附近唯一停車場（先 3km，沒有再放寬至 5km）。
    回傳 (車場 DataFrame含 distance_km, 實際使用半徑)；仍無則半徑為 None。
    """
    parks = (
        df[["park_id", "name", "district", "sub_district", "latitude", "longitude"]]
        .drop_duplicates(subset=["park_id"])
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )
    if parks.empty:
        empty = parks.copy()
        empty["distance_km"] = pd.Series(dtype=float)
        return empty, None

    parks["distance_km"] = [
        _haversine_km(lat, lon, float(plat), float(plon))
        for plat, plon in zip(parks["latitude"], parks["longitude"])
    ]
    for radius in (radius_km, fallback_km):
        filtered = parks[parks["distance_km"] <= radius].sort_values(
            "distance_km", kind="mergesort"
        )
        if not filtered.empty:
            deduped = dedupe_equivalent_carparks(filtered)
            return deduped.sort_values(
                "distance_km", kind="mergesort"
            ).reset_index(drop=True), float(radius)
    return parks.iloc[0:0].copy(), None

def call_deepseek(prompt: str) -> str:
    """呼叫 DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"DeepSeek API 錯誤：{e}")
        return ""

def parse_top_n_from_query(query: str):
    """從查詢解析要顯示幾個停車場；找不到則回傳 None。"""
    # 「邊兩個／哪三個／哪4個」等
    m = re.search(r"[邊哪]([一二两三兩四五六七八九十壹\d]+)個", query)
    if m:
        token = m.group(1)
        if token.isdigit():
            n = int(token)
            return n if n > 0 else None
        if token in CN_NUM:
            return CN_NUM[token]

    # 「邊個／哪個」（未指明數量，視為 1）
    if re.search(r"[邊哪]個", query):
        return 1

    return None


def ordinal(n: int) -> str:
    """將正整數轉成 1st / 2nd / 3rd …"""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def normalize_half_hour(hour: int, minute: int) -> tuple[int, int]:
    """將時間規範成半小時制（分鐘只為 0 或 30）。"""
    hour = int(hour) % 24
    minute = int(minute)
    if minute < 15:
        return hour, 0
    if minute < 45:
        return hour, 30
    return (hour + 1) % 24, 0


def to_half_hour_slot(hour: int, minute: int = 0) -> int:
    """將小時與分鐘轉成半小時槽位（0–47）。"""
    hour, minute = normalize_half_hour(hour, minute)
    return hour * 2 + (1 if minute == 30 else 0)


def format_time_12h(hour: int, minute: int = 0) -> str:
    """將時間轉為 12 小時制，例如 (15, 0) → 3:00pm、(15, 30) → 3:30pm。"""
    hour, minute = normalize_half_hour(hour, minute)
    period = "am" if hour < 12 else "pm"
    h12 = hour % 12 or 12
    return f"{h12}:{minute:02d}{period}"


def next_half_hour_datetime(now=None) -> datetime:
    """未指明時間時：取「未來 1–30 分鐘」內的下一個半小時（可跨日）。

    例如 3:56pm→當日 4:00pm、11:47pm→翌日 12:00am。
    """
    now = now or hong_kong_now()
    total_mins = now.hour * 60 + now.minute
    next_mins = ((total_mins // 30) + 1) * 30
    day_offset = next_mins // (24 * 60)
    next_mins %= 24 * 60
    base = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
    return base.replace(hour=next_mins // 60, minute=next_mins % 60)


def next_half_hour_time(now=None):
    """未指明時間時：取下一個半小時的 (hour, minute)。"""
    dt = next_half_hour_datetime(now)
    return dt.hour, dt.minute


def is_coming_soon_query(query: str) -> bool:
    """查詢是否指「嚟緊／即將」等即將到來的意思。"""
    q = (query or "").replace(" ", "")
    return bool(re.search(r"嚟緊|黎緊|即將|就嚟|快將|快到", q))


def has_explicit_day_period(query: str) -> bool:
    """句子是否已標明上／下午、夜晚、am/pm、嚟緊等時段。"""
    q = (query or "").replace(" ", "")
    return bool(
        re.search(
            r"中午|午晝|正午|晏晝|半夜|午夜|零晨|凌晨|清晨|夜晚|晚上|深夜|"
            r"上午|下午|上晝|下晝|早上|朝早|傍晚|黃昏|黄昏|"
            r"嚟緊|黎緊|即將|就嚟|快將|快到|"
            r"今早|今朝|今晨|聽朝|明早|今晚|聽晚|明晚|後晚|大後晚|今夜|明夜|"
            r"a\.?m\.?|p\.?m\.?",
            q,
            flags=re.I,
        )
    )


def has_unambiguous_24h_clock(query: str) -> bool:
    """是否為明確 24 小時制鐘點（如 15:30／15點），無需再標上午／下午。"""
    q = (query or "").replace(" ", "")
    m = re.search(r"(?<!\d)([0-2]?\d)[:：]\d{1,2}", q)
    if m and int(m.group(1)) >= 13:
        return True
    m_digit = re.search(r"(?<!\d)([1-2]?\d)[點时時]", q)
    if m_digit and int(m_digit.group(1)) >= 13:
        return True
    return False


def coming_soon_hour_candidates(
    hour: int, minute: int, query: str
) -> list[tuple[int, int]]:
    """「嚟緊」鐘點候選：未標時段則上／下午（或午夜／中午）都納入，稍後取最近未來。

    例如「嚟緊兩點」→ 2:00am 與 2:00pm；「嚟緊12點」→ 12:00am 與 12:00pm。
    已寫「晚上兩點」「15:30」等則只保留該鐘點。
    """
    hour, minute = normalize_half_hour(hour, minute)
    q = (query or "").replace(" ", "")
    if has_explicit_day_period(query):
        return [(hour, minute)]
    # 明確 24 小時制（15:30／15點）→ 不擴成 am/pm 兩候選
    if re.search(r"(?<!\d)([01]?\d|2[0-3])[:：]\d{1,2}", q):
        return [(hour, minute)]
    m_digit = re.search(r"(?<!\d)([1-2]?\d)[點时時]", q)
    if m_digit and int(m_digit.group(1)) >= 13:
        return [(hour, minute)]

    face = hour % 12
    if face == 0:
        return [(0, minute), (12, minute)]
    return [(face, minute), (face + 12, minute)]


def soonest_future_datetime(
    hour_minute_pairs: list[tuple[int, int]], now=None
) -> datetime:
    """從候選鐘點中，取現在起下一次即將到來的日期時間。"""
    now = (now or hong_kong_now()).replace(second=0, microsecond=0)
    best = None
    for offset in range(0, 8):
        base = (now + timedelta(days=offset)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for hour, minute in hour_minute_pairs:
            hour, minute = normalize_half_hour(hour, minute)
            when = base.replace(hour=hour, minute=minute)
            if when >= now and (best is None or when < best):
                best = when
        if best is not None:
            return best
    # 理論上不會到這裡；保底回傳一週後第一個候選
    hour, minute = normalize_half_hour(*hour_minute_pairs[0])
    return (now + timedelta(days=7)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


def resolve_coming_soon_intent(intent: dict, query: str, now=None) -> dict:
    """「嚟緊」＋鐘點：改為下一次即將到來的該鐘點（未標時段則 am/pm 取近者）。"""
    now = now or hong_kong_now()
    raw_hour = intent.get("hour")
    raw_minute = intent.get("minute")
    if not intent_has_clock_time(raw_hour):
        dt = next_half_hour_datetime(now)
    else:
        minute = (
            0
            if raw_minute is None or str(raw_minute).strip() in ("", "null", "None")
            else int(raw_minute)
        )
        hour = int(raw_hour)
        hour, minute = normalize_half_hour(hour, minute)
        candidates = coming_soon_hour_candidates(hour, minute, query)
        dt = soonest_future_datetime(candidates, now)
    intent["hour"] = dt.hour
    intent["minute"] = dt.minute
    intent["day"] = WEEKDAY_CN[dt.weekday()]
    return intent


def has_explicit_clock_time(query: str) -> bool:
    """查詢是否有具體鐘點（不含「後晚／聽晚」等相對日期用語）。"""
    q = query.replace(" ", "")
    # 先去掉相對日期與「而家／現在」，避免誤判
    q = re.sub(
        r"大後日|大後晚|後日|後晚|後天|聽日|聽晚|明天|明晚|明日|今日|今晚|今天|今夜",
        "",
        q,
    )
    q = re.sub(r"嚟緊|黎緊|即將|就嚟|快將|快到", "", q)
    q = re.sub(r"而家|依家|現在|目前|即刻|此刻|呢一刻|right\s*now|\bnow\b", "", q, flags=re.I)
    if re.search(r"\d{1,2}[:：]\d{1,2}", q):
        return True
    if re.search(r"\d{1,2}[點时時]", q):
        return True
    if re.search(r"[一二兩三两四五六七八九十廿][點时時]", q):
        return True
    if re.search(r"(上午|下午|早上|朝早|凌晨|夜晚|晚上|傍晚)\d", q):
        return True
    if re.search(r"(上午|下午|早上|朝早|凌晨|夜晚|晚上|傍晚)[一二兩三两四五六七八九十廿]", q):
        return True
    if "中午" in q or "午晝" in q:
        return True
    if re.search(r"(a\.?m\.?|p\.?m\.?)", q, re.I):
        return True
    return False


def is_now_query(query: str) -> bool:
    """查詢是否指「而家／現在」等即時意思。"""
    q = query.replace(" ", "")
    return bool(
        re.search(
            r"而家|依家|現在|目前|即刻|此刻|呢一刻|right\s*now|\bnow\b",
            q,
            flags=re.I,
        )
    )


def add_dense_ranks(df: pd.DataFrame, value_col: str = "預測空位") -> pd.DataFrame:
    """依預測空位密集聚名，同分同名次（1st、2nd、3rd…）。"""
    ranked = df.sort_values(value_col, ascending=False, kind="mergesort").reset_index(drop=True)
    # 空字串欄名：排名欄不顯示標題
    ranked.insert(0, "", ranked[value_col].rank(method="dense", ascending=False).astype(int).map(ordinal))
    return ranked


def select_top_with_ties(df: pd.DataFrame, top_n: int, value_col: str = "預測空位") -> pd.DataFrame:
    """取前 top_n 個；若第 N 個與後續同分，一併納入，並加上密集聚名。"""
    if df.empty or top_n <= 0:
        return df.iloc[0:0].copy()

    sorted_df = df.sort_values(value_col, ascending=False, kind="mergesort").reset_index(drop=True)
    if len(sorted_df) > top_n:
        threshold = sorted_df.iloc[top_n - 1][value_col]
        sorted_df = sorted_df[sorted_df[value_col] >= threshold].copy()
    return add_dense_ranks(sorted_df, value_col=value_col)


def select_top_nearby_with_ties(
    df: pd.DataFrame, top_n: int, value_col: str = "預測空位"
) -> pd.DataFrame:
    """
    地標「附近」專用：以距離加權分數取前 top_n（仍顯示原本預測空位）。
    分數 = 預測空位 / (1 + 距離)^power，避免遠距離大車場壟斷榜首。
    排名按加權分數（推薦次序），唔係純空位多寡。
    """
    if df.empty or top_n <= 0:
        return df.iloc[0:0].copy()
    if "距離(公里)" not in df.columns:
        return select_top_with_ties(df, top_n, value_col=value_col)

    work = df.copy()
    dist = pd.to_numeric(work["距離(公里)"], errors="coerce").fillna(
        NEARBY_RADIUS_FALLBACK_KM
    )
    vac = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
    work["_nearby_score"] = vac / ((1.0 + dist) ** NEARBY_DISTANCE_SCORE_POWER)
    ranked = select_top_with_ties(work, top_n, value_col="_nearby_score")
    return ranked.drop(columns=["_nearby_score"], errors="ignore")


RANK_SPOKEN_YUE = {
    "1st": "第一",
    "2nd": "第二",
    "3rd": "第三",
    "4th": "第四",
    "5th": "第五",
}

# 香港人慣常逐字母讀嘅英文縮寫（大小寫不拘）。其餘英文詞會讀成詞，唔拆字母。
SPELL_OUT_LATIN_ACRONYMS = frozenset(
    {
        "TKO",
        "APM",
        "IFC",
        "ICC",
        "MTR",
        "KTR",
        "HKO",
        "HKU",
        "UST",
        "CYT",
        "CWB",
    }
)
_ROMAN_NUMERAL_SPOKEN = {"II": "二", "III": "三", "IV": "四"}
# 英文詞／縮寫＋可選尾隨數字（AIRSIDE、KTR350、apm）
_LATIN_SPEECH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\+[A-Za-z]*)?\d*|\d+[A-Za-z]+")


def _latin_token_to_hk_speech(token: str) -> str:
    """將單個英文／縮寫 token 轉成符合香港口語習慣嘅朗讀形式。"""
    if token.upper() == "M+" or token == "M+":
        return "M plus"

    letters = token
    digits = ""
    m = re.match(r"^([A-Za-z]+)(\d+)$", token)
    if m:
        letters, digits = m.group(1), m.group(2)

    upper = letters.upper()
    if upper in _ROMAN_NUMERAL_SPOKEN:
        spoken = _ROMAN_NUMERAL_SPOKEN[upper]
    elif upper in SPELL_OUT_LATIN_ACRONYMS:
        # TKO / APM → "T K O"／"A P M"，逼 TTS 逐字母
        spoken = " ".join(upper)
    elif letters.isupper() and len(letters) >= 2:
        # AIRSIDE → Airside，避免全大階被逐字母讀
        spoken = letters.title()
    else:
        spoken = letters

    if digits:
        return f"{spoken} {digits}"
    return spoken


def prepare_spoken_text_for_tts(text: str) -> str:
    """朗讀前處理：英文名稱跟香港習慣；日期括號星期改口語，唔改書面顯示。"""
    text = (text or "").strip()
    if not text:
        return text
    text = re.sub(r"\bM\+", "M plus", text, flags=re.IGNORECASE)
    # 「2026年10月27日（二）」→「…星期二」；否則 Edge TTS 只讀「二」，聽落似月份／亂碼
    text = re.sub(r"[（(]([一二三四五六日])[）)]", r"星期\1", text)
    return _LATIN_SPEECH_TOKEN_RE.sub(
        lambda m: _latin_token_to_hk_speech(m.group(0)), text
    )


def _suggestion_rank_groups(
    result_df: pd.DataFrame,
) -> list[tuple[str, list[str], int]]:
    """最多前三名（含同分）→ [(名次, 名稱列表, 預測空位), ...]。"""
    if result_df.empty:
        return []
    rank_col = ""
    unique_ranks = list(dict.fromkeys(result_df[rank_col].tolist()))
    selected_ranks = unique_ranks[:3]
    groups: list[tuple[str, list[str], int]] = []
    for rank in selected_ranks:
        group = result_df[result_df[rank_col] == rank]
        names = [str(n) for n in group["名稱"].tolist()]
        vacancy = int(group.iloc[0]["預測空位"])
        groups.append((rank, names, vacancy))
    return groups


def format_time_spoken_yue(hour: int, minute: int = 0) -> str:
    """口語時間，例如 14:30 → 下晝2點半。"""
    hour, minute = normalize_half_hour(hour, minute)
    if hour < 6:
        period = "凌晨"
    elif hour < 12:
        period = "上晝"
    elif hour == 12:
        period = "中午"
    elif hour < 19:
        period = "下晝"
    else:
        period = "夜晚"
    h12 = hour % 12 or 12
    if minute == 0:
        return f"{period}{h12}點"
    return f"{period}{h12}點半"


def _join_names_spoken_yue(names: list[str]) -> str:
    """口語連接停車場名稱。"""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}同埋{names[1]}"
    return "、".join(names[:-1]) + f"，同埋{names[-1]}"


def format_top_suggestion(result_df: pd.DataFrame, hour: int, minute: int = 0) -> str:
    """依最多前三名（1st／2nd／3rd，各名次含全部分同分）產生建議文案。"""
    groups = _suggestion_rank_groups(result_df)
    if not groups:
        return ""

    time_str = format_time_12h(hour, minute)
    total = sum(len(names) for _, names, _ in groups)

    if total == 1:
        _, names, vacancy = groups[0]
        return f"建議於{time_str}前往{names[0]}，預計約有{vacancy}個空位。"

    lines = [f"建議於{time_str}優先考慮以下停車場："]
    for rank, names, vacancy in groups:
        names_text = "、".join(names)
        if len(names) == 1:
            lines.append(f"- {rank}：{names_text}（約{vacancy}個空位）")
        else:
            lines.append(f"- {rank}：{names_text}（各約{vacancy}個空位）")
    return "\n".join(lines)


def format_spoken_suggestion(result_df: pd.DataFrame, hour: int, minute: int = 0) -> str:
    """粵語口語建議（畀司機聽）。"""
    groups = _suggestion_rank_groups(result_df)
    if not groups:
        return ""

    time_spoken = format_time_spoken_yue(hour, minute)
    total = sum(len(names) for _, names, _ in groups)

    if total == 1:
        _, names, vacancy = groups[0]
        raw = (
            f"建議你喺{time_spoken}去{names[0]}，"
            f"預計大約有{vacancy}個空位。"
        )
        return prepare_spoken_text_for_tts(raw)

    parts = [f"建議你喺{time_spoken}優先考慮以下停車場。"]
    for rank, names, vacancy in groups:
        rank_yue = RANK_SPOKEN_YUE.get(rank, rank)
        if len(names) == 1:
            parts.append(
                f"{rank_yue}：{names[0]}，預計大約有{vacancy}個空位。"
            )
        else:
            names_text = _join_names_spoken_yue(names)
            parts.append(
                f"{rank_yue}有{len(names)}個選擇：{names_text}，"
                f"每個預計大約有{vacancy}個空位。"
            )
    return prepare_spoken_text_for_tts("".join(parts))


def _sync_speech_text_to_browser(text: str, *, speaking: bool | None = None) -> None:
    """把建議朗讀稿／播放狀態同步到 parent。"""
    text = prepare_spoken_text_for_tts(text)
    if not text and speaking is None:
        return
    payload = json.dumps(text, ensure_ascii=False)
    speaking_js = "null" if speaking is None else ("true" if speaking else "false")
    nonce = int(st.session_state.get("_speak_nonce", 0))
    components.html(
        f"""
        <script>
        (function () {{
          var root = window.parent;
          var text = {payload};
          if (text) {{
            root.__hkParkingSpeechLastText = text;
            if (root.__hkParkingSpeech) root.__hkParkingSpeech._lastText = text;
          }}
          var speaking = {speaking_js};
          if (speaking !== null) {{
            root.__hkParkingSpeechSpeaking = speaking;
            if (typeof root.__hkParkingSetSpeechSpeakingUi === "function") {{
              root.__hkParkingSetSpeechSpeakingUi(speaking);
            }}
          }}
        }})();
        </script>
        <!-- sync-speech-text:{nonce}:{len(text)}:{speaking_js} -->
        """,
        height=1,
    )


def flush_deferred_speech_browser_ops() -> None:
    """喺結果表之後先執行語音 sync／播／停，避免 iframe 撐開掣上下空隙。"""
    payload = st.session_state.pop("_deferred_speech_sync", None)
    if not payload:
        return
    if payload.get("stop"):
        stop_speech_zh_hk()
    spoken = payload.get("spoken") or ""
    speaking = payload.get("speaking")
    if spoken or speaking is not None:
        _sync_speech_text_to_browser(spoken, speaking=speaking)
    pending = payload.get("pending_speak")
    if pending:
        speak_text_zh_hk(pending, force=True)


def speak_text_zh_hk(text: str, *, force: bool = False) -> None:
    """透過 parent 常駐控制器朗讀（唔依賴會被 rerun 清掉嘅 iframe 生命週期）。"""
    text = prepare_spoken_text_for_tts(text)
    if not text:
        return
    if not force and st.session_state.get("_last_spoken_text") == text:
        return
    st.session_state._last_spoken_text = text
    nonce = int(st.session_state.get("_speak_nonce", 0)) + 1
    st.session_state._speak_nonce = nonce

    payload = json.dumps(text, ensure_ascii=False)
    components.html(
        f"""
        <script>
        (function () {{
          var root = window.parent;
          var text = {payload};
          root.__hkParkingSpeechLastText = text;
          function go() {{
            if (!root.__hkParkingSpeech) return false;
            root.__hkParkingSpeech.speak(text);
            return true;
          }}
          if (!go()) {{
            var n = 0;
            var t = setInterval(function () {{
              n += 1;
              if (go() || n > 40) clearInterval(t);
            }}, 50);
          }}
        }})();
        </script>
        <!-- speak-nonce:{nonce} -->
        """,
        height=1,
    )


def stop_speech_zh_hk() -> None:
    """立即停止瀏覽器語音朗讀。"""
    nonce = int(st.session_state.get("_speak_nonce", 0)) + 1
    st.session_state._speak_nonce = nonce
    st.session_state._suggestion_speaking = False
    st.session_state.pop("_pending_speak_text", None)
    st.session_state.pop("_last_spoken_text", None)
    components.html(
        f"""
        <script>
        (function () {{
          var root = window.parent;
          if (root.__hkParkingSpeech) root.__hkParkingSpeech.stop();
          else if (root.speechSynthesis) {{
            try {{ root.speechSynthesis.cancel(); }} catch (e) {{}}
            try {{ root.speechSynthesis.cancel(); }} catch (e) {{}}
          }}
        }})();
        </script>
        <!-- stop-nonce:{nonce} -->
        """,
        height=1,
    )


@st.cache_data(ttl=6 * 3600, show_spinner="正在產生語音…")
def synthesize_zh_hk_mp3(text: str) -> bytes | None:
    """用 Edge 粵語 Neural 聲線合成 MP3（手機 HTML5 Audio 播放較 speechSynthesis 可靠）。"""
    text = (text or "").strip()
    if not text:
        return None
    # 避免極長稿拖垮 Cloud／base64
    if len(text) > 800:
        text = text[:800]

    try:
        import edge_tts
    except ImportError:
        return None

    async def _run() -> bytes:
        communicate = edge_tts.Communicate(
            text,
            voice="zh-HK-HiuMaanNeural",
            rate="-5%",
        )
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                chunks.append(chunk["data"])
        return b"".join(chunks)

    try:
        return asyncio.run(_run())
    except Exception:
        return None


def render_iframe_speech_player(spoken: str, *, auto_speak: bool = False) -> None:
    """iframe 內建播放掣：優先播伺服器 MP3（iOS 可用），否則退回 Web Speech。

    手動撳掣：同一 iframe 內同步 play()／speak()。
    自動播：用 parent 喺「查詢」手勢解鎖過嘅 Audio（手機／iOS 先得）。
    """
    spoken = prepare_spoken_text_for_tts((spoken or "").strip())
    if not spoken:
        return

    mp3 = synthesize_zh_hk_mp3(spoken)
    payload = json.dumps(spoken, ensure_ascii=False)
    auto_js = "true" if auto_speak else "false"
    nonce = abs(hash(spoken)) % (10**9)

    if mp3:
        audio_src = "data:audio/mpeg;base64," + base64.b64encode(mp3).decode("ascii")
        audio_src_js = json.dumps(audio_src)
        media_js = f"""
  audioSrc = {audio_src_js};
  var aud = new Audio();
  aud.preload = "auto";
  aud.src = audioSrc;
  aud.addEventListener("ended", function () {{ setUi(false); }});
  aud.addEventListener("error", function () {{ setUi(false); }});

  function stopParentMedia() {{
    try {{
      if (window.parent && typeof window.parent.__hkParkingStopUnlockedMedia === "function") {{
        window.parent.__hkParkingStopUnlockedMedia();
      }}
    }} catch (e) {{}}
  }}

  function stopLocalOnly() {{
    try {{ aud.pause(); aud.currentTime = 0; }} catch (e) {{}}
    try {{ if (synth) synth.cancel(); }} catch (e) {{}}
    setUi(false);
  }}

  function stopNow() {{
    stopParentMedia();
    stopLocalOnly();
  }}

  function speakNow() {{
    // 必須喺用戶手勢內同步呼叫 play()（iOS 唔接受 setTimeout 後再 play）
    stopParentMedia();
    try {{ aud.pause(); }} catch (e) {{}}
    try {{ aud.currentTime = 0; }} catch (e) {{}}
    try {{ if (synth) synth.cancel(); }} catch (e) {{}}
    setUi(true);
    var p = aud.play();
    if (p && typeof p.then === "function") {{
      p.catch(function () {{
        setUi(false);
        speakViaWebSpeech();
      }});
    }}
  }}
"""
    else:
        media_js = """
  function stopParentMedia() {
    try {
      if (window.parent && typeof window.parent.__hkParkingStopUnlockedMedia === "function") {
        window.parent.__hkParkingStopUnlockedMedia();
      }
    } catch (e) {}
  }

  function stopLocalOnly() {
    try { if (synth) synth.cancel(); } catch (e) {}
    setUi(false);
  }

  function stopNow() {
    stopParentMedia();
    stopLocalOnly();
  }

  function speakNow() {
    stopParentMedia();
    speakViaWebSpeech();
  }
"""

    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  html, body {{
    margin: 0; padding: 0; background: transparent;
    font-family: "Source Sans", "Segoe UI", sans-serif;
  }}
  #btn {{
    display: flex; align-items: center; justify-content: center;
    width: 100%; height: 2.5rem; min-height: 2.5rem;
    margin: 0; padding: 0 0.6rem; box-sizing: border-box;
    border-radius: 0.5rem;
    border: 1px solid rgba(250, 250, 250, 0.2);
    background: #0e1117; color: rgba(250, 250, 250, 0.95);
    font-size: 1rem; line-height: 1; cursor: pointer;
    -webkit-tap-highlight-color: transparent;
  }}
  #btn:active {{ border-color: #ff4b4b; color: #ff4b4b; }}
</style>
</head>
<body>
<!-- speech-player-ui:{nonce} -->
<button id="btn" type="button">🔊 播放</button>
<script>
(function () {{
  var text = {payload};
  var wantAuto = {auto_js};
  var audioSrc = null;
  var synth = window.speechSynthesis;
  var Utterance = window.SpeechSynthesisUtterance;
  var btn = document.getElementById("btn");
  var uiSpeaking = false;
  var lockUntil = 0;

  function pickVoice(voices) {{
    var list = voices || [];
    return list.find(function (v) {{ return (v.lang || "").toLowerCase().startsWith("zh-hk"); }})
      || list.find(function (v) {{ return (v.lang || "").toLowerCase().includes("hk"); }})
      || list.find(function (v) {{ return (v.lang || "").toLowerCase().startsWith("zh"); }})
      || null;
  }}

  function setUi(speaking) {{
    uiSpeaking = !!speaking;
    btn.textContent = uiSpeaking ? "⏹ 停止播放" : "🔊 播放";
  }}

  function speakViaWebSpeech() {{
    if (!synth || !Utterance || !text) return;
    try {{ synth.resume(); }} catch (e) {{}}
    try {{ synth.cancel(); }} catch (e) {{}}
    var u = new Utterance(text);
    u.lang = "zh-HK";
    u.rate = 0.95;
    var voice = pickVoice(synth.getVoices());
    if (voice) u.voice = voice;
    u.onend = function () {{ setUi(false); }};
    u.onerror = function () {{ setUi(false); }};
    setUi(true);
    synth.speak(u);
  }}

{media_js}

  function onUserPlayToggle(ev) {{
    var now = Date.now();
    if (now < lockUntil) return;
    lockUntil = now + 400;
    if (ev) {{
      try {{ ev.preventDefault(); }} catch (e) {{}}
    }}
    if (uiSpeaking) {{
      stopNow();
      return;
    }}
    speakNow();
  }}

  btn.addEventListener("click", onUserPlayToggle);
  btn.addEventListener("touchend", function (ev) {{
    try {{ ev.preventDefault(); }} catch (e) {{}}
    onUserPlayToggle(ev);
  }}, {{ passive: false }});

  window.addEventListener("message", function (ev) {{
    var data = ev && ev.data;
    if (!data || data.type !== "hk-parking-stop-speech") return;
    // 只停 iframe 本地；唔好 stopParentMedia，否則會打掉查詢手勢嘅 unlock
    stopLocalOnly();
  }});

  try {{ synth && synth.getVoices(); }} catch (e) {{}}

  function tryAutoSpeak() {{
    if (uiSpeaking) return;
    var parent = window.parent;
    // 優先：parent 喺查詢手勢解鎖過嘅 Audio（手機／iOS 自動播）
    if (audioSrc && parent && typeof parent.__hkParkingPlayUnlockedMedia === "function") {{
      var started = false;
      parent.__hkParkingPlayUnlockedMedia(audioSrc, {{
        onStart: function () {{
          started = true;
          setUi(true);
        }},
        onEnded: function () {{ setUi(false); }},
        onError: function () {{
          if (!started) speakNow();
          else setUi(false);
        }}
      }});
      // 未解鎖時 play 可能靜默失敗：稍後再試 iframe 本地播（桌面通常得）
      setTimeout(function () {{
        if (!started && !uiSpeaking) speakNow();
      }}, 500);
      return;
    }}
    // 無 MP3：退回 Web Speech／本地播
    speakNow();
  }}

  if (wantAuto) {{
    setTimeout(tryAutoSpeak, 300);
  }}
}})();
</script>
</body>
</html>
        """,
        height=48,
    )


def render_text_with_speech(
    written: str,
    spoken: str | None = None,
    *,
    auto_speak: bool = True,
    button_key: str = "suggestion_speech_toggle",
    show_written: bool = True,
) -> bool:
    """顯示文字，並用 iframe 內建掣朗讀（手機／iOS 可出聲）。

    回傳值保留相容（而家唔再需要因掣狀態 rerun，固定 False）。
    """
    del button_key  # 舊 Streamlit toggle 已棄用
    written = (written or "").strip()
    spoken = prepare_spoken_text_for_tts(((spoken or written) or "").strip())
    if not written and not spoken:
        return False

    should_auto_speak = auto_speak and st.session_state.get("_auto_speak_token") != spoken
    if should_auto_speak:
        st.session_state._auto_speak_token = spoken

    # 固定右欄掣位（有／無書面文字都一樣），同預測成功排版一致
    text_col, btn_col = st.columns([3.6, 1.2], vertical_alignment="top")
    with text_col:
        if show_written and written:
            compact_lines = []
            for line in written.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("- "):
                    line = "• " + line[2:]
                compact_lines.append(html.escape(line))
            st.markdown(
                '<div style="line-height:1.55;margin:0;padding:0;">'
                + "<br>".join(compact_lines)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="mzd-speech-left-slot" aria-hidden="true"></div>',
                unsafe_allow_html=True,
            )
    with btn_col:
        render_iframe_speech_player(spoken, auto_speak=should_auto_speak)

    # 清除舊 deferred 路徑，避免再走 parent.speechSynthesis 自動播（手機無效）
    st.session_state.pop("_deferred_speech_sync", None)
    st.session_state.pop("_pending_speak_text", None)
    st.session_state._suggestion_speaking = False
    return False


def render_suggestion_with_speech(
    result_df: pd.DataFrame,
    hour: int,
    minute: int = 0,
    *,
    auto_speak: bool = True,
    written: str | None = None,
    spoken: str | None = None,
) -> bool:
    """顯示建議文字；可自動朗讀，並提供再聽／停止按鈕。回傳是否需要延後 rerun。"""
    written = written if written is not None else format_top_suggestion(result_df, hour, minute)
    spoken = spoken if spoken is not None else format_spoken_suggestion(result_df, hour, minute)
    return render_text_with_speech(
        written,
        spoken,
        auto_speak=auto_speak,
        show_written=SHOW_WRITTEN_SUGGESTION,
    )


def _plain_for_speech(text: str) -> str:
    """書面提示轉朗讀稿：去掉 markdown 強調符號。"""
    return prepare_spoken_text_for_tts((text or "").replace("**", "").strip())


def render_feedback_with_speech(
    written: str,
    *,
    spoken: str | None = None,
    auto_speak: bool = True,
    level: str = "warning",
    intent: dict | None = None,
    info_mode: bool = False,
    persist: bool = True,
) -> None:
    """顯示提示／警告，並提供播放掣；新查詢時可自動朗讀（方便駕駛中用耳聽）。"""
    written = (written or "").strip()
    if not written:
        return
    spoken_text = prepare_spoken_text_for_tts(
        ((spoken if spoken is not None else _plain_for_speech(written)) or "").strip()
    )
    if not spoken_text:
        spoken_text = _plain_for_speech(written)

    if persist:
        st.session_state.last_feedback = {
            "written": written,
            "spoken": spoken_text,
            "level": level,
            "intent": intent,
            "info_mode": info_mode,
        }
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_info_answer", None)

    if intent is not None:
        render_intent_summary(intent, info_mode=info_mode)

    if level == "error":
        st.error(written)
    elif level == "info":
        st.info(written)
    else:
        st.warning(written)

    render_text_with_speech(
        written,
        spoken_text,
        auto_speak=auto_speak,
        show_written=False,
    )


def render_saved_feedback(payload: dict, *, auto_speak: bool = False) -> None:
    """重顯示已儲存的提示／警告（含播放掣）。"""
    render_feedback_with_speech(
        payload.get("written") or "",
        spoken=payload.get("spoken"),
        auto_speak=auto_speak,
        level=payload.get("level") or "warning",
        intent=payload.get("intent"),
        info_mode=bool(payload.get("info_mode")),
        persist=False,
    )


def render_saved_prediction(payload: dict, *, auto_speak: bool = False) -> None:
    """重顯示已儲存的預測結果（例如撳「再聽一次」後）。"""
    intent = payload.get("intent") or {}
    result_df = pd.DataFrame(payload.get("result_records") or [])
    hour = int(payload["hour"])
    minute = int(payload["minute"])
    day = payload["day"]
    district_label = payload["district_label"]
    top_n = int(payload["top_n"])
    shown = int(payload["shown"])
    asking_now = bool(payload.get("asking_now"))
    time_str = format_time_12h(hour, minute)

    render_intent_summary(intent)
    if asking_now:
        st.info(
            f"我們的預測系統以半小時為單位。"
            f"你查詢的是「而家／現在」，因此顯示未來 1–30 分鐘內"
            f"下一個半小時（{time_str}）的預測結果。"
        )
    if shown > top_n:
        st.success(
            f"{day} {time_str} {district_label} 較易泊車的停車場"
            f"（前 {top_n} 名，含同分共 {shown} 個）"
        )
    else:
        st.success(f"{day} {time_str} {district_label} 較易泊車的停車場（前 {top_n} 名）")
    # 朗讀掣：success 之下、表格之上、靠右（同資料查詢一致）；空名單都要可聽
    need_rerun = False
    if not result_df.empty:
        need_rerun = render_suggestion_with_speech(
            result_df,
            hour,
            minute,
            auto_speak=auto_speak,
            written=payload.get("written"),
            spoken=payload.get("spoken"),
        )
    else:
        written = (payload.get("written") or "").strip() or (
            f"{day} {time_str} {district_label}暫時沒有可用的預測結果。"
        )
        spoken = (payload.get("spoken") or "").strip() or (
            f"{day} {time_str} {district_label}暫時冇可用嘅預測結果。"
        )
        need_rerun = render_text_with_speech(
            written,
            spoken,
            auto_speak=auto_speak,
            show_written=SHOW_WRITTEN_SUGGESTION,
        )
    render_ranked_table(result_df)
    render_results_map(result_df, intent=intent)
    flush_deferred_speech_browser_ops()
    if need_rerun:
        st.rerun()


_TABLE_HIDDEN_COLS = frozenset({"latitude", "longitude"})


def render_results_map(
    df: pd.DataFrame,
    *,
    intent: dict | None = None,
    height: int = 420,
) -> None:
    """喺結果表下方以 Google Maps 顯示停車場分布。"""
    if df is None or df.empty:
        return
    if not GOOGLE_MAPS_API_KEY:
        st.caption("尚未設定 Google Maps API Key，未能顯示地圖。")
        return
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.caption("結果沒有座標資料，未能顯示地圖。")
        return

    markers: list[dict] = []
    for _, row in df.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (TypeError, ValueError):
            continue
        if pd.isna(lat) or pd.isna(lon):
            continue
        name = str(row.get("名稱") or "").strip() or "停車場"
        details: list[str] = []
        if "預測空位" in df.columns:
            vacancy = row.get("預測空位")
            try:
                vacancy_txt = (
                    str(int(vacancy))
                    if vacancy is not None and not pd.isna(vacancy)
                    else "—"
                )
            except (TypeError, ValueError):
                vacancy_txt = str(vacancy) if vacancy is not None else "—"
            details.append(f"預測空位：{html.escape(vacancy_txt)}")
        fee = row.get("收費")
        if fee is not None and not (isinstance(fee, float) and pd.isna(fee)):
            fee_txt = str(fee).strip()
            if fee_txt:
                details.append(f"收費：{html.escape(fee_txt)}")
        dist = row.get("距離(公里)")
        if dist is not None and not (isinstance(dist, float) and pd.isna(dist)):
            try:
                details.append(f"距離：{float(dist):.1f} 公里")
            except (TypeError, ValueError):
                pass
        rank = row.get("")
        rank_txt = f"{rank} · " if rank not in (None, "") and not (isinstance(rank, float) and pd.isna(rank)) else ""
        detail_html = "".join(f"<br>{d}" for d in details)
        markers.append(
            {
                "lat": lat,
                "lng": lon,
                "title": name,
                "info": (
                    f"<strong>{html.escape(rank_txt + name)}</strong>{detail_html}"
                ),
            }
        )

    if not markers:
        st.caption("結果沒有有效座標，未能顯示地圖。")
        return

    center = None
    intent = intent or {}
    geo_lat, geo_lon = intent.get("geo_lat"), intent.get("geo_lon")
    if geo_lat is not None and geo_lon is not None:
        try:
            center = {
                "lat": float(geo_lat),
                "lng": float(geo_lon),
                "label": str(intent.get("geo_label") or "查詢位置").strip() or "查詢位置",
            }
        except (TypeError, ValueError):
            center = None

    markers_json = json.dumps(markers, ensure_ascii=False)
    center_json = json.dumps(center, ensure_ascii=False)
    maps_src = html.escape(
        "https://maps.googleapis.com/maps/api/js?"
        + urlencode(
            {
                "key": GOOGLE_MAPS_API_KEY,
                "callback": "initResultsMap",
                "language": "zh-HK",
            }
        )
    )

    # Google Maps JS 為主；若 API 未啟用／金鑰被拒，自動退回 Leaflet（同一批 marker）。
    map_html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; background: transparent; }}
    #map {{
      width: 100%;
      height: 100%;
      border-radius: 0.6rem;
      border: 1px solid rgba(128,128,128,0.28);
      overflow: hidden;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script>
    const MARKERS = {markers_json};
    const CENTER = {center_json};
    let usedFallback = false;

    function initLeafletFallback() {{
      if (usedFallback) return;
      usedFallback = true;
      const el = document.getElementById("map");
      el.innerHTML = "";
      // Leaflet 需要明確高度；清掉 Google 殘餘 DOM 後重建
      const map = L.map(el, {{ scrollWheelZoom: true }});
      L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap",
      }}).addTo(map);

      const bounds = [];
      MARKERS.forEach(function (m) {{
        const marker = L.marker([m.lat, m.lng]).addTo(map);
        marker.bindPopup(m.info);
        bounds.push([m.lat, m.lng]);
      }});
      if (CENTER) {{
        const c = L.circleMarker([CENTER.lat, CENTER.lng], {{
          radius: 8,
          color: "#ffffff",
          weight: 2,
          fillColor: "#4285F4",
          fillOpacity: 1,
        }}).addTo(map);
        c.bindPopup(CENTER.label || "查詢位置");
        bounds.push([CENTER.lat, CENTER.lng]);
      }}
      if (bounds.length === 1) {{
        map.setView(bounds[0], 15);
      }} else if (bounds.length > 1) {{
        map.fitBounds(bounds, {{ padding: [40, 40] }});
      }}
      setTimeout(function () {{ map.invalidateSize(); }}, 80);
    }}

    window.gm_authFailure = function () {{
      initLeafletFallback();
    }};

    function initResultsMap() {{
      try {{
        const map = new google.maps.Map(document.getElementById("map"), {{
          zoom: 12,
          center: {{ lat: MARKERS[0].lat, lng: MARKERS[0].lng }},
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: true,
        }});
        const bounds = new google.maps.LatLngBounds();
        const infoWindow = new google.maps.InfoWindow();

        MARKERS.forEach(function (m) {{
          const pos = {{ lat: m.lat, lng: m.lng }};
          bounds.extend(pos);
          const marker = new google.maps.Marker({{
            position: pos,
            map: map,
            title: m.title,
          }});
          marker.addListener("click", function () {{
            infoWindow.setContent(m.info);
            infoWindow.open(map, marker);
          }});
        }});

        if (CENTER) {{
          const cpos = {{ lat: CENTER.lat, lng: CENTER.lng }};
          bounds.extend(cpos);
          new google.maps.Marker({{
            position: cpos,
            map: map,
            title: CENTER.label,
            icon: {{
              path: google.maps.SymbolPath.CIRCLE,
              scale: 8,
              fillColor: "#4285F4",
              fillOpacity: 1,
              strokeColor: "#ffffff",
              strokeWeight: 2,
            }},
          }});
        }}

        if (MARKERS.length === 1 && !CENTER) {{
          map.setCenter({{ lat: MARKERS[0].lat, lng: MARKERS[0].lng }});
          map.setZoom(15);
        }} else {{
          map.fitBounds(bounds, 48);
        }}

        // 若 auth 失敗，Google 仍可能短暫建到 Map；等一下再檢查是否已 fallback
        setTimeout(function () {{
          if (!usedFallback) {{
            const sorry = document.querySelector("#map .gm-err-container, #map .dismissButton");
            if (sorry) initLeafletFallback();
          }}
        }}, 1500);
      }} catch (err) {{
        initLeafletFallback();
      }}
    }}

    // 腳本載入失敗時亦退回
    window.addEventListener("error", function (ev) {{
      if (ev && ev.target && ev.target.src &&
          String(ev.target.src).indexOf("maps.googleapis.com") !== -1) {{
        initLeafletFallback();
      }}
    }}, true);
  </script>
  <script async defer src="{maps_src}"
    onerror="initLeafletFallback()"></script>
</body>
</html>
"""
    components.html(map_html, height=height)


def render_ranked_table(df: pd.DataFrame) -> None:
    """以 HTML 表格顯示結果；同分排名用 rowspan 合併為同一格。"""
    if df.empty:
        st.info("沒有可顯示的停車場。")
        return

    rank_col = ""
    data_cols = [
        c for c in df.columns if c != rank_col and c not in _TABLE_HIDDEN_COLS
    ]

    # 計算每個排名連續區塊的 rowspan
    ranks = df[rank_col].tolist()
    rowspans: list[int | None] = [None] * len(ranks)
    i = 0
    while i < len(ranks):
        j = i + 1
        while j < len(ranks) and ranks[j] == ranks[i]:
            j += 1
        rowspans[i] = j - i
        i = j

    header_cells = "".join(
        f'<th style="text-align:left;padding:0.5rem 0.75rem;border-bottom:1px solid rgba(128,128,128,0.35);">{html.escape(str(col))}</th>'
        for col in ([rank_col] + data_cols)
    )

    body_rows = []
    for pos, (_, row) in enumerate(df.iterrows()):
        cells = []
        span = rowspans[pos]
        if span is not None:
            cells.append(
                f'<td rowspan="{span}" style="vertical-align:middle;padding:0.5rem 0.75rem;'
                f'border-bottom:1px solid rgba(128,128,128,0.25);white-space:nowrap;">'
                f'{html.escape(str(row[rank_col]))}</td>'
            )
        for col in data_cols:
            cells.append(
                f'<td style="padding:0.5rem 0.75rem;border-bottom:1px solid rgba(128,128,128,0.25);">'
                f'{html.escape(str(row[col]))}</td>'
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
<div style="width:100%;overflow-x:auto;margin:0 0 0.75rem 0;
  border:1px solid rgba(128,128,128,0.28);border-radius:0.6rem;">
  <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)



def resolve_relative_day(query: str, today=None):
    """從查詢解析相對日期（粵語），回傳「週一」…「週日」；無法判斷則回傳 None。"""
    today = today or hong_kong_now()
    q = query.replace(" ", "")

    # 由遠至近／較具體先匹配，避免「後日」被「日」誤傷
    relative_offsets = [
        (r"大後日|大後晚", 3),
        (r"後日|後晚|後天", 2),
        (r"聽日|聽晚|明天|明晚|明日", 1),
        (r"今日|今晚|今天|今夜", 0),
    ]
    for pattern, offset in relative_offsets:
        if re.search(pattern, q):
            return WEEKDAY_CN[(today + timedelta(days=offset)).weekday()]

    # 直接寫星期
    for name in WEEKDAY_CN.values():
        if name in q or name.replace("週", "星期") in q:
            return name
    return None


def district_display_name(district: str) -> str:
    """未指定地區時顯示「全港」；否則顯示地點原文（交由 Google 定位）。"""
    if _is_all_hong_kong(district):
        return "全港"
    return str(district or "").strip() or "全港"


def date_for_weekday_cn(day_cn: str, today=None) -> datetime | None:
    """將「週一」…「週日」轉成今天或之後最近的該日日期。"""
    today = today or hong_kong_now()
    target = DAY_TO_NUM.get(day_cn)
    if target is None:
        return None
    delta = (target - today.weekday()) % 7
    return (today + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)


def is_future_day(day_cn: str, today=None) -> bool:
    """日子是否為明天或之後（相對今天）。"""
    today = today or hong_kong_now()
    date_part = date_for_weekday_cn(day_cn, today)
    if date_part is None:
        return False
    return date_part.date() > today.date()


def intent_has_clock_time(hour) -> bool:
    """意圖是否已有具體小時。"""
    return hour is not None and str(hour).strip() not in ("", "null", "None")


def intent_query_datetime(day_cn: str, hour, minute, today=None) -> datetime | None:
    """將意圖的星期＋鐘點組合成具體日期時間；缺資料則回傳 None。"""
    if not intent_has_clock_time(hour):
        return None
    today = today or hong_kong_now()
    date_part = date_for_weekday_cn(day_cn, today)
    if date_part is None:
        return None
    m = 0 if minute is None or str(minute).strip() in ("", "null", "None") else int(minute)
    h, m = normalize_half_hour(int(hour), m)
    return date_part.replace(hour=h, minute=m, second=0, microsecond=0)


def is_past_intent_datetime(day_cn: str, hour, minute, today=None) -> bool:
    """意圖所指日期時間是否已過去（相對香港時間）。"""
    today = today or hong_kong_now()
    when = intent_query_datetime(day_cn, hour, minute, today)
    if when is None:
        return False
    return when < today.replace(second=0, microsecond=0)


def format_time_ampm(hour: int, minute: int = 0) -> str:
    """例如：9:30am、4:30pm"""
    hour12 = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    return f"{hour12}:{minute:02d}{suffix}"


def format_intent_date(day_cn, today=None) -> str:
    """例如：2026年10月24日（六）"""
    today = today or hong_kong_now()
    date_part = date_for_weekday_cn(day_cn, today) if day_cn else None
    if date_part is None:
        return "未指定"
    return (
        f"{date_part.year}年{date_part.month}月{date_part.day}日"
        f"（{WEEKDAY_SHORT[date_part.weekday()]}）"
    )


def format_intent_time(hour, minute) -> str:
    """例如：9:30am；未指定則回傳「未指定」。"""
    if hour is None or str(hour).strip() in ("", "null", "None"):
        return "未指定"
    m = 0 if minute is None or str(minute).strip() in ("", "null", "None") else int(minute)
    return format_time_ampm(int(hour), m)


def looks_like_parking_query(query: str) -> bool:
    """關鍵字快速判斷是否明顯與泊車／停車場有關。"""
    q = (query or "").replace(" ", "").lower()
    keywords = (
        "泊",
        "停車",
        "泊車",
        "停車場",
        "車位",
        "空位",
        "parking",
        "carpark",
        "garage",
    )
    return any(k in q for k in keywords)


def looks_like_vacancy_prediction_query(query: str) -> bool:
    """是否偏向空位／較易泊預測（而非純資料查詢）。"""
    q = (query or "").replace(" ", "").lower()
    hints = (
        "空位",
        "易泊",
        "較易泊",
        "邊度泊",
        "邊度較易",
        "預測",
        "有冇位",
        "有冇車位",
        "泊唔泊",
        "泊到唔到",
        "車位夠",
        "夠唔夠位",
    )
    return any(h in q for h in hints)


def looks_like_fee_query(query: str) -> bool:
    """是否在問停車場收費／價錢（而非空位預測）。

    「收費／價錢」等 soft 關鍵字視為已含「停車場」語意
    （例如「天水圍聽日嘅收費如何」＝「天水圍停車場聽日嘅收費如何」）。
    """
    q = (query or "").replace(" ", "").lower()
    strong = (
        "泊車費",
        "停車費",
        "停車場收費",
        "停車場幾錢",
        "停車場價錢",
        "時租",
        "日泊",
        "夜泊",
    )
    if any(s in q for s in strong):
        return True
    soft = ("收費", "價錢", "價格", "幾錢", "多少錢", "幾多錢")
    return any(s in q for s in soft)


def looks_like_parking_info_query(query: str) -> bool:
    """與停車場有關、但似在問數量／名單／有冇等資料，而非空位預測。"""
    if looks_like_vacancy_prediction_query(query):
        return False
    if looks_like_fee_query(query):
        return True
    q = (query or "").replace(" ", "")
    system_faq = (
        r"呢個系統",
        r"這個系統",
        r"點用呢個",
        r"怎麼用",
        r"可以問咩",
        r"可以問什麼",
        r"問啲咩",
    )
    if any(re.search(p, q) for p in system_faq):
        return True
    if not looks_like_parking_query(query):
        return False
    patterns = (
        r"幾個停車場",
        r"幾多個?停車場",
        r"有多少個?停車場",
        r"有幾多個?停車場",
        r"係咪得",
        r"得一個",
        r"得唔得一個",
        r"只有一個",
        r"有冇停車場",
        r"有沒有停車場",
        r"有哪些停車場",
        r"有邊啲停車場",
        r"邊啲停車場",
        r"列出.*停車場",
        r"停車場.*名單",
        r"停車場叫咩",
        r"停車場叫什麼",
        r"邊個停車場.*叫",
    )
    return any(re.search(p, q) for p in patterns)


def intent_is_parking_related(intent: dict) -> bool:
    """從意圖 JSON 讀取 related 欄位。"""
    related = intent.get("related", True)
    if isinstance(related, bool):
        return related
    if related is None:
        return True
    return str(related).strip().lower() in ("true", "1", "yes")


def intent_wants_prediction(intent: dict) -> bool:
    """是否應走空位預測流程（預設是，以保持舊行為）。"""
    if not intent_is_parking_related(intent):
        return False
    qtype = str(intent.get("query_type") or "prediction").strip().lower()
    return qtype not in ("info", "parking_info", "faq", "data")


def parse_query_with_llm(query: str):
    """用 DeepSeek 理解查詢意圖"""
    today = hong_kong_now()
    today_day = WEEKDAY_CN[today.weekday()]
    today_str = today.strftime("%Y-%m-%d")
    # 預先算好相對日期，寫進 prompt 減少 LLM 算錯
    day_today = today_day
    day_tomorrow = WEEKDAY_CN[(today + timedelta(days=1)).weekday()]
    day_after = WEEKDAY_CN[(today + timedelta(days=2)).weekday()]

    prompt = f"""
你是一個香港停車場查詢助手。今天是 {today_str}（{today_day}）。
請從以下句子提取資訊，並以 JSON 格式回傳。
只回傳 JSON，不要其他文字。

句子：{query}

請提取：
- related: 句子是否與香港停車場／泊車有關（true 或 false）。
  - true：空位預測、較易泊、停車場數量／名單、某區有冇停車場、本系統用法等（只要與停車場／泊車有關）
  - false：天氣、食肆、純閒聊、新聞、或其他與泊車完全無關的問題
- query_type: 查詢類型（只在 related 為 true 時有意義）
  - "prediction"：想知邊度較易泊、某時某地空位、預測／推薦停車場等（即使沒寫「預測」二字）
  - "info"：與停車場有關，但不是在問空位預測。例如：某區有幾個停車場、係咪得一個、有冇停車場、叫咩名、收費／價錢、本系統可以問咩
- district: 地點名稱，會交 Google Maps 定位。優先填用戶提到的具體地名（屋邨、商場、街道、地區，例如「彩雲邨」「栢慧豪園」「天水圍」）；沒有提到就填「全港」。related 為 false 時填「全港」。不要改寫成內部代號或加「／」合併名
- hour: 小時（0-23）。只在句子有具體鐘點時填數字（如「三點」「15:30」「晚上九點」）；沒有具體鐘點則填 null（「後晚／聽晚／今晚／嚟緊」單獨出現不算具體鐘點）。問收費但無具體鐘點時填 null（系統會用當刻時間）
- minute: 分鐘，時間以半小時為單位，只允許 0 或 30。規則：
  - 「三點半／3:30／15:30」→ 30
  - 「三點／3:00／15:00」或只提到整點 → 0
  - 其他分鐘請四捨五入到最近的半小時（例如 15→30、45→下一小時的 0）
  - 如果沒有具體鐘點，填 null
  - 「嚟緊12點／嚟緊十二點」若未寫中午／午夜：hour 先填 0（稍後會規則校正為下一次 12 點）
- day: 星期（「週一」到「週日」）。必須依相對日期換算，今天是{today_day}：
  - 「今日／今晚／今天」→ {day_today}
  - 「聽日／聽晚／明天／明晚」→ {day_tomorrow}
  - 「後日／後晚／後天」→ {day_after}
  - 「嚟緊」單獨或「嚟緊＋鐘點」：日期先填「{day_today}」（稍後會規則校正）
  - 如果句子沒有提到日期或星期，就填「{day_today}」
- top_n: 要顯示幾個較易泊車的停車場（整數）。規則：
  - 「邊個／哪個／邊一個／哪一個」→ 1
  - 「邊兩個／哪兩個」→ 2、「邊三個／哪三個」→ 3，如此類推
  - 如果沒有提到要幾個，填 null
  - query_type 為 "info" 時填 null

範例（後晚無具體鐘點 → hour/minute 為 null；日期＝{day_after}）：
{{"related": true, "query_type": "prediction", "district": "全港", "hour": null, "minute": null, "day": "{day_after}", "top_n": 4}}

範例（地點預測）：
{{"related": true, "query_type": "prediction", "district": "天水圍", "hour": 20, "minute": 0, "day": "{day_today}", "top_n": null}}

範例（屋邨／地標）：
{{"related": true, "query_type": "prediction", "district": "彩雲邨", "hour": 16, "minute": 30, "day": "{day_tomorrow}", "top_n": null}}

範例（停車場資料／非預測）：
{{"related": true, "query_type": "info", "district": "掃管笏", "hour": null, "minute": null, "day": "{day_today}", "top_n": null}}

範例（問收費、無具體鐘點 → 當刻收費）：
{{"related": true, "query_type": "info", "district": "天水圍", "hour": null, "minute": null, "day": "{day_today}", "top_n": null}}

範例（問某日某時收費）：
{{"related": true, "query_type": "info", "district": "天水圍", "hour": 20, "minute": 0, "day": "{day_tomorrow}", "top_n": null}}

範例（與停車場無關）：
{{"related": false, "query_type": "prediction", "district": "全港", "hour": null, "minute": null, "day": "{day_today}", "top_n": null}}
"""
    text = call_deepseek(prompt)
    try:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        intent = json.loads(text)
    except Exception as e:
        st.error(f"解析失敗：{e}")
        intent = {
            "related": True,
            "query_type": "prediction",
            "district": "全港",
            "hour": None,
            "minute": None,
            "day": today_day,
            "top_n": None,
        }

    # 明顯含泊車關鍵字／系統 FAQ → 視為相關（避免 LLM 誤判）
    if looks_like_parking_query(query) or looks_like_parking_info_query(query):
        intent["related"] = True

    # 規則覆寫 query_type（資料查詢 vs 空位預測）
    raw_qtype = str(intent.get("query_type") or "").strip().lower()
    if raw_qtype in ("info", "parking_info", "faq", "data"):
        intent["query_type"] = "info"
    elif raw_qtype in ("prediction", "vacancy", "predict"):
        intent["query_type"] = "prediction"
    else:
        intent["query_type"] = "prediction"
    if looks_like_parking_info_query(query):
        intent["query_type"] = "info"
    elif looks_like_vacancy_prediction_query(query):
        intent["query_type"] = "prediction"

    # 規則覆寫相對日期，避免 LLM 算錯（例如後晚應為後日）
    resolved_day = resolve_relative_day(query, today)
    if resolved_day:
        intent["day"] = resolved_day

    # 保留 LLM 抽出的地點原文予 Google Geocoding；只把「全部」等正規成全港
    if _is_all_hong_kong(intent.get("district")):
        intent["district"] = "全港"

    if is_now_query(query):
        intent["day"] = WEEKDAY_CN[today.weekday()]

    # 與停車場無關 → 不推時間／不進入預測
    if not intent_is_parking_related(intent):
        intent["related"] = False
        return intent

    intent["related"] = True

    # 資料／FAQ 類：不進入預測；收費問句可保留用戶指定鐘點
    if not intent_wants_prediction(intent):
        intent["query_type"] = "info"
        intent["top_n"] = None
        if looks_like_fee_query(query) and has_explicit_clock_time(query):
            if not intent_has_clock_time(intent.get("hour")):
                intent["hour"] = None
                intent["minute"] = None
            elif intent.get("minute") is None or str(intent.get("minute")).strip() in (
                "",
                "null",
                "None",
            ):
                intent["minute"] = 0
        else:
            intent["hour"] = None
            intent["minute"] = None
        return intent

    intent["query_type"] = "prediction"
    day_cn = intent.get("day") or today_day
    future = is_future_day(day_cn, today)
    has_clock = has_explicit_clock_time(query)
    coming_soon = is_coming_soon_query(query)

    # 「嚟緊」＋具體鐘點 → 下一次即將到來的該鐘點（可跨日；曖昧 12 點取最近未來）
    if coming_soon and has_clock:
        intent = resolve_coming_soon_intent(intent, query, today)
    # 明天或之後：未提供具體鐘點則保留 null（稍後提示用戶重新輸入完整查詢）
    elif future and not has_clock:
        intent["hour"] = None
        intent["minute"] = None
    # 「而家／現在／嚟緊」或今日未指明鐘點 → 下一個半小時（可跨日）
    elif is_now_query(query) or coming_soon or not has_clock:
        dt = next_half_hour_datetime(today)
        intent["hour"] = dt.hour
        intent["minute"] = dt.minute
        intent["day"] = WEEKDAY_CN[dt.weekday()]

    return intent


def _intent_display(value, fallback="未指定"):
    if value is None or str(value).strip() in ("", "null", "None"):
        return fallback
    return str(value)


def render_intent_summary(intent: dict, *, info_mode: bool = False):
    """顯示 AI 理解的意圖（非 JSON）。"""
    # 優先顯示 Google 定位結果；否則用地點原文
    if _is_all_hong_kong(intent.get("district")):
        place_label = "全港"
    else:
        place_label = (
            intent.get("geo_label")
            or intent.get("district")
            or "未指定"
        )
    if info_mode or not intent_wants_prediction(intent):
        intent_fields = [
            ("地區", _intent_display(place_label)),
            ("查詢類型", "停車場資料／說明"),
        ]
        fee_slot = intent.get("_fee_slot_label")
        if fee_slot:
            intent_fields.append(("收費時段", str(fee_slot)))
    else:
        intent_fields = [
            ("地區", _intent_display(place_label)),
            ("日期", format_intent_date(intent.get("day"))),
            ("時間", format_intent_time(intent.get("hour"), intent.get("minute"))),
            ("顯示數量", _intent_display(intent.get("top_n"))),
        ]
    intent_cells = "".join(
        f'<div style="flex:1 1 0;min-width:7.5rem;">'
        f'<div style="font-size:0.8rem;opacity:0.65;margin-bottom:0.2rem;">'
        f"{html.escape(label)}</div>"
        f'<div style="font-size:{font_size_for_content(value)};'
        f'font-weight:600;line-height:1.35;'
        f'word-break:break-word;">{html.escape(value)}</div></div>'
        for label, value in intent_fields
    )
    st.markdown("**AI 理解的意圖**")
    st.markdown(
        f'<div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:stretch;'
        f'margin:0.15rem 0 0.9rem 0;padding:0.75rem 0.9rem;'
        f'border:1px solid rgba(128,128,128,0.28);border-radius:0.6rem;">'
        f"{intent_cells}</div>",
        unsafe_allow_html=True,
    )


def unique_carparks_for_location(
    district: str,
    *,
    geo_lat: float | None = None,
    geo_lon: float | None = None,
    geo_label: str | None = None,
) -> tuple[str, str, pd.DataFrame]:
    """
    取出本系統有紀錄的停車場（去重）。
    非全港：一律以 Google 座標＋距離半徑篩選；全港則回傳全部。
    回傳 (match_type, district_label, DataFrame)。
    """
    if _is_all_hong_kong(district):
        parks = (
            df[["park_id", "name", "district", "sub_district", "latitude", "longitude"]]
            .drop_duplicates(subset=["park_id"])
            .copy()
        )
        filtered = dedupe_equivalent_carparks(parks)
        rows = pd.DataFrame(
            {
                "park_id": filtered["park_id"].astype(str).values,
                "名稱": filtered["name"].values,
                "地區": filtered["sub_district"].fillna(filtered["district"]).values,
                "latitude": filtered["latitude"].values,
                "longitude": filtered["longitude"].values,
            }
        ).sort_values(["地區", "名稱"], kind="mergesort").reset_index(drop=True)
        return "all", "全港", rows

    district_label = geo_label or district or "該地點"
    if geo_lat is None or geo_lon is None:
        return "nearby", district_label, pd.DataFrame(
            columns=[
                "park_id",
                "名稱",
                "地區",
                "距離(公里)",
                "latitude",
                "longitude",
            ]
        )

    nearby, _radius = carparks_near_coords(float(geo_lat), float(geo_lon))
    rows = pd.DataFrame(
        {
            "park_id": nearby["park_id"].astype(str).values,
            "名稱": nearby["name"].values,
            "地區": nearby["sub_district"].fillna(nearby["district"]).values,
            "距離(公里)": nearby["distance_km"].round(1).values,
            "latitude": nearby["latitude"].values,
            "longitude": nearby["longitude"].values,
        }
    )
    return "nearby", district_label, rows


def resolve_fee_query_slot(
    intent: dict, query: str, now=None
) -> tuple[str, int, int, bool]:
    """收費查詢用時段。回傳 (day, hour, minute, is_current)。

    - 有具體鐘點 → 該日該時
    - 否則 → 現在（非今日日子缺時間應先被 needs_fee_missing_time_reentry 攔截）
    """
    now = now or hong_kong_now()
    explicit_day = resolve_relative_day(query, now)

    if has_explicit_clock_time(query) and intent_has_clock_time(intent.get("hour")):
        day = explicit_day or intent.get("day") or WEEKDAY_CN[now.weekday()]
        minute_raw = intent.get("minute")
        minute = (
            0
            if minute_raw is None or str(minute_raw).strip() in ("", "null", "None")
            else int(minute_raw)
        )
        hour, minute = normalize_half_hour(int(intent["hour"]), minute)
        return str(day), hour, minute, False

    return WEEKDAY_CN[now.weekday()], int(now.hour), int(now.minute), True


def needs_fee_missing_time_reentry(intent: dict, query_text: str) -> bool:
    """收費查詢：非今天（聽日／某一日）且未提供時間 → 要求重新輸入。"""
    if not looks_like_fee_query(query_text):
        return False
    if is_now_query(query_text):
        return False
    if has_explicit_clock_time(query_text) and intent_has_clock_time(intent.get("hour")):
        return False
    explicit_day = resolve_relative_day(query_text)
    if not explicit_day:
        return False
    today_day = WEEKDAY_CN[hong_kong_now().weekday()]
    return explicit_day != today_day


def attach_fees_to_parks_df(
    parks_df: pd.DataFrame,
    day_cn: str,
    hour: int,
    minute: int,
    fee_map: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """為停車場表加上「收費」欄（需有 park_id）。"""
    out = parks_df.copy()
    if out.empty or "park_id" not in out.columns:
        if "收費" not in out.columns:
            out["收費"] = pd.Series(dtype=str)
        return out
    fee_map = fee_map if fee_map is not None else load_carpark_fee_rules()
    out["收費"] = [
        format_carpark_fee(str(pid), day_cn, hour, minute, fee_map)
        for pid in out["park_id"]
    ]
    return out


def parks_df_for_display(parks_df: pd.DataFrame) -> pd.DataFrame:
    """結果表顯示欄（隱藏 park_id）。"""
    preferred = ["名稱", "地區", "距離(公里)", "收費"]
    cols = [c for c in preferred if c in parks_df.columns]
    cols += [c for c in parks_df.columns if c not in cols and c != "park_id"]
    return parks_df[cols] if cols else parks_df.drop(columns=["park_id"], errors="ignore")


def _fallback_parking_info_answer(
    query: str,
    district_label: str,
    parks_df: pd.DataFrame,
    *,
    fee_slot_label: str | None = None,
) -> tuple[str, str]:
    """LLM 失敗時的規則回覆。"""
    n = len(parks_df)
    names = parks_df["名稱"].tolist() if "名稱" in parks_df.columns else []
    has_fees = fee_slot_label and "收費" in parks_df.columns and n > 0

    if n == 0:
        written = (
            f"本系統在「{district_label}」附近目前沒有停車場紀錄。"
            f"你可以改試其他地點，或直接問空位預測"
            f"（例如：天水圍今晚八點邊度較易泊？）。"
        )
    elif has_fees:
        samples = []
        for _, row in parks_df.head(5).iterrows():
            fee = str(row.get("收費") or "未提供")
            samples.append(f"「{row['名稱']}」{fee}")
        sample_txt = "；".join(samples)
        more = f" 等共 {n} 個" if n > 5 else f"，共 {n} 個"
        written = (
            f"「{district_label}」附近停車場{fee_slot_label}收費如下"
            f"（{sample_txt}{more}）。"
            f"詳細見下表；未提供者為政府開放資料暫無該時段規則。"
        )
    elif n == 1:
        written = (
            f"係，以本系統資料，「{district_label}」目前只錄得 1 個停車場："
            f"「{names[0]}」。"
            f"若想知該處某時段空位，可以問我例如："
            f"「{district_label}今晚八點邊度較易泊？」。"
        )
    elif district_label == "全港":
        written = (
            f"本系統全港共有 {n} 個有空位紀錄的停車場。"
            f"範圍太大，建議你指定地區再問"
            f"（例如：掃管笏有幾個停車場？），"
            f"或直接問空位預測（例如：觀塘今晚八點邊度較易泊？）。"
        )
    else:
        preview = "、".join(names[:8])
        more = f" 等共 {n} 個" if n > 8 else f"，共 {n} 個"
        written = (
            f"以本系統資料，「{district_label}」目前有 {n} 個停車場"
            f"（{preview}{more}）。"
            f"若想知哪裡較易泊，可以告訴我時間"
            f"（例如：{district_label}今晚八點邊度較易泊？）。"
        )
    return written, written


def answer_parking_info_with_llm(
    query: str,
    district_label: str,
    parks_df: pd.DataFrame,
    *,
    fee_slot_label: str | None = None,
) -> tuple[str, str]:
    """根據本系統停車場名單，用 LLM 認真回答非預測問題。"""
    n = len(parks_df)
    has_fees = bool(fee_slot_label) and "收費" in parks_df.columns
    if district_label == "全港" and n > 30:
        name_block = f"（全港共 {n} 個，名單過長不逐一列出）"
    else:
        lines = []
        for i, (_, row) in enumerate(parks_df.head(40).iterrows(), start=1):
            line = f"{i}. {row['名稱']}（{row['地區']}）"
            if has_fees:
                line += f"｜收費：{row.get('收費') or '未提供'}"
            lines.append(line)
        listed = "\n".join(lines)
        if n > 40:
            listed += f"\n…另有 {n - 40} 個未列出"
        name_block = listed or "（無）"

    fee_fact = ""
    fee_guidance = ""
    if has_fees:
        fee_fact = f"- 收費時段：{fee_slot_label}\n- 收費來源：政府停車場開放資料（私家車時租／日夜泊；無資料則顯示未提供）\n"
        fee_guidance = (
            "用戶在問收費：請依事實說明該時段收費概況，可舉數個例子，"
            "並提醒下表有完整名單；不要捏造價錢。\n"
        )

    prompt = f"""
你是「咪再兜」香港停車場空位預測系統的助手。用戶的問題與停車場有關，但不是在問空位預測。
請根據「本系統資料庫事實」認真回答，用繁體中文、簡潔清楚（可用粵語口語）。
只根據事實作答，不要捏造停車場或收費；若資料不足就誠實說明。
{fee_guidance}回答後可自然地提一句：若要查某時段較易泊／空位，可以再問預測問題。

用戶問題：{query}

事實：
- 查詢地區：{district_label}
- 該範圍停車場數量：{n}
{fee_fact}- 停車場名單：
{name_block}
- 本系統主要功能：預測香港各區停車場在指定時段的空位，並推薦較易泊的停車場

只回傳 JSON，不要其他文字：
{{"answer": "給用戶看的書面回覆", "spoken": "給語音朗讀的粵語口語版本，一句至數句即可"}}
"""
    text = call_deepseek(prompt)
    if not text:
        return _fallback_parking_info_answer(
            query, district_label, parks_df, fee_slot_label=fee_slot_label
        )
    try:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        payload = json.loads(text)
        answer = str(payload.get("answer") or "").strip()
        spoken = str(payload.get("spoken") or answer).strip()
        if not answer:
            return _fallback_parking_info_answer(
                query, district_label, parks_df, fee_slot_label=fee_slot_label
            )
        return answer, spoken or answer
    except Exception:
        return _fallback_parking_info_answer(
            query, district_label, parks_df, fee_slot_label=fee_slot_label
        )


def render_saved_info_answer(payload: dict, *, auto_speak: bool = False) -> None:
    """重顯示已儲存的停車場資料／說明回覆。"""
    intent = dict(payload.get("intent") or {})
    fee_slot_label = payload.get("fee_slot_label")
    # 舊 session payload 無 wants_fee：有 fee_slot_label 即當明確問收費
    wants_fee = (
        bool(payload["wants_fee"])
        if "wants_fee" in payload
        else bool(fee_slot_label)
    )
    if fee_slot_label:
        intent["_fee_slot_label"] = fee_slot_label
    render_intent_summary(intent, info_mode=True)
    district_label = payload.get("district_label") or district_display_name(
        intent.get("district")
    )
    count = int(payload.get("count") or 0)
    nearby_suffix = "附近" if intent.get("geo_lat") is not None else ""
    if wants_fee and fee_slot_label:
        st.success(
            f"「{district_label}」{nearby_suffix}停車場收費"
            f"（{fee_slot_label}，共 {count} 個）"
        )
    else:
        st.success(
            f"「{district_label}」{nearby_suffix}本系統停車場資料（共 {count} 個）"
        )
    # 朗讀掣：success 之下、表格之上、靠右（同預測成功一致）；無稿都畀司機聽到一句
    written = (payload.get("written") or "").strip()
    spoken = (payload.get("spoken") or written).strip()
    if not written and not spoken:
        written = "暫時沒有可顯示的回覆。"
        spoken = "暫時冇可顯示嘅回覆。"
    need_rerun = render_text_with_speech(
        written,
        spoken,
        auto_speak=auto_speak,
        button_key="info_speech_toggle",
        show_written=SHOW_WRITTEN_SUGGESTION,
    )
    parks_df = parks_df_for_display(pd.DataFrame(payload.get("park_records") or []))
    if not parks_df.empty and count <= 40:
        # 保留 latitude／longitude 畀地圖；表格顯示時隱藏
        display_df = parks_df.drop(columns=list(_TABLE_HIDDEN_COLS), errors="ignore")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        render_results_map(parks_df, intent=intent)
        if fee_slot_label and "收費" in parks_df.columns:
            st.caption(
                f"收費為「{fee_slot_label}」時段（政府開放資料私家車規則；無資料則顯示未提供）。"
            )
    elif count > 40:
        st.caption(f"該範圍共有 {count} 個停車場，名單較長，以上方文字說明為準。")
    flush_deferred_speech_browser_ops()
    if need_rerun:
        st.rerun()


def run_parking_info_answer(intent: dict, query_text: str) -> None:
    """處理與停車場有關、但非空位預測的查詢。"""
    district = intent.get("district", "全港")
    geo_lat = intent.get("geo_lat")
    geo_lon = intent.get("geo_lon")
    geo_label = intent.get("geo_label")
    match_type, district_label, parks_df = unique_carparks_for_location(
        district,
        geo_lat=geo_lat,
        geo_lon=geo_lon,
        geo_label=geo_label,
    )
    if match_type == "nearby":
        district_label = geo_label or district or district_label

    # 結果表一律附上收費（預設用當刻／已指定時段）；成功橫幅／LLM 聚焦收費只喺明確問價時開啟
    wants_fee = looks_like_fee_query(query_text)
    fee_slot_label = None
    if not parks_df.empty:
        day_cn, hour, minute, is_current = resolve_fee_query_slot(intent, query_text)
        parks_df = attach_fees_to_parks_df(parks_df, day_cn, hour, minute)
        if is_current:
            fee_slot_label = f"現在 {format_time_ampm(hour, minute)}"
        else:
            fee_slot_label = f"{day_cn} {format_time_12h(hour, minute)}"

    if match_type == "nearby" and parks_df.empty:
        intent_for_summary = dict(intent)
        if wants_fee and fee_slot_label:
            intent_for_summary["_fee_slot_label"] = fee_slot_label
        render_feedback_with_speech(
            f"「{district_label}」附近本系統沒有停車場紀錄。"
            "請改試其他地點後重新輸入。",
            spoken=(
                f"「{district_label}」附近本系統冇停車場紀錄。"
                "請改試其他地點之後重新輸入。"
            ),
            intent=intent_for_summary,
            info_mode=True,
            auto_speak=True,
        )
        return

    with st.spinner("正在根據本系統資料整理回覆..."):
        written, spoken = answer_parking_info_with_llm(
            query_text,
            district_label,
            parks_df,
            fee_slot_label=fee_slot_label if wants_fee else None,
        )

    payload = {
        "intent": intent,
        "district_label": district_label,
        "count": int(len(parks_df)),
        "park_records": parks_df_for_display(parks_df).to_dict(orient="records"),
        "written": written,
        "spoken": spoken,
        "fee_slot_label": fee_slot_label,
        "wants_fee": wants_fee,
    }
    st.session_state.last_info_answer = payload
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_feedback", None)
    render_saved_info_answer(payload, auto_speak=True)


def resolve_top_n(query_text: str, intent: dict) -> int:
    top_n = parse_top_n_from_query(query_text)
    if top_n is None:
        raw_top_n = intent.get("top_n")
        if raw_top_n is not None and str(raw_top_n).strip() not in ("", "null", "None"):
            try:
                top_n = int(raw_top_n)
            except (TypeError, ValueError):
                top_n = DEFAULT_TOP_N
        else:
            top_n = DEFAULT_TOP_N
    if top_n < 1:
        top_n = DEFAULT_TOP_N
    return top_n


def run_vacancy_prediction(intent: dict, query_text: str):
    """依意圖執行預測並顯示結果。"""
    district = intent.get("district", "全港")
    geo_lat = intent.get("geo_lat")
    geo_lon = intent.get("geo_lon")
    geo_label = intent.get("geo_label")
    asking_now = is_now_query(query_text)
    now = hong_kong_now()
    day = intent.get("day", WEEKDAY_CN[now.weekday()])

    raw_hour = intent.get("hour")
    raw_minute = intent.get("minute")
    if asking_now or not intent_has_clock_time(raw_hour):
        dt = next_half_hour_datetime(now)
        hour, minute = dt.hour, dt.minute
        day = WEEKDAY_CN[dt.weekday()]
    else:
        hour = int(raw_hour)
        minute = (
            0
            if raw_minute is None or str(raw_minute).strip() in ("", "null", "None")
            else int(raw_minute)
        )
        hour, minute = normalize_half_hour(hour, minute)
    half_hour = to_half_hour_slot(hour, minute)
    top_n = resolve_top_n(query_text, intent)
    dow_num = DAY_TO_NUM.get(day, 0)
    dist_map: dict[str, float] = {}
    match_type = "all"
    district_label = "全港"

    le = LabelEncoder()
    df["park_code"] = le.fit_transform(df["park_id"])

    features = ["half_hour", "dow", "park_code"]
    X = df[features]
    y = df["vacancy"]

    model = GradientBoostingRegressor(random_state=42, n_estimators=50)
    model.fit(X, y)

    # 非全港：一律 Google 座標＋半徑篩附近車場（不再用 sub_district）
    if _is_all_hong_kong(district):
        df_filtered = df
    elif geo_lat is not None and geo_lon is not None:
        nearby, _radius = carparks_near_coords(float(geo_lat), float(geo_lon))
        district_label = geo_label or district or "該地點"
        if nearby.empty:
            render_feedback_with_speech(
                f"「{district_label}」附近本系統沒有停車場紀錄。"
                "請改試其他地點後重新輸入。",
                spoken=(
                    f"「{district_label}」附近本系統冇停車場紀錄。"
                    "請改試其他地點之後重新輸入。"
                ),
                intent=intent,
                auto_speak=True,
            )
            return
        dist_map = {
            str(pid): float(dist)
            for pid, dist in zip(nearby["park_id"], nearby["distance_km"])
        }
        df_filtered = df[df["park_id"].astype(str).isin(list(dist_map.keys()))]
        if df_filtered.empty:
            render_feedback_with_speech(
                f"「{district_label}」附近本系統沒有可用的空位預測資料。"
                "請改試其他地點後重新輸入。",
                spoken=(
                    f"「{district_label}」附近本系統冇可用嘅空位預測資料。"
                    "請改試其他地點之後重新輸入。"
                ),
                intent=intent,
                auto_speak=True,
            )
            return
        match_type = "nearby"
    else:
        place = district_display_name(district)
        render_feedback_with_speech(
            f"未能定位「{place}」。請重新輸入更清楚的地名。",
            spoken=f"未能定位「{place}」。請重新輸入更清楚嘅地名。",
            intent=intent,
            auto_speak=True,
        )
        return

    fee_map = load_carpark_fee_rules()
    results = []
    for park_id in df_filtered["park_id"].unique():
        park_code = le.transform([park_id])[0]
        pred = model.predict([[half_hour, dow_num, park_code]])[0]
        info = df[df["park_id"] == park_id].iloc[0]
        row = {
            "park_id": park_id,
            "name": info["name"],
            "latitude": info.get("latitude"),
            "longitude": info.get("longitude"),
            "名稱": info["name"],
            "地區": info.get("sub_district") or info["district"],
            "收費": format_carpark_fee(park_id, day, hour, minute, fee_map),
            "預測空位": max(0, round(pred)),
        }
        pid_key = str(park_id)
        if match_type == "nearby" and pid_key in dist_map:
            row["距離(公里)"] = round(dist_map[pid_key], 1)
        results.append(row)

    pred_df = pd.DataFrame(results)
    if not pred_df.empty:
        pred_df = dedupe_equivalent_carparks(
            pred_df, prefer_higher_col="預測空位"
        )
        # 保留 latitude／longitude 畀地圖用；表格顯示時會隱藏
        pred_df = pred_df.drop(columns=["park_id", "name"], errors="ignore")
    if match_type == "nearby":
        result_df = select_top_nearby_with_ties(pred_df, top_n)
        district_label = f"{district_label}附近"
    else:
        result_df = select_top_with_ties(pred_df, top_n)
    # 欄位順序：排名、名稱、地區、（距離）、收費、預測空位；（隱藏）座標
    preferred = ["", "名稱", "地區", "距離(公里)", "收費", "預測空位"]
    ordered = [c for c in preferred if c in result_df.columns]
    ordered += [c for c in result_df.columns if c not in ordered]
    result_df = result_df[ordered]

    shown = len(result_df)
    written = format_top_suggestion(result_df, hour, minute)
    spoken = format_spoken_suggestion(result_df, hour, minute)

    payload = {
        "intent": intent,
        "asking_now": asking_now,
        "day": day,
        "hour": hour,
        "minute": minute,
        "district_label": district_label,
        "top_n": top_n,
        "shown": shown,
        "result_records": result_df.to_dict(orient="records"),
        "written": written,
        "spoken": spoken,
    }
    st.session_state.last_prediction = payload
    st.session_state.pop("last_info_answer", None)
    st.session_state.pop("last_feedback", None)
    render_saved_prediction(payload, auto_speak=True)


def needs_missing_time_reentry(intent: dict, query_text: str) -> bool:
    """明天或之後且未提供時間 → 不執行預測，要求重新輸入完整查詢。"""
    asking_now = is_now_query(query_text)
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    if asking_now:
        return False
    return is_future_day(day) and not intent_has_clock_time(intent.get("hour"))


def needs_ambiguous_ampm_reentry(intent: dict, query_text: str) -> bool:
    """有鐘點但未標明上午／下午等時段 → 不執行預測，要求重新輸入。

    例如「星期日三點」曖昧；「下午三點」「嚟緊三點」「15點」則可接受。
    """
    if is_now_query(query_text):
        return False
    if not has_explicit_clock_time(query_text):
        return False
    if has_explicit_day_period(query_text):
        return False
    if is_coming_soon_query(query_text):
        return False
    if has_unambiguous_24h_clock(query_text):
        return False
    return True


def needs_past_time_followup(intent: dict, query_text: str) -> bool:
    """今日（或最近該日）且鐘點已過去 → 不執行預測，要求重新輸入。"""
    if is_now_query(query_text):
        return False
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    return is_past_intent_datetime(day, intent.get("hour"), intent.get("minute"))


def render_missing_time_blocked(intent: dict, *, auto_speak: bool = True):
    """未來日期但未指明時間：顯示意圖並要求重新輸入完整查詢。"""
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    day_label = format_intent_date(day)
    render_feedback_with_speech(
        f"你想查詢的是 **{day_label}**，但未提供具體時間。"
        f"請重新輸入查詢內容（例如：灣仔大後日下午三點邊度較易泊？）。",
        spoken=(
            f"你想查嘅係{day_label}，但未有提供具體時間。"
            "請重新輸入查詢內容，例如灣仔大後日下午三點邊度較易泊。"
        ),
        intent=intent,
        auto_speak=auto_speak,
    )


def render_fee_missing_time_blocked(
    intent: dict, query_text: str, *, auto_speak: bool = True
):
    """收費查詢指定非今日日子但未指明時間：要求重新輸入。"""
    day = (
        resolve_relative_day(query_text)
        or intent.get("day")
        or WEEKDAY_CN[hong_kong_now().weekday()]
    )
    day_label = format_intent_date(day)
    render_feedback_with_speech(
        f"你想查詢的是 **{day_label}** 的停車場收費，但未提供具體時間。"
        f"查詢今天以外的日子必須註明時間。"
        f"請重新輸入查詢內容（例如：彩雲邨星期日下午三點停車場收費？）。",
        spoken=(
            f"你想查嘅係{day_label}嘅停車場收費，但未有提供具體時間。"
            "查詢今天以外嘅日子必須註明時間。"
            "請重新輸入，例如彩雲邨星期日下午三點停車場收費。"
        ),
        intent=intent,
        info_mode=True,
        auto_speak=auto_speak,
    )


def render_ambiguous_ampm_blocked(intent: dict, *, auto_speak: bool = True):
    """有鐘點但未標上午／下午：顯示意圖並要求重新輸入完整查詢。"""
    # 唔顯示 LLM 臆測嘅 am/pm，避免誤導
    display_intent = {**intent, "hour": None, "minute": None}
    render_feedback_with_speech(
        "你提供了鐘點，但未標明是上午還是下午"
        "（例如：上午／下午／上晝／下晝／朝早／夜晚／凌晨／嚟緊）。"
        "請重新輸入查詢內容（例如：天晴邨附近星期日下午三點邊度較易泊？）。",
        spoken=(
            "你提供咗鐘點，但未標明係上午定下午。"
            "請重新輸入查詢內容，例如天晴邨附近星期日下午三點邊度較易泊。"
        ),
        intent=display_intent,
        auto_speak=auto_speak,
    )


def render_past_time_blocked(intent: dict, *, auto_speak: bool = True):
    """查詢時間已過：顯示意圖並要求重新輸入完整查詢。"""
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    when = (
        f"{format_intent_date(day)} "
        f"{format_intent_time(intent.get('hour'), intent.get('minute'))}"
    )
    render_feedback_with_speech(
        f"你想查詢的是 **{when}**，但該時間已過去。請重新輸入查詢內容。",
        spoken=f"你想查嘅係{when}，但該時間已經過去。請重新輸入查詢內容。",
        intent=intent,
        auto_speak=auto_speak,
    )


auto_submit = st.session_state.pop("auto_submit_query", False)
should_query = submitted or auto_submit
query_text = (user_query or "").strip()

# 新查詢：只清 session 狀態。
# 唔喺呢度注入 stop JS，否則會同稍後嘅 auto-speak 搶跑（iframe 執行次序不穩 → 查詢後無聲）。
# 舊朗讀已由「查詢／語音」掣前端 pointerdown 即時 cancel。
if submitted or should_query:
    st.session_state._suggestion_speaking = False
    st.session_state.pop("_pending_speak_text", None)

# 新查詢會清掉舊結果／舊狀態
if should_query and query_text:
    st.session_state.pop("awaiting_time", None)
    st.session_state.pop("past_time_blocked", None)
    st.session_state.pop("time_followup_ready", None)
    st.session_state.pop("_last_spoken_text", None)
    st.session_state.pop("_auto_speak_token", None)
    st.session_state._suggestion_speaking = False
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_info_answer", None)
    st.session_state.pop("last_feedback", None)
    with st.spinner("AI 正在理解你的問題..."):
        intent = parse_query_with_llm(query_text)

    if not intent_is_parking_related(intent):
        render_feedback_with_speech(
            "你的查詢似乎與停車場／泊車無關。"
            "請重新輸入與停車場空位有關的問題"
            "（例如：天水圍今晚八點邊度較易泊？）。",
            spoken=(
                "你嘅查詢似乎同停車場或者泊車無關。"
                "請重新輸入同停車場空位有關嘅問題，"
                "例如天水圍今晚八點邊度較易泊。"
            ),
            auto_speak=True,
        )
    else:
        geo_error = None
        if not _is_all_hong_kong(intent.get("district")):
            with st.spinner("正在用 Google Maps 搜尋地點..."):
                geo_error = attach_geocode_to_intent(intent)
        if geo_error:
            render_feedback_with_speech(
                geo_error,
                spoken=_plain_for_speech(geo_error),
                intent=intent,
                info_mode=not intent_wants_prediction(intent),
                auto_speak=True,
            )
        elif not intent_wants_prediction(intent):
            if needs_fee_missing_time_reentry(intent, query_text):
                render_fee_missing_time_blocked(intent, query_text)
            else:
                run_parking_info_answer(intent, query_text)
        elif needs_missing_time_reentry(intent, query_text):
            render_missing_time_blocked(intent)
        elif needs_ambiguous_ampm_reentry(intent, query_text):
            render_ambiguous_ampm_blocked(intent)
        elif needs_past_time_followup(intent, query_text):
            render_past_time_blocked(intent)
        else:
            run_vacancy_prediction(intent, query_text)
elif should_query:
    render_feedback_with_speech(
        "請先輸入查詢內容。",
        spoken="請先輸入查詢內容。",
        auto_speak=True,
    )

# 保留上次結果（例如撳「再聽一次」後仍可看到建議）
elif st.session_state.get("last_feedback") and not should_query:
    render_saved_feedback(st.session_state.last_feedback, auto_speak=False)
elif st.session_state.get("last_info_answer") and not should_query:
    render_saved_info_answer(st.session_state.last_info_answer, auto_speak=False)
elif st.session_state.get("last_prediction") and not should_query:
    render_saved_prediction(st.session_state.last_prediction, auto_speak=False)
