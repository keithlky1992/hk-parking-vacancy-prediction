import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
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
from parking_db import (
    baselines_frame_to_maps,
    get_secret,
    load_baselines_dataframe,
    load_carparks_frame,
    load_vacancy_frame,
)
from parking_result_utils import (
    build_equivalent_park_id_map,
    build_fee_park_id_aliases,
    coerce_park_id,
    fee_availability_mode,
    fee_coverage_counts,
    fee_value_missing,
    filter_parks_with_fee,
    filter_parks_without_fee,
    first_available_fee,
    format_top_n_shortfall_notice,
    looks_like_fee_availability_query,
    looks_like_hard_to_park_ranking_query,
    merge_baseline_packs,
    normalize_asr_query_text,
    order_parks_for_fee_display,
    park_name_matches_place_tokens,
    parks_present_in_baselines,
    pin_venue_parks_into_ranking,
    is_beyond_prediction_horizon,
    MAX_PREDICTION_HORIZON_DAYS,
    parse_past_relative_calendar_date,
    parse_top_n_from_query,
    query_mentions_invalid_calendar_date,
    query_mentions_invalid_clock_time,
    query_specifies_non_half_hour_time,
    relative_day_offset_from_query,
    resolve_yearless_calendar_date,
    RELATIVE_DAY_STRIP_RE,
    top_n_exceeds_available,
)

WEEKDAY_CN = {0: "週一", 1: "週二", 2: "週三", 3: "週四", 4: "週五", 5: "週六", 6: "週日"}
WEEKDAY_SHORT = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}
DAY_TO_NUM = {"週一": 0, "週二": 1, "週三": 2, "週四": 3, "週五": 4, "週六": 5, "週日": 6}

NUM_CN = {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
DEFAULT_TOP_N = 10
# 未指定顯示數量時，朗讀只讀頭幾名（畫面仍可顯示 DEFAULT_TOP_N）
SPEAK_DEFAULT_TOP_N = 3
# 預測 shrinkage：同星期樣本少時向平日／週末均值收斂，仍保留星期差異
BASELINE_PRIOR_K = 3.0
SEARCH_HISTORY_MAX = 8
LIVE_NEARBY_TOP_N = 8
# 暫時隱藏「預測建議」書面文字；朗讀掣／自動播仍然保留。改 True 即可恢復顯示
SHOW_WRITTEN_SUGGESTION = False
# 停車場資料／說明類：書面答覆唔顯示（保留朗讀掣／自動播；表格／地圖仍會出）
SHOW_WRITTEN_INFO_ANSWER = False
# 資料答覆：表格顯示完整名單；地圖標記過多會卡，只畫前 N 個
INFO_MAP_MARKER_MAX = 40
# LLM／fallback 名單抽樣上限（同地圖預覽對齊）
INFO_TABLE_PREVIEW_MAX = INFO_MAP_MARKER_MAX
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
# 「某地容唔容易泊」：視為該場所車場嘅距離上限（公里）
# 適用所有同類問法（商場／邨／地標），唔限單一例子。
PLACE_AT_VENUE_KM = 0.35
# 極近且名稱實質相同 → 視為同一停車場（運輸署常有兩套 park_id）
EQUIVALENT_CARPARK_MAX_KM = 0.06
# 查詢地名 → 車場名稱常見別名（小階；可選加速對應，主邏輯仍靠名稱 token＋距離）
PLACE_NAME_ALIASES = {
    "yoho": ("yoho", "形點"),
    "yohomall": ("yoho", "形點"),
    "apm": ("apm",),
    "mikiki": ("mikiki",),
    "airside": ("airside",),
    "ifc": ("ifc",),
    "icc": ("icc",),
    "elements": ("elements", "圓方"),
    "landmark": ("landmark", "置地"),
    "timesquare": ("timesquare", "時代廣場"),
    "moko": ("moko", "摩企"),
    "festivalwalk": ("festivalwalk", "又一城"),
    "harbourcity": ("harbourcity", "海港城"),
    "pacificplace": ("pacificplace", "太古廣場"),
    "hysan": ("hysan", "希慎"),
    "telford": ("telford", "德福"),
    "telfordplaza": ("telford", "德福"),
    "德福": ("德福", "telford"),
    "德福廣場": ("德福", "telford"),
}
# 廣域地區名：即使用「容唔容易泊」亦較似問成區，保持附近排名（唔當單一場所）
MAJOR_AREA_PLACE_LABELS = frozenset(
    {
        "全港",
        "香港",
        "九龍",
        "新界",
        "港島",
        "香港島",
        "中西區",
        "灣仔",
        "東區",
        "南區",
        "油尖旺",
        "深水埗",
        "九龍城",
        "黃大仙",
        "觀塘",
        "荃灣",
        "屯門",
        "元朗",
        "北區",
        "大埔",
        "西貢",
        "沙田",
        "葵青",
        "離島",
        "天水圍",
        "將軍澳",
        "東涌",
        "旺角",
        "銅鑼灣",
        "尖沙咀",
        "中環",
        "金鐘",
        "佐敦",
        "紅磡",
        "旺角東",
        "九龍灣",
        "觀塘區",
        "油麻地",
        "太子",
        "深水步",
        "荃灣區",
        "屯門區",
        "元朗區",
        "沙田區",
    }
)

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


def _normalize_voice_audio(audio) -> dict | None:
    """將 st.audio_input（UploadedFile）或舊 mic_recorder dict 統一成 {bytes, id, ...}。"""
    if audio is None:
        return None
    if isinstance(audio, dict) and audio.get("bytes") is not None:
        if not audio.get("id"):
            raw = audio["bytes"]
            audio = {
                **audio,
                "id": f"dict:{len(raw)}:{hash(bytes(raw[:2048]))}",
            }
        return audio
    getval = getattr(audio, "getvalue", None)
    if not callable(getval):
        return None
    raw = getval()
    if not raw:
        return None
    name = getattr(audio, "name", None) or "voice.wav"
    size = int(getattr(audio, "size", None) or len(raw))
    file_id = getattr(audio, "file_id", None) or f"{name}:{size}:{hash(bytes(raw[:2048]))}"
    return {
        "bytes": raw,
        "id": str(file_id),
        "sample_rate": 16000,
        "sample_width": 2,
    }


def _pcm_from_audio(audio: dict) -> tuple[np.ndarray, int]:
    """從語音結果取出 mono int16 PCM 與 sample rate。"""
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
    """將語音音訊轉成文字。"""
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

    # 香港時間由前端 JS 用 data-mzd-clock + CSS ::after 顯示，唔好寫入 textContent，
    # 否則同 React reconcile 搶子節點 → NotFoundError: removeChild。
    rows = [
        ("香港時間", "", "mzd-hk-time"),
        ("你目前所在的地區", location_label, "mzd-location"),
        ("天氣", weather_label, "mzd-weather"),
    ]
    cells = []
    for label, value, value_class in rows:
        class_attr = f' class="{html.escape(value_class)}"' if value_class else ""
        # 保留初始時間於 data attr（JS 未跑前／SEO）；之後只改 attribute
        data_attr = ""
        if value_class == "mzd-hk-time":
            data_attr = f' data-mzd-clock="{html.escape(hk_time, quote=True)}"'
        cells.append(
            f'<div style="min-width:0;flex:1 1 0;">'
            f'<div style="font-size:0.85rem;line-height:1.2;opacity:0.65;margin-bottom:0.15rem;">'
            f"{html.escape(label)}</div>"
            f'<div{class_attr}{data_attr} style="font-size:1.05rem;font-weight:600;line-height:1.35;'
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
  /* 語音：st.audio_input 高度同查詢掣。
     Streamlit 真正撐高嘅係內層 emotion div（預設 height: 4.25rem），唔係外層 testid。 */
  .st-key-voice_input,
  div[data-testid="stAudioInput"] {
    width: 100% !important;
  }
  .st-key-voice_input [data-testid="stAudioInput"],
  div[data-testid="stAudioInput"] {
    height: 2.5rem !important;
  }
  .st-key-voice_input [data-testid="stAudioInput"] > div,
  div[data-testid="stAudioInput"] > div {
    height: 2.5rem !important;
    max-height: 2.5rem !important;
  }
  .st-key-voice_input button,
  div[data-testid="stAudioInput"] button {
    height: 2.5rem !important;
  }
  /* 隱藏結果區原生 autoplay <audio> 外殼，只保留出聲 */
  .mzd-native-autoplay,
  [data-testid="stElementContainer"]:has(.mzd-native-autoplay) {
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
  /* 香港時間：用 ::after 顯示，避免 JS 改 React 子節點 */
  .mzd-hk-time {
    min-height: 1.35em;
  }
  .mzd-hk-time::after {
    content: attr(data-mzd-clock);
  }
  /* Streamlit 1.51 會把 height=0 當 falsy 回落 ~150px；用 height=1 再收起。 */
  iframe[height="0"],
  iframe[height="1"] {
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
  [data-testid="stElementContainer"]:has(iframe[height="0"]),
  [data-testid="stElementContainer"]:has(iframe[height="1"]) {
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

  // iOS／語音：撳掣手勢內開始「靜音 keepalive」loop，撑過辨識＋AI 等待；
  // 結果 MP3 就緒後喺同一 Audio 元素換 src 再 play（中途 pause 會令 iOS 收回自動播權限）。
  function buildSilentWavDataUri(seconds) {
    var rate = 8000;
    var n = Math.max(1, Math.floor(rate * (seconds || 0.5)));
    var dataSize = n * 2;
    var buffer = new ArrayBuffer(44 + dataSize);
    var view = new DataView(buffer);
    function wstr(offset, s) {
      for (var i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    }
    wstr(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    wstr(8, "WAVE");
    wstr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    wstr(36, "data");
    view.setUint32(40, dataSize, true);
    var bytes = new Uint8Array(buffer);
    var binary = "";
    for (var i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return "data:audio/wav;base64," + root.btoa(binary);
  }
  // 稍長靜音段：語音辨識＋LLM 等得耐時，loop 換段較穩
  var SILENT_WAV = buildSilentWavDataUri(2.0);

  function ensureUnlockedMediaAudio() {
    if (!root.__hkParkingMediaAudio) {
      root.__hkParkingMediaAudio = new root.Audio();
      root.__hkParkingMediaAudio.preload = "auto";
      try { root.__hkParkingMediaAudio.playsInline = true; } catch (e) {}
      try { root.__hkParkingMediaAudio.setAttribute("playsinline", "true"); } catch (e) {}
      try { root.__hkParkingMediaAudio.setAttribute("webkit-playsinline", "true"); } catch (e) {}
      // iOS：掛入 DOM 較易保持解鎖
      try {
        root.__hkParkingMediaAudio.style.cssText =
          "position:fixed;width:0;height:0;opacity:0;pointer-events:none;";
      } catch (e) {}
    }
    // Streamlit 重繪後 Audio 可能被移出 DOM；掛返 body 先保住 iOS 解鎖
    try {
      var mediaEl = root.__hkParkingMediaAudio;
      var host = root.document.body || root.document.documentElement;
      if (mediaEl && host && !mediaEl.isConnected) {
        host.appendChild(mediaEl);
      } else if (mediaEl && host && !mediaEl.parentNode) {
        host.appendChild(mediaEl);
      }
    } catch (e) {}
    return root.__hkParkingMediaAudio;
  }

  function ensureAudioContext() {
    var AC = root.AudioContext || root.webkitAudioContext;
    if (!AC) return null;
    try {
      if (!root.__hkParkingAudioCtx || root.__hkParkingAudioCtx.state === "closed") {
        root.__hkParkingAudioCtx = new AC();
      }
      return root.__hkParkingAudioCtx;
    } catch (e) {
      return null;
    }
  }

  function stopAudioContextSource() {
    var srcNode = root.__hkParkingAudioCtxSource;
    root.__hkParkingAudioCtxSource = null;
    if (!srcNode) return;
    try { srcNode.onended = null; } catch (e) {}
    try { srcNode.stop(); } catch (e) {}
    try { srcNode.disconnect(); } catch (e) {}
  }

  function stopOscKeepalive() {
    var osc = root.__hkParkingOscKeepalive;
    root.__hkParkingOscKeepalive = null;
    if (!osc) return;
    try { osc.stop(); } catch (e) {}
    try { osc.disconnect(); } catch (e) {}
  }

  function startOscKeepalive() {
    try {
      var ctx = ensureAudioContext();
      if (!ctx) return;
      stopOscKeepalive();
      var gain = ctx.createGain();
      // 極低音量：保持 context running，唔令人聽到
      gain.gain.value = 0.0001;
      gain.connect(ctx.destination);
      var osc = ctx.createOscillator();
      osc.frequency.value = 20;
      osc.connect(gain);
      osc.start(0);
      root.__hkParkingOscKeepalive = osc;
      root.__hkParkingAudioCtxUnlocked = true;
    } catch (e) {}
  }

  function stopKeepaliveWatch() {
    if (!root.__hkParkingKeepaliveWatch) return;
    try { root.clearInterval(root.__hkParkingKeepaliveWatch); } catch (e) {}
    root.__hkParkingKeepaliveWatch = null;
  }

  function startKeepaliveWatch() {
    stopKeepaliveWatch();
    root.__hkParkingKeepaliveWatch = root.setInterval(function () {
      if (!root.__hkParkingMediaKeepalive) {
        stopKeepaliveWatch();
        return;
      }
      var a = root.__hkParkingMediaAudio;
      if (!a) return;
      try {
        if (a.paused || a.ended) {
          try {
            if (!a.src || a.src.indexOf("data:audio/wav") !== 0) {
              a.src = SILENT_WAV;
            }
          } catch (e2) {}
          try { a.loop = true; } catch (e2) {}
          try { a.volume = 0.01; } catch (e2) {}
          var p = a.play();
          if (p && typeof p.catch === "function") {
            p.catch(function () {
              try {
                a.src = SILENT_WAV;
                a.loop = true;
                a.play().catch(function () {});
              } catch (e3) {}
            });
          }
        }
      } catch (e) {}
      // 同步撑住 AudioContext，避免長等待後 suspended
      try {
        var ctx = root.__hkParkingAudioCtx;
        if (ctx && ctx.state === "suspended") ctx.resume();
        if (root.__hkParkingAudioCtxUnlocked && !root.__hkParkingOscKeepalive) {
          startOscKeepalive();
        }
      } catch (e) {}
    }, 400);
  }

  function markVoiceQueryArmed() {
    root.__hkParkingVoiceQueryArmed = true;
    // 涵蓋 Edge TTS 冷啟動；太短會令第一次自動播喺 MP3 就緒前就放棄
    root.__hkParkingVoiceQueryArmedUntil = Date.now() + 90000;
  }

  function clearVoiceQueryArmed() {
    root.__hkParkingVoiceQueryArmed = false;
    root.__hkParkingVoiceQueryArmedUntil = 0;
  }

  function isVoiceQueryArmed() {
    if (!root.__hkParkingVoiceQueryArmed) return false;
    var until = root.__hkParkingVoiceQueryArmedUntil || 0;
    if (until && Date.now() > until) {
      clearVoiceQueryArmed();
      return false;
    }
    return true;
  }

  function isRealSpeechAudioActive() {
    // 正在播建議 MP3／解碼中：唔好 revive 靜音 loop（否則會改 src 切斷「你想…」）
    if (root.__hkParkingMediaPlayPending) return true;
    if (root.__hkParkingAudioCtxSource) return true;
    if (root.__hkParkingMediaKeepalive) return false;
    var a = root.__hkParkingMediaAudio;
    if (!a) return false;
    try {
      return !a.paused && !a.ended;
    } catch (e) {
      return false;
    }
  }

  function reviveKeepaliveIfPossible() {
    // 語音結果頁：若曾經解鎖但 silent loop 斷咗，嘗試喺仍有 unlock 旗標時救返
    if (!(root.__hkParkingMediaUnlocked || root.__hkParkingAudioCtxUnlocked)) {
      return false;
    }
    // 關鍵：朗讀已開始就絕對唔好把 src 改返 SILENT_WAV
    if (isRealSpeechAudioActive()) return false;
    try {
      var a = ensureUnlockedMediaAudio();
      a.muted = false;
      a.volume = 0.01;
      a.loop = true;
      root.__hkParkingMediaKeepalive = true;
      a.onended = null;
      a.onerror = null;
      a.src = SILENT_WAV;
      var p = a.play();
      if (p && typeof p.then === "function") {
        p.then(function () { startKeepaliveWatch(); }).catch(function () {
          startKeepaliveWatch();
        });
      } else {
        startKeepaliveWatch();
      }
      try {
        var ctx = ensureAudioContext();
        if (ctx && ctx.state === "suspended") ctx.resume();
        if (!root.__hkParkingOscKeepalive) startOscKeepalive();
      } catch (e2) {}
      return true;
    } catch (e) {
      return false;
    }
  }

  function stopUnlockedMedia() {
    root.__hkParkingMediaPlayPending = false;
    root.__hkParkingMediaPlayPendingSince = 0;
    root.__hkParkingAudioCtxPlayGen = (root.__hkParkingAudioCtxPlayGen || 0) + 1;
    stopAudioContextSource();
    stopOscKeepalive();
    root.__hkParkingMediaKeepalive = false;
    stopKeepaliveWatch();
    var a = root.__hkParkingMediaAudio;
    if (!a) return;
    try { a.loop = false; } catch (e) {}
    try { a.pause(); } catch (e) {}
    try { a.currentTime = 0; } catch (e) {}
    a.onended = null;
    a.onerror = null;
  }

  function unlockMediaNow() {
    speech.unlock();
    markVoiceQueryArmed();
    // 1) Web Audio：必須喺手勢 call stack 同步 resume＋開 oscillator（唔好放 .then）
    try {
      var ctx = ensureAudioContext();
      if (ctx) {
        if (ctx.state === "suspended") {
          try { ctx.resume(); } catch (e) {}
        }
        startOscKeepalive();
      }
    } catch (e) {}
    // 2) HTML5：靜音 loop keepalive（語音辨識可等十幾秒，唔 loop 會被 iOS 收回權限）
    try {
      var a = ensureUnlockedMediaAudio();
      a.muted = false;
      a.volume = 0.01;
      a.loop = true;
      root.__hkParkingMediaKeepalive = true;
      a.onended = null;
      a.onerror = null;
      // 已在播同一段 keepalive 就唔好重設 src（避免語音「開始→停止」兩下手勢打斷）
      var sameSilent = false;
      try {
        sameSilent = a.src && a.src.indexOf("data:audio/wav") === 0 && !a.paused && a.loop;
      } catch (e2) {}
      if (!sameSilent) {
        a.src = SILENT_WAV;
      }
      var markUnlockedAndFlush = function () {
        root.__hkParkingMediaUnlocked = true;
        startKeepaliveWatch();
        // 擋查結果好快返：MP3 可能已排隊，unlock 完成先至可換 src
        try { flushPendingUnlockedMedia(); } catch (e3) {}
      };
      var p = a.play();
      if (p && typeof p.then === "function") {
        p.then(markUnlockedAndFlush).catch(markUnlockedAndFlush);
      } else {
        markUnlockedAndFlush();
      }
      // 防 silent play() 掛住：逾時仍標 unlocked 並 flush 排隊中嘅自動播
      root.setTimeout(function () {
        if (root.__hkParkingMediaUnlocked) return;
        if (!root.__hkParkingMediaKeepalive && !root.__hkParkingMediaAudio) return;
        markUnlockedAndFlush();
      }, 450);
    } catch (e) {}
  }

  function playViaAudioContext(src, cbs) {
    var ctx = root.__hkParkingAudioCtx;
    if (!ctx || !root.__hkParkingAudioCtxUnlocked) return false;
    // keepalive 期間優先 HTML5 換 src；AudioContext 長等待後常係 suspended
    if (root.__hkParkingMediaKeepalive) return false;
    var playGen = (root.__hkParkingAudioCtxPlayGen || 0) + 1;
    root.__hkParkingAudioCtxPlayGen = playGen;
    root.__hkParkingMediaPlayPending = true;
    var fail = function () {
      if (playGen !== root.__hkParkingAudioCtxPlayGen) return;
      root.__hkParkingMediaPlayPending = false;
      if (typeof cbs.onError === "function") cbs.onError();
    };
    var resumeAndDecode = function () {
      if (playGen !== root.__hkParkingAudioCtxPlayGen) return;
      root.fetch(src)
        .then(function (res) { return res.arrayBuffer(); })
        .then(function (ab) {
          if (playGen !== root.__hkParkingAudioCtxPlayGen) return null;
          return ctx.decodeAudioData(ab.slice(0));
        })
        .then(function (buffer) {
          if (!buffer || playGen !== root.__hkParkingAudioCtxPlayGen) return;
          stopAudioContextSource();
          stopOscKeepalive();
          root.__hkParkingMediaKeepalive = false;
          stopKeepaliveWatch();
          try {
            var aKeep = root.__hkParkingMediaAudio;
            if (aKeep) {
              aKeep.loop = false;
              aKeep.pause();
            }
          } catch (e) {}
          var node = ctx.createBufferSource();
          node.buffer = buffer;
          node.connect(ctx.destination);
          node.onended = function () {
            if (root.__hkParkingAudioCtxSource === node) {
              root.__hkParkingAudioCtxSource = null;
            }
            root.__hkParkingMediaPlayPending = false;
            if (typeof cbs.onEnded === "function") cbs.onEnded();
          };
          root.__hkParkingAudioCtxSource = node;
          root.__hkParkingMediaPlayPending = false;
          node.start(0);
          if (typeof cbs.onStart === "function") cbs.onStart();
        })
        .catch(fail);
    };
    try {
      if (ctx.state === "suspended") {
        var rp = ctx.resume();
        if (rp && typeof rp.then === "function") {
          rp.then(resumeAndDecode).catch(fail);
        } else {
          resumeAndDecode();
        }
      } else {
        resumeAndDecode();
      }
      return true;
    } catch (e) {
      root.__hkParkingMediaPlayPending = false;
      return false;
    }
  }

  function revokeMediaBlobUrl() {
    if (!root.__hkParkingMediaBlobUrl) return;
    try { root.URL.revokeObjectURL(root.__hkParkingMediaBlobUrl); } catch (e) {}
    root.__hkParkingMediaBlobUrl = null;
  }

  function dataUriToBlobUrl(dataUri) {
    // 唔用 fetch(data:)：iOS 對大 MP3 data URI 有時會卡住，令 playPending 掛住、自動播無聲
    var comma = dataUri.indexOf(",");
    if (comma < 0) return null;
    var meta = dataUri.slice(0, comma);
    var data = dataUri.slice(comma + 1);
    var mimeMatch = meta.match(/data:([^;]+)/);
    var mime = (mimeMatch && mimeMatch[1]) || "audio/mpeg";
    var binary;
    try {
      if (/;base64/i.test(meta)) binary = root.atob(data);
      else binary = root.decodeURIComponent(data);
    } catch (e) {
      return null;
    }
    var len = binary.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    revokeMediaBlobUrl();
    root.__hkParkingMediaBlobUrl = root.URL.createObjectURL(
      new root.Blob([bytes], { type: mime })
    );
    return root.__hkParkingMediaBlobUrl;
  }

  function startHtml5Src(a, src, cbs, wasKeepalive) {
    var finishStart = function () {
      var p = a.play();
      if (p && typeof p.then === "function") {
        p.then(function () {
          root.__hkParkingMediaPlayPending = false;
          root.__hkParkingMediaPlayPendingSince = 0;
          if (typeof cbs.onStart === "function") cbs.onStart();
        }).catch(function () {
          root.__hkParkingMediaPlayPending = false;
          root.__hkParkingMediaPlayPendingSince = 0;
          if (typeof cbs.onError === "function") cbs.onError();
        });
      } else {
        root.__hkParkingMediaPlayPending = false;
        root.__hkParkingMediaPlayPendingSince = 0;
        if (typeof cbs.onStart === "function") cbs.onStart();
      }
    };
    // data URI 大 MP3 → 同步轉 blob URL（避免 iOS fetch 掛起）
    if (src && src.indexOf("data:audio/") === 0 && root.URL) {
      var blobUrl = dataUriToBlobUrl(src);
      if (blobUrl) {
        a.src = blobUrl;
        try { a.load(); } catch (e) {}
        finishStart();
        return;
      }
    }
    a.src = src;
    try { a.load(); } catch (e) {}
    finishStart();
  }

  function playViaHtml5Audio(src, cbs) {
    try {
      var a = ensureUnlockedMediaAudio();
      stopAudioContextSource();
      var wasKeepalive = !!root.__hkParkingMediaKeepalive;
      root.__hkParkingMediaKeepalive = false;
      stopKeepaliveWatch();
      root.__hkParkingMediaPlayPending = true;
      root.__hkParkingMediaPlayPendingSince = Date.now();
      try { a.loop = false; } catch (e) {}
      // keepalive 中唔好先 pause：iOS 會當解鎖失效
      if (!wasKeepalive) {
        try { a.pause(); } catch (e) {}
      }
      a.muted = false;
      a.volume = 1;
      var wrapped = {
        onStart: function () {
          clearVoiceQueryArmed();
          if (typeof cbs.onStart === "function") cbs.onStart();
        },
        onEnded: function () {
          root.__hkParkingMediaPlayPending = false;
          root.__hkParkingMediaPlayPendingSince = 0;
          revokeMediaBlobUrl();
          if (typeof cbs.onEnded === "function") cbs.onEnded();
        },
        onError: function () {
          root.__hkParkingMediaPlayPending = false;
          root.__hkParkingMediaPlayPendingSince = 0;
          revokeMediaBlobUrl();
          if (typeof cbs.onError === "function") cbs.onError();
        }
      };
      a.onended = wrapped.onEnded;
      a.onerror = wrapped.onError;
      startHtml5Src(a, src, wrapped, wasKeepalive);
      return true;
    } catch (e) {
      root.__hkParkingMediaPlayPending = false;
      root.__hkParkingMediaPlayPendingSince = 0;
      if (typeof cbs.onError === "function") cbs.onError();
      return false;
    }
  }

  function playUnlockedMedia(src, cbs) {
    cbs = cbs || {};
    if (!src) {
      if (typeof cbs.onError === "function") cbs.onError();
      return false;
    }
    // iOS：silent keepalive 嘅 play() 未 resolve 前唔好換 MP3 src，否則會丟手勢解鎖
    // （擋查／規則攔截好快返結果時最易踩中；語音長等待通常已 unlocked）
    if (!root.__hkParkingMediaUnlocked) {
      root.__hkParkingPendingAutoSrc = src;
      root.__hkParkingPendingAutoCbs = cbs;
      return true;
    }
    // 語音長等待：優先同一 HTML5 元素（keepalive → 換 MP3）
    if (root.__hkParkingMediaKeepalive || root.__hkParkingMediaUnlocked) {
      stopOscKeepalive();
      if (playViaHtml5Audio(src, cbs)) return true;
    }
    var acCbs = {
      onStart: cbs.onStart,
      onEnded: cbs.onEnded,
      onError: function () { playViaHtml5Audio(src, cbs); }
    };
    if (playViaAudioContext(src, acCbs)) return true;
    return playViaHtml5Audio(src, cbs);
  }

  function isUnlockedMediaPlaying() {
    // playPending 逾時當失敗（大 data URI／decode 卡住時唔好永遠擋住自動播 fallback）
    if (root.__hkParkingMediaPlayPending) {
      var since = root.__hkParkingMediaPlayPendingSince || 0;
      if (since && (Date.now() - since) > 8000) {
        root.__hkParkingMediaPlayPending = false;
      } else {
        return true;
      }
    }
    if (root.__hkParkingAudioCtxSource) return true;
    var a = root.__hkParkingMediaAudio;
    if (!a) return false;
    try {
      // keepalive 靜音 loop 唔算「正在朗讀」
      if (root.__hkParkingMediaKeepalive) return false;
      // 唔要求 currentTime > 0：剛起播嗰幾百 ms 仍係 0，否則重試會 stopParent 切斷「你想」
      return !a.paused && !a.ended;
    } catch (e) {
      return false;
    }
  }

  function queueUnlockedMedia(src, cbs) {
    // 語音結果 iframe 可能喺 setTimeout 先叫 play；改排隊由 parent keepalive 期間即播
    root.__hkParkingPendingAutoSrc = src || null;
    root.__hkParkingPendingAutoCbs = cbs || null;
    flushPendingUnlockedMedia();
  }

  function flushPendingUnlockedMedia() {
    var src = root.__hkParkingPendingAutoSrc;
    if (!src) return false;
    // 未完成 HTML5 unlock：保留排隊，等 unlock .then／timeout flush
    if (!root.__hkParkingMediaUnlocked) {
      if (isVoiceQueryArmed()) {
        try { reviveKeepaliveIfPossible(); } catch (e) {}
      }
      return false;
    }
    if (!root.__hkParkingMediaKeepalive && root.__hkParkingMediaUnlocked) {
      reviveKeepaliveIfPossible();
    }
    root.__hkParkingPendingAutoSrc = null;
    var cbs = root.__hkParkingPendingAutoCbs || {};
    root.__hkParkingPendingAutoCbs = null;
    return playUnlockedMedia(src, cbs);
  }

  root.__hkParkingUnlockMedia = unlockMediaNow;
  root.__hkParkingPlayUnlockedMedia = playUnlockedMedia;
  root.__hkParkingQueueUnlockedMedia = queueUnlockedMedia;
  root.__hkParkingFlushPendingUnlockedMedia = flushPendingUnlockedMedia;
  root.__hkParkingStopUnlockedMedia = stopUnlockedMedia;
  root.__hkParkingIsUnlockedMediaPlaying = isUnlockedMediaPlaying;
  root.__hkParkingReviveKeepalive = reviveKeepaliveIfPossible;
  root.__hkParkingMarkVoiceQueryArmed = markVoiceQueryArmed;
  root.__hkParkingIsVoiceQueryArmed = isVoiceQueryArmed;

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
  function collapseHiddenSpeechIframes() {
    doc.querySelectorAll("iframe").forEach(function (iframe) {
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

  function armVoiceUnlock(ev) {
    // 語音掣喺主頁 DOM：手勢內 unlock／keepalive，先撑到辨識完自動播
    // 開始錄同停止錄都會觸發 → 第二次手勢可重新撑住長等待
    try { unlockSpeechNow(); } catch (e) {}
    try { markVoiceQueryArmed(); } catch (e) {}
    try { stopSuggestionPlaybackOnly(); } catch (e) {}
  }

  function bindAudioInputUnlock() {
    var nodes = doc.querySelectorAll(
      '.st-key-voice_input, [data-testid="stAudioInput"], .st-key-voice_input button, [data-testid="stAudioInput"] button'
    );
    nodes.forEach(function (el) {
      if (el.dataset.voiceUnlockBound) return;
      el.dataset.voiceUnlockBound = "1";
      el.addEventListener("touchstart", armVoiceUnlock, { capture: true, passive: true });
      el.addEventListener("pointerdown", armVoiceUnlock, true);
      el.addEventListener("click", armVoiceUnlock, true);
    });
  }

  function styleMicButtons() {
    // 舊 mic_recorder 已棄用；改綁 st.audio_input（主頁手勢）
    bindAudioInputUnlock();
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
      // 查詢掣手勢內只 unlock／keepalive；自動播等 MediaUnlocked 後由結果頁排隊播放
      // （擋查太快時若喺 silent play() 未完成就換 MP3，iOS 會丟解鎖）
      btn.addEventListener("pointerdown", function () {
        stopSuggestionPlaybackOnly();
        unlockSpeechNow();
      }, true);
      btn.addEventListener("click", function () {
        // 唔 stop：避免打斷 pointerdown 已開始嘅 unlock keepalive
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

  // 查詢送出後 blur，唔好留住 primary 紅邊／「Press Enter to submit form」
  function blurUserQueryInput() {
    var input = doc.querySelector(".st-key-user_query input")
      || doc.querySelector('div[data-testid="stTextInput"] input');
    if (!input) return;
    try { input.blur(); } catch (e) {}
  }

  function scheduleBlurUserQuery() {
    [0, 50, 150, 400, 800].forEach(function (ms) {
      setTimeout(blurUserQueryInput, ms);
    });
  }

  function bindQueryInputBlurOnSubmit() {
    var input = doc.querySelector(".st-key-user_query input")
      || doc.querySelector('div[data-testid="stTextInput"] input');
    if (!input || input.dataset.mzdBlurBound) return;
    input.dataset.mzdBlurBound = "1";
    input.addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      // Enter 送出唔經「查詢」掣 pointerdown；必須喺手勢內停舊播＋解鎖
      stopSuggestionPlaybackOnly();
      unlockSpeechNow();
      scheduleBlurUserQuery();
    });
  }

  function bindQueryButtonBlur() {
    doc.querySelectorAll('div[data-testid="stFormSubmitButton"] button').forEach(function (btn) {
      if (btn.dataset.mzdBlurBound) return;
      btn.dataset.mzdBlurBound = "1";
      btn.addEventListener("pointerdown", scheduleBlurUserQuery, true);
      btn.addEventListener("click", scheduleBlurUserQuery, true);
    });
  }

  function blurUserQueryIfMarked() {
    // 用 parent flag（唔注入／刪除 React markdown 節點）
    if (!root.__hkParkingBlurQueryOnce) return;
    root.__hkParkingBlurQueryOnce = false;
    blurUserQueryInput();
  }

  // 與 format_hong_kong_now() 一致：只改 data-mzd-clock，唔動 React 子節點
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
    if (el.getAttribute("data-mzd-clock") !== text) {
      el.setAttribute("data-mzd-clock", text);
    }
  }

  function runUiPolish() {
    styleMicButtons();
    styleMicLikeActionButtons();
    collapseJsEvalIframes();
    collapseHiddenSpeechIframes();
    bindQueryStopSpeech();
    bindSuggestionSpeechButtons();
    hideSpeechEndedSentinel();
    bindQueryInputBlurOnSubmit();
    bindQueryButtonBlur();
    blurUserQueryIfMarked();
    try { flushPendingUnlockedMedia(); } catch (e) {}
    tickHkClock();
  }
  runUiPolish();
  // 單一 parent interval：每次 Streamlit remount components.html 唔好再開一個
  if (root.__hkParkingUiInterval) {
    try { root.clearInterval(root.__hkParkingUiInterval); } catch (e) {}
  }
  root.__hkParkingUiInterval = root.setInterval(runUiPolish, 400);
})();
</script>
        """,
        height=1,
    )


def normalize_asr_query(query: str) -> tuple[str, str | None]:
    """校正常見粵語語音誤辨（例如「易泊」→「200」、「易唔易」→「252」、「容」→「用」）。

    「比較200／邊度200／九點半252泊車」等喺本 app 幾乎一定係誤辨；
    若數字後接價錢單位（蚊／元）則不改，避免誤傷。
    回傳 (校正後句子, 提示字串或 None)。
    """
    original = (query or "").strip()
    text, changed = normalize_asr_query_text(original)
    if not changed:
        return original, None
    note = f"已校正語音／輸入：「{text}」（原：「{original}」）"
    return text, note


def apply_pending_input(widget_key: str, pending_key: str) -> None:
    """必須在對應 text_input 建立之前呼叫；把語音結果寫入 widget state。"""
    if pending_key not in st.session_state:
        return
    # 只喺呢度改 widget key，避免 StreamlitAPIException
    st.session_state[widget_key] = st.session_state.pop(pending_key)


def handle_voice_transcription(
    audio,
    *,
    last_id_key: str,
    pending_key: str,
    auto_submit_key: str,
    feedback_key: str,
) -> None:
    """辨識後只寫入 pending_key（唔好直接改 text_input 的 key），再 rerun。"""
    if not pending_key.startswith("_pending_"):
        raise ValueError(f"pending_key 必須係暫存鍵，收到：{pending_key}")
    normalized = _normalize_voice_audio(audio)
    if not normalized or normalized.get("id") == st.session_state.get(last_id_key):
        return
    st.session_state[last_id_key] = normalized["id"]
    with st.spinner("正在辨識語音..."):
        try:
            voice_text = transcribe_audio(normalized)
            service_error = None
        except Exception as e:
            voice_text = ""
            service_error = str(e)

    voice_text = (voice_text or "").strip()
    if voice_text:
        voice_text, asr_note = normalize_asr_query(voice_text)
        if asr_note:
            st.session_state["_asr_correction_note"] = asr_note

    # 成功／空白／錯誤都更新；空字串會清空輸入框
    st.session_state[pending_key] = voice_text
    # 下一輪清掉 audio_input，方便再錄；並標記今次係語音送出（自動播解鎖靠 parent 手勢）
    st.session_state["_clear_voice_input"] = True
    st.session_state["_voice_query_unlock"] = True
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
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# 歷史紀錄重查：喺 form 前套用 pending，避免改已建立嘅 widget
if st.session_state.pop("_history_rerun", False):
    st.session_state.auto_submit_query = True
apply_pending_input("user_query", "_pending_history_query")

# 文字輸入 + 查詢 + 語音同一列（Enter 或按「查詢」皆可送出）
st.caption(
    "輸入查詢（例如：「天水圍今晚八點邊度較易泊？」；"
    "「APM 星期三朝早九點附近邊五個停車場好泊？」；"
    "「彩雲邨附近停車場收費如何？」；"
    "「我想知道屯門兆康附近星期六下晝三點半有邊啲位泊」）"
    f"· 未指定數量時預設顯示前 {DEFAULT_TOP_N} 名"
)
# 語音改用 st.audio_input（主頁 DOM）：iPhone 手勢先解鎖得到自動朗讀
if st.session_state.pop("_clear_voice_input", False):
    st.session_state.pop("voice_input", None)

form_col, voice_col = st.columns([5.6, 1.6], vertical_alignment="bottom")
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
    audio = st.audio_input(
        "語音",
        sample_rate=16000,
        label_visibility="collapsed",
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
asr_note = st.session_state.pop("_asr_correction_note", None)
if asr_note:
    st.caption(asr_note)

# 搜尋歷史（可一鍵重查）
if st.session_state.search_history:
    st.caption("最近搜尋")
    hist_cols = st.columns(min(4, len(st.session_state.search_history)))
    for i, past_q in enumerate(st.session_state.search_history[:4]):
        label = past_q if len(past_q) <= 18 else past_q[:16] + "…"
        if hist_cols[i].button(
            label,
            key=f"hist_btn_{i}",
            use_container_width=True,
            help=past_q,
        ):
            st.session_state["_pending_history_query"] = past_q
            st.session_state["_history_rerun"] = True
            st.rerun()

# 送出後 blur 查詢欄（去掉 focus 紅邊），文字保留。
# 用 parent flag，唔好注入／刪除 React markdown 節點（會觸發 removeChild）。
if submitted or st.session_state.get("auto_submit_query"):
    components.html(
        """
        <script>
        (function () {
          var root = window.parent;
          root.__hkParkingBlurQueryOnce = true;
          try {
            var doc = root.document;
            var input = doc.querySelector(".st-key-user_query input")
              || doc.querySelector('div[data-testid="stTextInput"] input');
            if (input) input.blur();
          } catch (e) {}
        })();
        </script>
        """,
        height=1,
    )
    # 唔用 st.info「正在處理」：會同結果同一輪一齊顯示，睇落卡住；進度改靠下方 spinner



@st.cache_data(ttl=600, show_spinner="正在載入停車場清單…")
def load_carparks_meta() -> pd.DataFrame:
    """只載入車場靜態資料（附近篩選／預設地圖用；唔拉歷史空位）。"""
    df = load_carparks_frame()
    if df.empty:
        return df
    # 一律字串：避免 31（int）同 baseline key「31」對唔上而整場被 skip
    df["park_id"] = df["park_id"].astype(str)
    df["sub_district"] = df["google_area"].fillna(df["district"])
    return df


@st.cache_data(ttl=600, show_spinner="正在載入歷史空位…")
def load_data(park_ids: tuple[str, ...] | None = None) -> pd.DataFrame:
    """
    從 Supabase 載入空位歷史（香港時間；cache 10 分鐘）。
    傳 park_ids 時只拉該批車場（附近查詢大幅加速）。
    """
    df = load_vacancy_frame(park_ids=list(park_ids) if park_ids else None)
    if df.empty:
        return df
    df["park_id"] = df["park_id"].astype(str)
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


def build_park_baselines(
    df: pd.DataFrame,
) -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    """
    每車場歷史平均空位 + 樣本數（忽略 vacancy=-1）。
    回傳 mean／count 各四層：
      (park_id, half_hour, dow)
      → (park_id, half_hour, is_weekend)
      → (park_id, half_hour)
      → park_id
    """
    empty = ({}, {}, {}, {}, {}, {}, {}, {})
    if df is None or df.empty:
        return empty
    valid = df[df["vacancy"] >= 0].copy()
    if valid.empty:
        return empty
    valid["park_id"] = valid["park_id"].astype(str)
    valid["is_weekend"] = (valid["dow"] >= 5).astype(int)

    def _maps(keys: list[str]) -> tuple[dict, dict]:
        agg = valid.groupby(keys, sort=False)["vacancy"].agg(["mean", "count"])
        means = agg["mean"].to_dict()
        counts = agg["count"].astype(int).to_dict()
        return means, counts

    by_slot_dow, n_slot_dow = _maps(["park_id", "half_hour", "dow"])
    by_slot_weekend, n_slot_weekend = _maps(["park_id", "half_hour", "is_weekend"])
    by_slot, n_slot = _maps(["park_id", "half_hour"])
    by_park = valid.groupby("park_id", sort=False)["vacancy"].mean().to_dict()
    n_park = valid.groupby("park_id", sort=False)["vacancy"].count().astype(int).to_dict()
    return (
        by_slot_dow,
        by_slot_weekend,
        by_slot,
        by_park,
        n_slot_dow,
        n_slot_weekend,
        n_slot,
        n_park,
    )


@st.cache_data(ttl=600, show_spinner="")
def load_park_baselines(
    park_ids: tuple[str, ...] | None = None,
    _cache_ver: str = "v4-equiv-baselines",
) -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    """
    載入歷史 baseline（方案 1：優先摘要表／本地 parquet；
    若無則退回由 vacancy_records 即時聚合）。

    重要：parquet／摘要表可能只有 tdc* id、冇政府數字 id。
    對缺失嘅 park_id 會再用 vacancy_records 補建，避免成批有收費／地標場被 skip。

    附近查詢會先 dedupe 雙編號，可能淨留政府數字 id；呢度會一併載入
    等效 tdc* 基準，避免預測跌去「即時空位」而令唔同日子／時段睇落冇變化。
    """
    id_list = [coerce_park_id(p) for p in park_ids] if park_ids is not None else None
    id_list = [p for p in id_list if p] if id_list is not None else None
    if id_list is not None:
        eq_map = load_equivalent_park_ids()
        expanded: list[str] = []
        seen: set[str] = set()
        for pid in id_list:
            for cand in (pid, *eq_map.get(pid, ())):
                c = coerce_park_id(cand)
                if c and c not in seen:
                    seen.add(c)
                    expanded.append(c)
        id_list = expanded
    df = load_baselines_dataframe(park_ids=id_list)
    packed = baselines_frame_to_maps(df)
    if id_list is not None:
        have = parks_present_in_baselines(packed)
        missing = [p for p in id_list if p not in have]
        if missing:
            extra = build_park_baselines(
                load_data(tuple(sorted(set(missing))))
            )
            packed = merge_baseline_packs(packed, extra)
        return packed
    if any(packed):
        return packed
    return build_park_baselines(load_data(park_ids))


@st.cache_data(ttl=6 * 3600)
def load_equivalent_park_ids() -> dict[str, tuple[str, ...]]:
    """等效車場 id（運輸署雙編號）：借 baseline／收費時用。"""
    meta = load_carparks_meta()[
        ["park_id", "name", "latitude", "longitude"]
    ].drop_duplicates(subset=["park_id"])
    return build_equivalent_park_id_map(
        meta,
        max_distance_km=EQUIVALENT_CARPARK_MAX_KM,
        normalize_name=_normalize_park_name,
        haversine_km=_haversine_km,
    )


def _shrink_mean(
    primary: float | None,
    n_primary: int,
    prior: float | None,
    *,
    k: float = BASELINE_PRIOR_K,
) -> float | None:
    """樣本少時向 prior 收斂，保留星期差異同時避免噪音。"""
    if primary is None and prior is None:
        return None
    if primary is None:
        return float(prior) if prior is not None else None
    if prior is None or n_primary <= 0:
        return float(primary)
    w = n_primary / (n_primary + k)
    return float(w * primary + (1.0 - w) * prior)


def _predict_park_vacancy_for_id(
    park_id: str,
    half_hour: int,
    dow: int,
    packed: tuple,
) -> tuple[float | None, dict]:
    """單一 park_id 嘅 baseline 預測（唔做等效回退）。"""
    (
        by_slot_dow,
        by_slot_weekend,
        by_slot,
        by_park,
        n_slot_dow,
        n_slot_weekend,
        n_slot,
        n_park,
    ) = packed

    is_weekend = 1 if dow >= 5 else 0
    key_dow = (park_id, half_hour, dow)
    key_we = (park_id, half_hour, is_weekend)
    key_slot = (park_id, half_hour)

    mean_dow = by_slot_dow.get(key_dow)
    n_dow = int(n_slot_dow.get(key_dow, 0))
    mean_we = by_slot_weekend.get(key_we)
    n_we = int(n_slot_weekend.get(key_we, 0))
    mean_slot = by_slot.get(key_slot)
    n_s = int(n_slot.get(key_slot, 0))
    mean_park = by_park.get(park_id)
    n_p = int(n_park.get(park_id, 0))

    prior = mean_we if mean_we is not None else mean_slot
    if prior is None:
        prior = mean_park
    pred = _shrink_mean(mean_dow, n_dow, prior)
    source = "同星期同時段"
    n_used = n_dow

    if pred is None:
        pred = _shrink_mean(mean_we, n_we, mean_slot if mean_slot is not None else mean_park)
        source = "平日／週末同時段"
        n_used = n_we
    if pred is None:
        pred = mean_slot if mean_slot is not None else mean_park
        source = "同時段" if mean_slot is not None else "車場全日平均"
        n_used = n_s if mean_slot is not None else n_p

    if pred is None:
        return None, {"source": "無資料", "n": 0, "confidence": "低"}

    if n_used >= 6:
        confidence = "高"
    elif n_used >= 2:
        confidence = "中"
    else:
        confidence = "低"

    return float(pred), {
        "source": source,
        "n": int(n_used),
        "confidence": confidence,
        "mean_dow": None if mean_dow is None else float(mean_dow),
        "mean_weekend_group": None if mean_we is None else float(mean_we),
    }


def predict_park_vacancy_baseline(
    park_id,
    half_hour: int,
    dow: int,
    baselines: tuple | None = None,
) -> tuple[float | None, dict]:
    """
    Per-park baseline（含 shrinkage）：優先同車場＋半小時＋星期，
    向平日／週末／同時段均值收斂。回傳 (預測值, 元資料)。

    通用後備（適用全部車場，唔限單一商場）：
    1) 本 id 歷史基準
    2) 名稱＋距離等效雙編號嘅歷史基準
    3) 政府即時空位（歷史摘要缺數字 id 時仍可上榜同顯示收費）
    """
    packed = baselines or load_park_baselines()
    park_id = coerce_park_id(park_id)
    if not park_id:
        return None, {"source": "無資料", "n": 0, "confidence": "低"}

    candidates = [park_id]
    for alt in load_equivalent_park_ids().get(park_id, ()):
        alt = coerce_park_id(alt)
        if alt and alt not in candidates:
            candidates.append(alt)

    for pid in candidates:
        pred, meta = _predict_park_vacancy_for_id(pid, half_hour, dow, packed)
        if pred is None:
            continue
        if pid != park_id:
            meta = dict(meta)
            meta["source"] = f"{meta.get('source', '')}（等效資料）".strip()
            meta["equivalent_park_id"] = pid
        return pred, meta

    # 歷史基準完全無：用即時空位頂住（多數有收費規則嘅政府數字 id 屬此情況）
    live_map = load_live_private_car_vacancies()
    for pid in candidates:
        live = live_map.get(pid)
        if not live:
            continue
        try:
            vac = float(live.get("vacancy"))
        except (TypeError, ValueError):
            continue
        if vac < 0:
            continue
        return vac, {
            "source": "即時空位（無歷史基準）",
            "n": 1,
            "confidence": "低",
            "mean_dow": None,
            "mean_weekend_group": None,
            "equivalent_park_id": None if pid == park_id else pid,
        }
    return None, {"source": "無資料", "n": 0, "confidence": "低"}


@st.cache_data(ttl=120, show_spinner="")
def load_live_private_car_vacancies() -> dict[str, dict]:
    """政府即時空位 API → park_id -> {vacancy, lastupdate}。"""
    try:
        response = requests.get(
            CARPARK_INFO_VACANCY_URL,
            params={
                "data": "vacancy",
                "lang": "zh_TW",
                "vehicleTypes": "privateCar",
            },
            timeout=20,
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
        private = item.get("privateCar") or []
        if not private:
            continue
        v = private[0] or {}
        raw = v.get("vacancy")
        try:
            vacancy = int(raw)
        except (TypeError, ValueError):
            continue
        if vacancy < 0:
            continue
        out[park_id] = {
            "vacancy": vacancy,
            "lastupdate": v.get("lastupdate"),
        }
    return out


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


@st.cache_data(ttl=6 * 3600)
def load_fee_park_id_aliases() -> dict[str, str]:
    """無收費規則嘅 park_id → 等效且有規則嘅 id（解決運輸署雙 id）。"""
    fee_map = load_carpark_fee_rules()
    if not fee_map:
        return {}
    meta = load_carparks_meta()[
        ["park_id", "name", "latitude", "longitude"]
    ].drop_duplicates(subset=["park_id"])
    return build_fee_park_id_aliases(
        meta,
        set(fee_map.keys()),
        max_distance_km=EQUIVALENT_CARPARK_MAX_KM,
        normalize_name=_normalize_park_name,
        haversine_km=_haversine_km,
    )


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
    若 park_id 本身無規則，會試等效 id（運輸署雙編號）。
    """
    fee_map = fee_map if fee_map is not None else load_carpark_fee_rules()
    pid = str(park_id)
    rules = fee_map.get(pid)
    if not rules:
        alias = load_fee_park_id_aliases().get(pid)
        if alias and alias != pid:
            rules = fee_map.get(alias)
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
            winner_idx = int(best.name)
        else:
            winner_idx = min(
                members,
                key=lambda idx: _prefer_canonical_park_id(work.iloc[idx][id_col]),
            )
        # 去重後仍保留叢集內任一有效收費（避免 tdc 分身蓋過有規則嘅 id）
        if "收費" in work.columns:
            merged_fee = first_available_fee(
                work.loc[members, "收費"].tolist()
            )
            if merged_fee is not None:
                work.at[winner_idx, "收費"] = merged_fee
        keep_indices.append(winner_idx)

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
    只用車場靜態清單，唔拉歷史空位。
    """
    parks = (
        load_carparks_meta()[
            ["park_id", "name", "district", "sub_district", "latitude", "longitude"]
        ]
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
    intent["day_next_week"] = False
    return intent


def has_explicit_clock_time(query: str) -> bool:
    """查詢是否有具體鐘點（不含「後晚／聽晚」等相對日期用語）。"""
    q = query.replace(" ", "")
    # 先去掉相對日期與「而家／現在」，避免誤判
    q = re.sub(RELATIVE_DAY_STRIP_RE, "", q)
    q = re.sub(r"嚟緊|黎緊|即將|就嚟|快將|快到", "", q)
    q = re.sub(r"而家|依家|現在|目前|即刻|此刻|呢一刻|right\s*now|\bnow\b", "", q, flags=re.I)
    if re.search(r"\d{1,2}[:：]\d{1,2}", q):
        return True
    if re.search(r"\d{1,2}[點时時]", q):
        return True
    if re.search(r"[一二兩三两四五六七八九十廿][點时時]", q):
        return True
    period = r"(上午|下午|上晝|下晝|晏晝|早上|朝早|凌晨|夜晚|晚上|傍晚)"
    if re.search(period + r"\d", q):
        return True
    if re.search(period + r"[一二兩三两四五六七八九十廿]", q):
        return True
    if "中午" in q or "午晝" in q:
        return True
    if re.search(r"(a\.?m\.?|p\.?m\.?)", q, re.I):
        return True
    return False


def _cn_hour_token_to_int(token: str) -> int | None:
    """中文鐘點詞 → 1–12（或 0 表示十二點）；無法解析則 None。"""
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    mapping = {
        "零": 0,
        "〇": 0,
        "一": 1,
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
        "十一": 11,
        "十二": 12,
        "廿": 20,
    }
    if token in mapping:
        return mapping[token]
    if token.startswith("十") and len(token) == 2 and token[1] in mapping:
        return 10 + mapping[token[1]]
    return None


def extract_clock_time_from_query(query: str) -> tuple[int, int] | None:
    """規則抽出鐘點（hour 0–23, minute 0/30）；抽不到則 None。

    用作 LLM 漏填 hour／minute 時的後備，避免「星期日上午10:30」被當無時間。
    """
    q = (query or "").replace(" ", "")
    if not q:
        return None

    period_shift = None  # None / "am" / "pm"
    if re.search(
        r"下午|下晝|晏晝|夜晚|晚上|傍晚|黃昏|黄昏|深夜|今晚|聽晚|明晚|後晚|大後晚|今夜|明夜|p\.?m\.?",
        q,
        flags=re.I,
    ):
        period_shift = "pm"
    elif re.search(
        r"上午|上晝|早上|朝早|凌晨|清晨|今早|今朝|今晨|聽朝|明早|a\.?m\.?",
        q,
        flags=re.I,
    ):
        period_shift = "am"
    elif "中午" in q or "午晝" in q or "正午" in q:
        period_shift = "noon"

    hour = None
    minute = 0

    m = re.search(r"(?<!\d)([0-2]?\d)[:：]([0-5]?\d)", q)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(
            r"(?<!\d)([0-2]?\d|[一二兩三两四五六七八九十廿]+)[點时時](半)?",
            q,
        )
        if not m:
            return None
        parsed = _cn_hour_token_to_int(m.group(1))
        if parsed is None:
            return None
        hour = parsed
        minute = 30 if m.group(2) else 0

    if hour is None:
        return None

    # 已是 13–23 → 視為 24 小時制
    if hour >= 13:
        return normalize_half_hour(hour, minute)

    if period_shift == "pm":
        if hour < 12:
            hour += 12
    elif period_shift == "am":
        if hour == 12:
            hour = 0
    elif period_shift == "noon" and hour < 12:
        hour = 12
    # 未標 am/pm 且 1–12：保留字面（稍後 ambiguous 檢查會攔截）

    return normalize_half_hour(hour, minute)


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
    "6th": "第六",
    "7th": "第七",
    "8th": "第八",
    "9th": "第九",
    "10th": "第十",
}
# 名次粵語「第 N」（排名用「二」唔用「兩」）
_RANK_NUM_CN = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
}


def _rank_spoken_yue(rank: str) -> str:
    """1st／2nd… → 第一／第二…；超出對照表時動態組「第N」。"""
    key = str(rank)
    if key in RANK_SPOKEN_YUE:
        return RANK_SPOKEN_YUE[key]
    m = re.match(r"^(\d+)", key)
    if not m:
        return key
    n = int(m.group(1))
    if n in _RANK_NUM_CN:
        return f"第{_RANK_NUM_CN[n]}"
    return f"第{n}"

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
# 商場／品牌英文名：zh-HK TTS 直讀好怪 → 改口語／中文讀法（只影響朗讀，唔改畫面）
LATIN_BRAND_SPOKEN = {
    # 逼 TTS 按英文音節讀：Mi．ki．ki（唔好讀成一個怪詞／中文）
    "MIKIKI": "Mi ki ki",
}
# TTS 詞庫常缺／跳過嘅罕用字 → 同音／常用字（只影響朗讀）
TTS_CHAR_SPOKEN = str.maketrans(
    {
        "壐": "璽",  # 天壐·天 → 天璽（資料罕用字，Edge／系統 TTS 常唔出聲）
    }
)
_ROMAN_NUMERAL_SPOKEN = {
    "I": "一",
    "II": "二",
    "III": "三",
    "IV": "四",
    "V": "五",
    "VI": "六",
}
# 英文詞／縮寫＋可選尾隨數字（AIRSIDE、KTR350、apm）
_LATIN_SPEECH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\+[A-Za-z]*)?\d*|\d+[A-Za-z]+")
# 名稱後括號英文別名，例如「元政路停車場 (Delta Parking)」
_PAREN_LATIN_ALIAS_RE = re.compile(
    r"\s*[（(][^）)]*[A-Za-z][^）)]*[）)]"
)


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
    if upper in LATIN_BRAND_SPOKEN:
        spoken = LATIN_BRAND_SPOKEN[upper]
    elif upper in _ROMAN_NUMERAL_SPOKEN:
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
    """朗讀前處理：去掉括號英文別名；羅馬數字；罕用字；日期口語。唔改書面顯示。"""
    text = (text or "").strip()
    if not text:
        return text
    text = text.translate(TTS_CHAR_SPOKEN)
    # 間隔號「·」TTS 常停頓／吞字 → 改成輕頓
    text = text.replace("·", "、").replace("•", "、")
    text = re.sub(r"\bM\+", "M plus", text, flags=re.IGNORECASE)
    # 「2026年10月27日（二）」→「…星期二」；否則 Edge TTS 只讀「二」，聽落似月份／亂碼
    text = re.sub(r"[（(]([一二三四五六日])[）)]", r"星期\1", text)
    # 「元政路停車場 (Delta Parking)」→ 唔讀英文別名
    text = _PAREN_LATIN_ALIAS_RE.sub("", text)
    # 「較易泊／易泊」單字「易」粵語 TTS 常讀成 yik（改）而非 yi（容易）
    # → 改寫成「容易」逼正確讀音（只影響朗讀）
    text = text.replace("較易泊車", "比較容易泊車")
    text = text.replace("較易泊", "比較容易泊")
    text = text.replace("好易泊", "好容易泊")
    text = re.sub(r"(?<!容)易泊", "容易泊", text)
    text = re.sub(r"比較比較容易", "比較容易", text)
    return _LATIN_SPEECH_TOKEN_RE.sub(
        lambda m: _latin_token_to_hk_speech(m.group(0)), text
    )


def _suggestion_rank_groups(
    result_df: pd.DataFrame,
    max_ranks: int = SPEAK_DEFAULT_TOP_N,
) -> list[tuple[str, list[str], int]]:
    """最多前 max_ranks 名（含同分）→ [(名次, 名稱列表, 預測空位), ...]。"""
    if result_df is None or result_df.empty:
        return []
    try:
        limit = int(max_ranks)
    except (TypeError, ValueError):
        limit = SPEAK_DEFAULT_TOP_N
    if limit < 1:
        limit = SPEAK_DEFAULT_TOP_N
    work = result_df
    rank_col = ""
    # 地標置頂後可能未有／弄亂名次欄 → 自動補密集聚名，避免 iloc 爆
    if rank_col not in work.columns or work[rank_col].isna().all():
        if "預測空位" not in work.columns:
            return []
        work = add_dense_ranks(
            work.drop(columns=[rank_col], errors="ignore").copy()
        )
    unique_ranks: list = []
    for rank in work[rank_col].tolist():
        if rank is None or (isinstance(rank, float) and pd.isna(rank)):
            continue
        if str(rank).strip() == "":
            continue
        if rank not in unique_ranks:
            unique_ranks.append(rank)
    selected_ranks = unique_ranks[:limit]
    groups: list[tuple[str, list[str], int]] = []
    for rank in selected_ranks:
        group = work[work[rank_col] == rank]
        if group.empty or "名稱" not in group.columns or "預測空位" not in group.columns:
            continue
        names = [str(n) for n in group["名稱"].tolist()]
        if not names:
            continue
        try:
            vacancy = int(group.iloc[0]["預測空位"])
        except (TypeError, ValueError, IndexError):
            continue
        groups.append((str(rank), names, vacancy))
    return groups


def _hour_spoken_yue_12(hour_12: int) -> str:
    """12 小時制鐘點 → 粵語口語（2 用「兩」唔用「二」）。"""
    h = int(hour_12)
    if h == 12:
        return "十二"
    if h == 11:
        return "十一"
    if h == 10:
        return "十"
    return NUM_CN.get(h, str(h))


def format_time_spoken_yue(hour: int, minute: int = 0) -> str:
    """口語時間，例如 14:30 → 下晝兩點半。"""
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
    h_spoken = _hour_spoken_yue_12(h12)
    if minute == 0:
        return f"{period}{h_spoken}點"
    return f"{period}{h_spoken}點半"


def format_date_spoken_yue(date_part: datetime) -> str:
    """口語日期，例如 2026年11月6日星期五。"""
    wd = WEEKDAY_SHORT[date_part.weekday()]
    return f"{date_part.year}年{date_part.month}月{date_part.day}日星期{wd}"


def format_query_when_spoken_yue(
    date_part: datetime | None,
    hour: int,
    minute: int = 0,
) -> str:
    """朗讀用「邊年邊月邊日＋咩時間」；缺日期則只讀鐘點。"""
    time_spoken = format_time_spoken_yue(hour, minute)
    if date_part is None:
        return time_spoken
    return f"{format_date_spoken_yue(date_part)}{time_spoken}"


def _join_names_spoken_yue(names: list[str]) -> str:
    """口語連接停車場名稱。"""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}同埋{names[1]}"
    return "、".join(names[:-1]) + f"，同埋{names[-1]}"


def format_top_suggestion(
    result_df: pd.DataFrame,
    hour: int,
    minute: int = 0,
    *,
    max_ranks: int = SPEAK_DEFAULT_TOP_N,
) -> str:
    """依最多前 max_ranks 名（各名次含全部分同分）產生建議文案。"""
    groups = _suggestion_rank_groups(result_df, max_ranks=max_ranks)
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


def format_spoken_suggestion(
    result_df: pd.DataFrame,
    hour: int,
    minute: int = 0,
    *,
    max_ranks: int = SPEAK_DEFAULT_TOP_N,
    query_date: datetime | None = None,
) -> str:
    """粵語口語建議（畀司機聽）。開頭先講年月日同時間；未指定數量時預設讀頭 SPEAK_DEFAULT_TOP_N 名。"""
    groups = _suggestion_rank_groups(result_df, max_ranks=max_ranks)
    if not groups:
        return ""

    when = format_query_when_spoken_yue(query_date, hour, minute)
    total = sum(len(names) for _, names, _ in groups)

    if total == 1:
        _, names, vacancy = groups[0]
        raw = (
            f"{when}。建議你去{names[0]}，"
            f"預計大約有{vacancy}個空位。"
        )
        return prepare_spoken_text_for_tts(raw)

    parts = [f"{when}。建議你優先考慮以下停車場。"]
    for rank, names, vacancy in groups:
        rank_yue = _rank_spoken_yue(rank)
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


# 擋查提示用固定講稿（書面仍顯示具體無效值），令 MP3 cache 可跨查詢共用，
# 避免「76點／96點半」每次都冷啟動 Edge TTS → 手機第一次自動播失敗。
SPOKEN_INVALID_CALENDAR_DATE = (
    "你輸入嘅日期並唔存在。"
    "請重新輸入查詢內容，例如十一月三十號朝早六點半彩雲邨邊度較易泊。"
)
SPOKEN_INVALID_CLOCK_TIME = (
    "你輸入嘅時間並唔存在。"
    "請重新輸入查詢內容，例如元朗後日上晝八點半邊度較易泊。"
)
SPOKEN_NON_HALF_HOUR = (
    "咁樣係查詢唔到。必須用半小時為單位嘅方式嚟查問，"
    "例如六點半、十二點、四點半、十一點。"
    "唔接受四點九、四點四十五分、九點二十七分呢類時間。請重新輸入。"
)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _synthesize_zh_hk_mp3_cached(text: str) -> bytes:
    """只快取成功嘅 MP3。失敗要 raise，避免把 None／空音訊鎖死 6 小時。

    show_spinner=False：避免「正在產生語音」spinner 觸發前端重繪，打斷 iOS keepalive。
    """
    import edge_tts

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

    audio = asyncio.run(_run())
    if not audio:
        raise RuntimeError("edge_tts returned empty audio")
    return audio


_FEEDBACK_MP3_BY_SPOKEN = {
    SPOKEN_INVALID_CALENDAR_DATE: "invalid_calendar_date.mp3",
    SPOKEN_INVALID_CLOCK_TIME: "invalid_clock_time.mp3",
    SPOKEN_NON_HALF_HOUR: "non_half_hour.mp3",
}


def _load_bundled_feedback_mp3(text: str) -> bytes | None:
    """擋查固定講稿：讀預製 MP3，唔經 Edge TTS（第一次自動播唔使等網絡）。"""
    fname = _FEEDBACK_MP3_BY_SPOKEN.get((text or "").strip())
    if not fname:
        return None
    path = Path(__file__).resolve().parent / "assets" / "tts" / fname
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data or None


def synthesize_zh_hk_mp3(text: str) -> bytes | None:
    """用 Edge 粵語 Neural 聲線合成 MP3（手機 HTML5 Audio 播放較 speechSynthesis 可靠）。

    注意：唔好把失敗結果寫入 cache_data。否則某一句（例如「下晝四點…」）
    第一次合成失敗後，之後幾小時自動朗讀都會無 MP3，手機只能手動撳播放。
    """
    text = (text or "").strip()
    if not text:
        return None
    # 避免極長稿拖垮 Cloud／base64
    if len(text) > 800:
        text = text[:800]

    bundled = _load_bundled_feedback_mp3(text)
    if bundled:
        return bundled

    try:
        import edge_tts  # noqa: F401
    except ImportError:
        return None

    # 短暫網路／冷啟動失敗時再試一次（成功結果仍由 cache 加速）
    for _ in range(2):
        try:
            return _synthesize_zh_hk_mp3_cached(text)
        except Exception:
            continue
    return None


def render_iframe_speech_player(
    spoken: str, *, auto_speak: bool = False, voice_auto: bool = False
) -> None:
    """iframe 內建播放掣：優先播伺服器 MP3（iOS 可用），否則退回 Web Speech。

    手動撳掣：同一 iframe 內同步 play()／speak()。
    自動播（每個新結果第一次）：用 parent 喺「查詢」手勢解鎖過嘅
    AudioContext／Audio；同一段講稿重聽要撳播放掣。
    voice_auto=True（語音送出）：拉長 parent flush／重試，提高第一次自動播成功率。
    """
    spoken = prepare_spoken_text_for_tts((spoken or "").strip())
    if not spoken:
        return

    mp3 = synthesize_zh_hk_mp3(spoken)
    payload = json.dumps(spoken, ensure_ascii=False)
    auto_js = "true" if auto_speak else "false"
    voice_js = "true" if voice_auto else "false"
    # 每次自動播都升 nonce，逼 Streamlit 重建 iframe；否則同一段講稿 HTML 不變 → 腳本唔跑 → 無聲
    if auto_speak:
        play_nonce = int(st.session_state.get("_speech_player_nonce", 0)) + 1
        st.session_state._speech_player_nonce = play_nonce
    else:
        play_nonce = int(st.session_state.get("_speech_player_nonce", 0))
    nonce = f"{play_nonce}-{abs(hash(spoken)) % (10**9)}"

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
  var voiceAuto = {voice_js};
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

  function parentMediaPlaying() {{
    try {{
      var parent = window.parent;
      if (parent && typeof parent.__hkParkingIsUnlockedMediaPlaying === "function") {{
        return !!parent.__hkParkingIsUnlockedMediaPlaying();
      }}
    }} catch (e) {{}}
    return false;
  }}

  var handedToParent = false;
  var parentPlayFailed = false;

  function tryLocalAutoSpeak() {{
    if (uiSpeaking || parentMediaPlaying()) {{
      if (parentMediaPlaying()) setUi(true);
      return;
    }}
    // 已交俾 parent 播就唔好再 speakNow()：speakNow 會 stopParentMedia，切斷「你想…」
    if (handedToParent && !parentPlayFailed) {{
      if (parentMediaPlaying()) setUi(true);
      return;
    }}
    speakNow();
  }}

  function tryAutoSpeak() {{
    if (uiSpeaking) return;
    var parent = window.parent;
    // 優先：排隊畀 parent；未 MediaUnlocked 會留喺 queue，等 unlock flush
    if (audioSrc && parent && (
      typeof parent.__hkParkingQueueUnlockedMedia === "function"
      || typeof parent.__hkParkingPlayUnlockedMedia === "function"
    )) {{
      var started = false;
      var cbs = {{
        onStart: function () {{
          started = true;
          setUi(true);
        }},
        onEnded: function () {{ setUi(false); }},
        onError: function () {{
          parentPlayFailed = true;
          // 只有確認 parent 失敗先本地 fallback（唔好同 parent 搶播）
          if (!started) tryLocalAutoSpeak();
          else setUi(false);
        }}
      }};
      handedToParent = true;
      parentPlayFailed = false;
      if (typeof parent.__hkParkingQueueUnlockedMedia === "function") {{
        parent.__hkParkingQueueUnlockedMedia(audioSrc, cbs);
      }} else {{
        parent.__hkParkingPlayUnlockedMedia(audioSrc, cbs);
      }}
      // 語音查詢：拉長重試；一般查詢維持較短
      var voiceArmed = !!voiceAuto;
      try {{
        if (parent.__hkParkingIsVoiceQueryArmed && parent.__hkParkingIsVoiceQueryArmed()) {{
          voiceArmed = true;
        }}
      }} catch (e) {{}}
      var delays = voiceArmed
        ? [500, 1200, 2200, 4000, 7000, 11000, 16000, 22000]
        : [700, 1400, 2200, 4000, 7000];
      var lastDelay = delays[delays.length - 1];
      delays.forEach(function (ms) {{
        setTimeout(function () {{
          if (started || parentPlayFailed || uiSpeaking || parentMediaPlaying()) {{
            if (!started && parentMediaPlaying()) {{
              started = true;
              setUi(true);
            }}
            return;
          }}
          // 只 revive／flush；中途唔 tryLocalAutoSpeak（避免 stopParent 切斷已開始嘅朗讀）
          try {{
            if (voiceArmed && parent.__hkParkingReviveKeepalive) {{
              parent.__hkParkingReviveKeepalive();
            }}
          }} catch (e) {{}}
          try {{
            if (parent.__hkParkingFlushPendingUnlockedMedia) {{
              parent.__hkParkingFlushPendingUnlockedMedia();
            }}
          }} catch (e) {{}}
          // 最後一檔仍未起播：unlock 已失效時要退回本地／Web Speech，否則永遠無聲
          if (ms === lastDelay) {{
            setTimeout(function () {{
              if (started || uiSpeaking || parentMediaPlaying()) {{
                if (!started && parentMediaPlaying()) {{
                  started = true;
                  setUi(true);
                }}
                return;
              }}
              parentPlayFailed = true;
              tryLocalAutoSpeak();
            }}, 600);
          }}
        }}, ms);
      }});
      return;
    }}
    // 無 MP3／無 parent API：退回 Web Speech／本地播
    tryLocalAutoSpeak();
  }}

  if (wantAuto) {{
    setTimeout(tryAutoSpeak, voiceAuto ? 30 : 50);
  }}
}})();
</script>
</body>
</html>
        """,
        height=48,
    )


def inject_voice_auto_speak_prime() -> None:
    """語音送出後、結果渲染前：延長 armed 視窗並嘗試救返 silent keepalive。"""
    components.html(
        """
<script>
(function () {
  var root = window.parent;
  try {
    if (typeof root.__hkParkingMarkVoiceQueryArmed === "function") {
      root.__hkParkingMarkVoiceQueryArmed();
    } else {
      root.__hkParkingVoiceQueryArmed = true;
      root.__hkParkingVoiceQueryArmedUntil = Date.now() + 60000;
    }
  } catch (e) {}
  try {
    if (typeof root.__hkParkingReviveKeepalive === "function") {
      root.__hkParkingReviveKeepalive();
    }
  } catch (e) {}
  // 只喺「尚未真正朗讀」時 flush／revive；朗讀中唔好改 src
  // 涵蓋 Edge TTS 冷啟動（可十秒級），唔好喺 MP3 就緒前就放棄 keepalive
  [400, 1200, 2500, 5000, 9000, 14000, 20000].forEach(function (ms) {
    root.setTimeout(function () {
      try {
        if (root.__hkParkingMediaPlayPending) return;
        var a = root.__hkParkingMediaAudio;
        if (a && !root.__hkParkingMediaKeepalive && !a.paused && !a.ended) return;
      } catch (e) {}
      try {
        if (root.__hkParkingReviveKeepalive) root.__hkParkingReviveKeepalive();
      } catch (e) {}
      try {
        if (root.__hkParkingFlushPendingUnlockedMedia) {
          root.__hkParkingFlushPendingUnlockedMedia();
        }
      } catch (e) {}
    }, ms);
  });
})();
</script>
        """,
        height=1,
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

    voice_armed = bool(st.session_state.get("_voice_auto_speak_this_query"))

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
        render_iframe_speech_player(
            spoken,
            auto_speak=should_auto_speak,
            voice_auto=voice_armed and should_auto_speak,
        )

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


def render_nearby_live_slots() -> None:
    """未搜尋前：顯示用戶附近即時空位（政府 API + 本地車場座標）。"""
    coords = st.session_state.get("user_geo_coords")
    if not coords:
        st.info(
            "開始查詢前，可先允許定位以查看附近即時空位；"
            "或直接輸入／語音查詢未來時段預測。"
        )
        st.caption(
            "資料庫約每 10 分鐘自動採集運輸署開放數據。"
        )
        return

    lat, lon = coords
    with st.spinner("正在載入附近即時空位…"):
        nearby, radius = carparks_near_coords(float(lat), float(lon))
        live = load_live_private_car_vacancies()

    if nearby.empty:
        st.info("附近暫時找不到本系統有紀錄的停車場。請直接輸入查詢。")
        return

    rows = []
    for _, park in nearby.iterrows():
        pid = str(park["park_id"])
        live_row = live.get(pid)
        if not live_row:
            continue
        rows.append(
            {
                "park_id": pid,
                "名稱": park["name"],
                "地區": park.get("sub_district") or park.get("district"),
                "距離(公里)": round(float(park["distance_km"]), 1),
                "即時空位": int(live_row["vacancy"]),
                "latitude": park.get("latitude"),
                "longitude": park.get("longitude"),
            }
        )

    if not rows:
        st.info("已定位附近車場，但暫時未能取得即時空位。請直接輸入查詢。")
        return

    live_df = pd.DataFrame(rows).sort_values(
        ["即時空位", "距離(公里)"], ascending=[False, True], kind="mergesort"
    )
    live_df["name"] = live_df["名稱"]
    live_df = dedupe_equivalent_carparks(live_df, prefer_higher_col="即時空位")
    live_df = live_df.drop(columns=["name"], errors="ignore")
    live_df = add_dense_ranks(live_df, value_col="即時空位").head(LIVE_NEARBY_TOP_N)
    preferred = ["", "名稱", "地區", "距離(公里)", "即時空位"]
    ordered = [c for c in preferred if c in live_df.columns]
    ordered += [c for c in live_df.columns if c not in ordered]
    live_df = live_df[ordered]

    radius_txt = f"{radius:.1f} km" if radius else "附近"
    st.success(f"你附近（約 {radius_txt}）即時空位較多的停車場")
    st.caption(
        "以下為開頁預設：政府即時數據。"
        "輸入時間／地點可查預測；資料庫約每 10 分鐘自動更新歷史。"
    )
    render_ranked_table(live_df)
    render_results_map(
        live_df,
        intent={
            "geo_lat": lat,
            "geo_lon": lon,
            "geo_label": "你目前位置",
        },
    )


def render_day_profile_chart(profile_df: pd.DataFrame, *, day: str, half_hour: int) -> None:
    """顯示前列車場全日預測曲線，方便判斷信心／走勢。"""
    if profile_df is None or profile_df.empty:
        return
    import altair as alt

    st.markdown(f"**{day} 全日預測空位走勢（前列車場）**")
    highlight_time = f"{half_hour // 2:02d}:{30 if half_hour % 2 else 0:02d}"
    chart = (
        alt.Chart(profile_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("時間:N", sort=None, title="時間"),
            y=alt.Y("預測空位:Q", title="預測空位"),
            color=alt.Color("名稱:N", title="停車場"),
            tooltip=["名稱", "時間", "預測空位"],
        )
        .properties(height=260)
    )
    rule = (
        alt.Chart(pd.DataFrame({"時間": [highlight_time]}))
        .mark_rule(strokeDash=[4, 4], color="#888")
        .encode(x=alt.X("時間:N", sort=None))
    )
    st.altair_chart(chart + rule, use_container_width=True)


def render_saved_prediction(payload: dict, *, auto_speak: bool = False) -> None:
    """重顯示已儲存的預測結果（例如撳「再聽一次」後）。"""
    intent = payload.get("intent") or {}
    result_df = pd.DataFrame(payload.get("result_records") or [])
    profile_df = pd.DataFrame(payload.get("profile_records") or [])
    hour = int(payload["hour"])
    minute = int(payload["minute"])
    day = payload["day"]
    district_label = payload["district_label"]
    top_n = int(payload["top_n"])
    shown = int(payload["shown"])
    asking_now = bool(payload.get("asking_now"))
    top_n_default = bool(payload.get("top_n_default"))
    place_ease = bool(payload.get("place_ease"))
    top_n_shortfall = bool(payload.get("top_n_shortfall"))
    requested_top_n = int(payload.get("requested_top_n") or top_n)
    available_count = int(payload.get("available_count") or shown)
    time_str = format_time_12h(hour, minute)
    half_hour = to_half_hour_slot(hour, minute)

    render_intent_summary(
        intent,
        resolved_top_n=requested_top_n if top_n_shortfall else top_n,
        top_n_default=top_n_default,
    )
    if asking_now:
        st.info(
            f"我們的預測系統以半小時為單位。"
            f"你查詢的是「而家／現在」，因此顯示未來 1–30 分鐘內"
            f"下一個半小時（{time_str}）的預測結果。"
        )
    if top_n_shortfall:
        st.warning(
            format_top_n_shortfall_notice(
                requested_top_n,
                available_count,
                district_label,
                style="written",
            )
        )
    if place_ease:
        st.caption("你問的是該地點本身容唔容易泊，因此只顯示該場所停車場。")
    elif top_n_default:
        st.caption(f"未指定顯示數量 → 預設顯示前 {top_n} 名")
    if place_ease:
        ease = ""
        if not result_df.empty and "預測空位" in result_df.columns:
            try:
                best = float(pd.to_numeric(result_df["預測空位"], errors="coerce").max())
                ease = f"（整體：{vacancy_ease_label(best)}）"
            except (TypeError, ValueError):
                ease = ""
        st.success(
            f"{day} {time_str}「{district_label}」停車場空位預測"
            f"（共 {shown} 個）{ease}"
        )
    elif top_n_shortfall:
        st.success(
            f"{day} {time_str} {district_label} 較易泊車的停車場"
            f"（共 {shown} 個）"
        )
    elif shown > top_n:
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
    render_day_profile_chart(profile_df, day=day, half_hour=half_hour)
    st.caption(
        "資料來源：運輸署開放數據，約每 10 分鐘自動採集寫入資料庫（非人手更新）。"
    )
    flush_deferred_speech_browser_ops()
    if need_rerun:
        st.rerun()


_TABLE_HIDDEN_COLS = frozenset(
    {"latitude", "longitude", "park_id", "樣本數", "依據", "信心"}
)


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

    def _vacancy_color(v) -> str:
        try:
            n = float(v)
        except (TypeError, ValueError):
            return "#9aa0a6"
        if n >= 30:
            return "#34a853"  # 充足
        if n >= 10:
            return "#f9ab00"  # 一般
        return "#ea4335"  # 緊張

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
        vacancy_num = None
        vacancy_col = (
            "預測空位"
            if "預測空位" in df.columns
            else ("即時空位" if "即時空位" in df.columns else None)
        )
        if vacancy_col:
            vacancy = row.get(vacancy_col)
            try:
                vacancy_num = int(vacancy) if vacancy is not None and not pd.isna(vacancy) else None
                vacancy_txt = str(vacancy_num) if vacancy_num is not None else "—"
            except (TypeError, ValueError):
                vacancy_txt = str(vacancy) if vacancy is not None else "—"
            details.append(f"{vacancy_col}：{html.escape(vacancy_txt)}")
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
        # 無空位數時用藍色「P」，避免灰點喺地圖上幾乎睇唔到
        if vacancy_num is not None:
            label = str(vacancy_num)
            color = _vacancy_color(vacancy_num)
        else:
            label = "P"
            color = "#4285F4"
        markers.append(
            {
                "lat": lat,
                "lng": lon,
                "title": name,
                "color": color,
                "label": label,
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
    # Pin 顏色：綠≥30／橙10–29／紅<10；圓內數字＝空位數。
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
    #map-legend {{
      position: absolute; left: 10px; bottom: 10px; z-index: 1000;
      background: rgba(255,255,255,0.92); color: #222;
      padding: 6px 10px; border-radius: 8px; font: 12px/1.35 sans-serif;
      box-shadow: 0 1px 4px rgba(0,0,0,0.18);
    }}
    #map-legend span.dot {{
      display: inline-block; width: 10px; height: 10px; border-radius: 50%;
      margin-right: 4px; vertical-align: -1px; border: 1px solid #fff;
    }}
    .mzd-pin-icon {{
      background: transparent; border: none;
    }}
    .mzd-pin {{
      width: 28px; height: 28px; border-radius: 50%;
      border: 2px solid #fff; color: #fff; font: 700 11px/24px sans-serif;
      text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.35);
    }}
  </style>
</head>
<body style="position:relative">
  <div id="map"></div>
  <div id="map-legend">
    <div><span class="dot" style="background:#34a853"></span>空位充足（≥30）</div>
    <div><span class="dot" style="background:#f9ab00"></span>一般（10–29）</div>
    <div><span class="dot" style="background:#ea4335"></span>緊張（&lt;10）</div>
    <div><span class="dot" style="background:#4285F4"></span>位置（無空位數）</div>
    <div style="margin-top:2px;opacity:0.75">圓內數字＝空位數；P＝只有位置</div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script>
    const MARKERS = {markers_json};
    const CENTER = {center_json};
    let usedFallback = false;

    function pinSvg(color, label) {{
      const text = String(label == null ? "·" : label);
      const fontSize = text.length > 2 ? 9 : 11;
      const svg =
        '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28">' +
        '<circle cx="14" cy="14" r="12" fill="' + color + '" stroke="#fff" stroke-width="2"/>' +
        '<text x="14" y="18" text-anchor="middle" fill="#fff" font-size="' + fontSize +
        '" font-family="sans-serif" font-weight="700">' + text + '</text></svg>';
      return "data:image/svg+xml;charset=UTF-8," + encodeURIComponent(svg);
    }}

    function initLeafletFallback() {{
      if (usedFallback) return;
      usedFallback = true;
      const el = document.getElementById("map");
      el.innerHTML = "";
      const map = L.map(el, {{ scrollWheelZoom: true }});
      L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap",
      }}).addTo(map);

      const bounds = [];
      MARKERS.forEach(function (m) {{
        const icon = L.divIcon({{
          className: "mzd-pin-icon",
          html: '<div class="mzd-pin" style="background:' + (m.color || "#9aa0a6") +
                '">' + (m.label || "·") + '</div>',
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        }});
        const marker = L.marker([m.lat, m.lng], {{ icon: icon }}).addTo(map);
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
            title: m.title + (m.label ? (" · " + m.label) : ""),
            icon: {{
              url: pinSvg(m.color || "#9aa0a6", m.label),
              scaledSize: new google.maps.Size(28, 28),
              anchor: new google.maps.Point(14, 14),
            }},
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
    st.caption("地圖標記：顏色＝空位鬆緊，數字＝預測／即時空位。")
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



# 「下星期二／下週三／下個星期一」等 → 下一個日曆週嘅該日
_NEXT_WEEK_DAY_RE = re.compile(r"下個?(?:星期|週|周)([一二三四五六日天])")
_DAY_CHAR_TO_CN = {
    "一": "週一",
    "二": "週二",
    "三": "週三",
    "四": "週四",
    "五": "週五",
    "六": "週六",
    "日": "週日",
    "天": "週日",
}


def query_implies_next_week_weekday(query: str) -> bool:
    """查詢是否寫明「下星期／下週／下個星期」＋星期。"""
    q = (query or "").replace(" ", "")
    return bool(_NEXT_WEEK_DAY_RE.search(q))


def resolve_relative_day(query: str, today=None):
    """從查詢解析相對日期（粵語），回傳「週一」…「週日」；無法判斷則回傳 None。

    過去相對日子（琴日／前日等）亦會回傳對應星期；實際日曆日應另以
    parse_past_relative_calendar_date／absolute_date 鎖定，避免被推去下一週。
    """
    today = today or hong_kong_now()
    q = query.replace(" ", "")

    # 「下星期二」等：先抽星期（實際日期由 day_next_week＋date_for_weekday_cn 處理）
    m = _NEXT_WEEK_DAY_RE.search(q)
    if m:
        return _DAY_CHAR_TO_CN.get(m.group(1))

    offset = relative_day_offset_from_query(q)
    if offset is not None:
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


def date_for_weekday_cn(
    day_cn: str, today=None, *, next_week: bool = False
) -> datetime | None:
    """將「週一」…「週日」轉成日期。

    - 預設：今天或之後最近的該日（含今日）
    - next_week=True（「下星期二」等）：下一個日曆週的該日（由下星期一起計）
    """
    today = today or hong_kong_now()
    target = DAY_TO_NUM.get(day_cn)
    if target is None:
        return None
    if next_week:
        # Mon→+7、Tue→+6 … Sun→+1，再到該星期幾
        days_to_next_monday = 7 - today.weekday()
        next_monday = today + timedelta(days=days_to_next_monday)
        return (next_monday + timedelta(days=target)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    delta = (target - today.weekday()) % 7
    return (today + timedelta(days=delta)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def parse_absolute_date_from_query(query: str, today=None) -> datetime | None:
    """解析絕對日期（如「11月29號」「2026年11月29日」「11/29」）。

    無年份時：今個月／上個月用今年；其餘更早月份用明年；今年較後月份仍用今年。
    今個月內已過日子保留為過去（由後續流程提示重新查詢）。
    """
    today = (today or hong_kong_now()).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    q = (query or "").replace(" ", "")
    if not q:
        return None

    year = month = day = None
    m = re.search(
        r"(?:(?P<year>\d{4})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})[號号日]?",
        q,
    )
    if m:
        year = int(m.group("year")) if m.group("year") else None
        month = int(m.group("month"))
        day = int(m.group("day"))
    else:
        m2 = re.search(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})(?!\d)", q)
        if not m2:
            return None
        month, day = int(m2.group(1)), int(m2.group(2))

    return resolve_yearless_calendar_date(
        month, day, today=today, year=year
    )


def absolute_date_from_intent(intent: dict | None, today=None) -> datetime | None:
    """由 intent['absolute_date']（YYYY-MM-DD）還原日期。"""
    if not intent:
        return None
    raw = intent.get("absolute_date")
    if raw is None or str(raw).strip() in ("", "null", "None"):
        return None
    today = today or hong_kong_now()
    try:
        dt = datetime.strptime(str(raw).strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return dt.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=today.tzinfo
    )


def resolve_intent_calendar_date(
    day_cn: str | None = None,
    today=None,
    *,
    next_week: bool = False,
    absolute_date=None,
    intent: dict | None = None,
) -> datetime | None:
    """優先用絕對日期，否則用星期推算。"""
    today = today or hong_kong_now()
    if intent is not None and absolute_date is None:
        absolute_date = intent.get("absolute_date")
    if absolute_date is not None and str(absolute_date).strip() not in (
        "",
        "null",
        "None",
    ):
        if isinstance(absolute_date, datetime):
            return absolute_date.replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=today.tzinfo
            )
        try:
            dt = datetime.strptime(str(absolute_date).strip()[:10], "%Y-%m-%d")
            return dt.replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=today.tzinfo
            )
        except ValueError:
            pass
    if intent is not None and day_cn is None:
        day_cn = intent.get("day")
        next_week = bool(intent.get("day_next_week"))
    if not day_cn:
        return None
    return date_for_weekday_cn(day_cn, today, next_week=next_week)


def is_future_day(
    day_cn: str,
    today=None,
    *,
    next_week: bool = False,
    absolute_date=None,
) -> bool:
    """日子是否為明天或之後（相對今天）。"""
    today = today or hong_kong_now()
    date_part = resolve_intent_calendar_date(
        day_cn, today, next_week=next_week, absolute_date=absolute_date
    )
    if date_part is None:
        return False
    return date_part.date() > today.date()


def intent_has_clock_time(hour) -> bool:
    """意圖是否已有具體小時。"""
    return hour is not None and str(hour).strip() not in ("", "null", "None")


def intent_query_datetime(
    day_cn: str,
    hour,
    minute,
    today=None,
    *,
    next_week: bool = False,
    absolute_date=None,
) -> datetime | None:
    """將意圖的日期／星期＋鐘點組合成具體日期時間；缺資料則回傳 None。"""
    if not intent_has_clock_time(hour):
        return None
    today = today or hong_kong_now()
    date_part = resolve_intent_calendar_date(
        day_cn, today, next_week=next_week, absolute_date=absolute_date
    )
    if date_part is None:
        return None
    m = 0 if minute is None or str(minute).strip() in ("", "null", "None") else int(minute)
    h, m = normalize_half_hour(int(hour), m)
    return date_part.replace(hour=h, minute=m, second=0, microsecond=0)


def is_past_intent_calendar_date(
    day_cn: str | None = None,
    today=None,
    *,
    next_week: bool = False,
    absolute_date=None,
    intent: dict | None = None,
) -> bool:
    """意圖所指日曆日是否早於今天（唔使有鐘點）。"""
    today = today or hong_kong_now()
    date_part = resolve_intent_calendar_date(
        day_cn,
        today,
        next_week=next_week,
        absolute_date=absolute_date,
        intent=intent,
    )
    if date_part is None:
        return False
    return date_part.date() < today.date()


def is_past_intent_datetime(
    day_cn: str,
    hour,
    minute,
    today=None,
    *,
    next_week: bool = False,
    absolute_date=None,
) -> bool:
    """意圖所指日期時間是否已過去（相對香港時間）。"""
    today = today or hong_kong_now()
    if is_past_intent_calendar_date(
        day_cn,
        today,
        next_week=next_week,
        absolute_date=absolute_date,
    ):
        return True
    when = intent_query_datetime(
        day_cn,
        hour,
        minute,
        today,
        next_week=next_week,
        absolute_date=absolute_date,
    )
    if when is None:
        return False
    return when < today.replace(second=0, microsecond=0)


def format_time_ampm(hour: int, minute: int = 0) -> str:
    """例如：9:30am、4:30pm"""
    hour12 = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    return f"{hour12}:{minute:02d}{suffix}"


def format_intent_date(
    day_cn,
    today=None,
    *,
    next_week: bool = False,
    absolute_date=None,
) -> str:
    """例如：2026年10月24日（六）"""
    today = today or hong_kong_now()
    date_part = resolve_intent_calendar_date(
        day_cn, today, next_week=next_week, absolute_date=absolute_date
    )
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
        "好泊",
        "難泊",
        "易拍",  # 語音／錯字常見：泊→拍
        "好拍",
        "比較易",
        "較易",
        "鬆啲",
        "鬆位",
        "易唔易泊",
        "易唔易拍",
        "容唔容易",
        "容不容易",
        "用唔容易",  # 語音常誤「容」→「用」（normalize_asr_query 亦會校正）
        "用不容易",
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
    if any(h in q for h in hints):
        return True
    # 「星期日上午10:30邊啲停車場…」＝問邊度較易泊，唔係純名單
    if looks_like_fee_query(query):
        return False
    if not looks_like_parking_query(query):
        return False
    if not (
        has_explicit_clock_time(query)
        or resolve_relative_day(query) is not None
    ):
        return False
    return bool(
        re.search(
            r"邊[度啲個]|哪[裡里些個]|比較|推薦|建議|易[泊拍]|好[泊拍]",
            q,
        )
    )


def looks_like_place_ease_query(query: str) -> bool:
    """問某地點本身容唔容易泊（而非「附近邊啲較易泊」排名）。

    通用規則（商場／邨／地標皆同）：
    - 「YOHO mall／APM／彩雲邨…容唔容易／易唔易／有冇位」→ True
    - 「天水圍今晚八點邊度較易泊」→ False（有邊度＝附近排名）
    """
    q = re.sub(r"\s+", "", query or "")
    if not q:
        return False
    # 「附近／邊度／邊啲／推薦」→ 附近排名（唔當單一場所）
    if re.search(r"附近|邊[度啲個]|哪[裡里些個]|推薦|建議邊", q):
        return False
    ease_hints = (
        "容唔容易",
        "容不容易",
        "容不容易泊",
        "用唔容易",  # 語音／打字常誤「容」→「用」
        "用不容易",
        "容易唔容易",
        "容易不容易",
        "易唔易",
        "好唔好泊",
        "好不好泊",
        "難唔難泊",
        "难不难泊",
        "好泊唔好",
        "易泊唔易",
        "泊唔泊到",
        "得唔得泊",
        "泊得到嗎",
        "泊得到吗",
        "有冇位",
        "有沒有位",
        "有没有位",
        "夠唔夠位",
        "够不够位",
        "車位夠唔夠",
        "车位够不够",
        "鬆唔鬆",
        "挤不挤",
        "擠不擠",
        "會唔會冇位",
        "会不会没位",
        "是否容易泊",
        "容易嗎",
        "容易吗",
    )
    return any(h in q for h in ease_hints)


def _normalize_place_key(place: str) -> str:
    return re.sub(r"\s+", "", (place or "").strip()).lower()


def _place_is_major_area(place: str) -> bool:
    """廣域地區（區／市中心）而非單一商場／邨／地標。"""
    key = _normalize_place_key(place)
    if not key or _is_all_hong_kong(place):
        return True
    major_keys = {_normalize_place_key(x) for x in MAJOR_AREA_PLACE_LABELS}
    if key in major_keys:
        return True
    # 「元朗區」「沙田區」等
    if key.endswith("區") and key[:-1] in major_keys:
        return True
    return False


def _place_match_tokens(place: str) -> list[str]:
    """由用戶地名抽出用以對車場名稱嘅 token（小階／原文）。"""
    raw = (place or "").strip()
    if not raw or _is_all_hong_kong(raw):
        return []
    compact = _normalize_place_key(raw)
    stripped = re.sub(
        r"(mall|商場|廣場|中心|center|centre|parking|carpark|停車場)$",
        "",
        compact,
        flags=re.I,
    )
    tokens: list[str] = []
    if compact:
        tokens.append(compact)
    if stripped and stripped != compact:
        tokens.append(stripped)
    # 英文詞分段（YOHO mall → yoho）
    for eng in re.findall(r"[A-Za-z]{2,}", raw):
        tokens.append(eng.lower())
    for key, aliases in PLACE_NAME_ALIASES.items():
        if key in compact or compact in key or key in stripped:
            tokens.extend(aliases)
    # 中文片段；邨／苑／園等可再加詞幹（彩雲邨→彩雲）
    for part in re.findall(r"[\u4e00-\u9fff]{2,}", raw):
        tokens.append(part)
        if len(part) >= 3 and part[-1] in "邨苑園城灣莊村":
            tokens.append(part[:-1])
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        t = (t or "").strip().lower()
        if len(t) < 2 or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def extract_place_mention_from_query(query: str) -> str | None:
    """由原句盡量抽出地點片段（唔依賴 LLM district，避免被收成「元朗」等區名）。

    例如「YOHO mall聽晚九點半容唔容易泊車」→「YOHO mall」
    """
    q = (query or "").strip()
    if not q:
        return None
    cut = q
    # 去掉相對／絕對日期、時段、鐘點
    cut = re.sub(
        rf"({RELATIVE_DAY_STRIP_RE}|"
        r"下(?:個)?星期[一二三四五六日天]|下(?:個)?週[一二三四五六日]|"
        r"星期[一二三四五六日天]|週[一二三四五六日]|"
        r"下晝|上晝|下午|上午|早上|朝早|夜晚|晚上|傍晚|凌晨|清晨|中午|午晝|深夜)",
        " ",
        cut,
    )
    cut = re.sub(r"(?:\d{4}年)?\d{1,2}月\d{1,2}[號号日]?", " ", cut)
    cut = re.sub(r"(?<!\d)\d{1,2}[/\-.]\d{1,2}(?!\d)", " ", cut)
    cut = re.sub(
        r"\d{1,2}\s*[:：]\s*\d{1,2}|\d{1,2}\s*[點时時]\s*半?|"
        r"[一二兩三两四五六七八九十廿]+\s*[點时時]\s*半?",
        " ",
        cut,
    )
    # 去掉「容唔容易泊…／收費如何」等意圖尾巴
    cut = re.sub(
        r"(容唔容易|容不容易|用唔容易|用不容易|容易唔容易|容易不容易|易唔易|"
        r"好唔好泊|好不好泊|難唔難泊|难不难泊|"
        r"好泊唔好|易泊唔易|泊唔泊到|得唔得泊|有冇位|有沒有位|有没有位|"
        r"夠唔夠位|够不够位|車位夠唔夠|车位够不够|鬆唔鬆|是否容易泊|"
        r"容易嗎|容易吗|會唔會冇位|会不会没位|"
        r"附近|邊[度啲個]|哪[裡里些個]|比較易|較易泊|好泊|難泊|"
        r"收費|价钱|價錢|價格|几钱|幾錢|多少錢|幾多錢|如何|怎麼樣|怎么样|"
        r"停車場|泊車|停車|車位|空位|parking).*$",
        " ",
        cut,
        flags=re.I,
    )
    cut = re.sub(r"\s+", " ", cut).strip(" ，,的嘅喺在")
    if len(cut) >= 2:
        return cut
    return None


def resolve_place_ease_label(query: str, district: str) -> str | None:
    """若應走「該場所容唔容易泊」，回傳篩車場用地名；否則 None。

    優先用原句抽出嘅地名（YOHO mall／APM／彩雲邨…），
    唔好只信 LLM district（常被收窄成元朗／觀塘等廣域名而誤判）。
    """
    if not looks_like_place_ease_query(query):
        return None
    return resolve_venue_focus_label(query, district)


def resolve_venue_focus_label(query: str, district: str) -> str | None:
    """具體商場／邨／地標（非廣域）→ 結果應優先顯示該場所車場。

    適用「德福廣場」「德福廣場今晚收費如何」等；廣域如觀塘／天水圍回傳 None。
    """
    for candidate in (
        extract_place_mention_from_query(query),
        str(district or "").strip() or None,
    ):
        if not candidate or _is_all_hong_kong(candidate):
            continue
        if _place_is_major_area(candidate):
            continue
        return candidate
    return None


def should_use_place_ease_scope(query: str, place: str) -> bool:
    """是否用「該場所車場」範圍（通用：任何具體地點＋容唔容易類問句）。

    廣域地區（如觀塘／天水圍）即使講容唔容易，仍用附近排名，避免只剩地心附近幾場。
    新碼請優先用 resolve_place_ease_label()。
    """
    return resolve_place_ease_label(query, place) is not None


def filter_parks_at_queried_place(
    df: pd.DataFrame,
    place: str,
    *,
    max_km: float = PLACE_AT_VENUE_KM,
    name_match_required: bool = False,
) -> pd.DataFrame:
    """只保留「該場所」車場：名稱吻合，或距離極近（喺地點範圍內）。

    對所有具體地點一視同仁（唔限 YOHO）；別名表只係輔助。
    name_match_required=True：只用名稱／別名（附近排名置頂時避免把隔壁邨當該商場）。
    """
    if df is None or df.empty:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()
    tokens = _place_match_tokens(place)
    name_col = "名稱" if "名稱" in df.columns else ("name" if "name" in df.columns else None)
    dist_col = "距離(公里)" if "距離(公里)" in df.columns else None

    def _name_hit(row) -> bool:
        name = str(row[name_col]) if name_col else ""
        return park_name_matches_place_tokens(
            name, tokens, normalize_name=_normalize_park_name
        )

    def _at_place(row) -> bool:
        if _name_hit(row):
            return True
        if name_match_required:
            return False
        if dist_col is not None:
            try:
                d = float(row[dist_col])
            except (TypeError, ValueError):
                d = None
            if d is not None and d <= max_km:
                return True
        return False

    masked = df[df.apply(_at_place, axis=1)]
    return masked.copy()


def vacancy_ease_label(vacancy) -> str:
    """由預測空位數轉口語鬆緊。"""
    try:
        n = float(vacancy)
    except (TypeError, ValueError):
        return "暫時無法判斷"
    if n >= 30:
        return "較易泊"
    if n >= 10:
        return "一般"
    return "較緊張"


def format_spoken_place_ease(
    result_df: pd.DataFrame,
    place_label: str,
    hour: int,
    minute: int = 0,
    *,
    query_date: datetime | None = None,
) -> str:
    """「某地容唔容易泊」專用粵語稿；開頭先講年月日同時間。"""
    when = format_query_when_spoken_yue(query_date, hour, minute)
    place = (place_label or "該處").strip() or "該處"
    if result_df is None or result_df.empty:
        raw = f"{when}。{place}暫時冇可用嘅空位預測。"
        return prepare_spoken_text_for_tts(raw)

    rows = result_df.head(5)
    parts = [f"{when}。關於{place}泊車。"]
    vacancies = []
    for _, row in rows.iterrows():
        name = str(row.get("名稱") or "停車場")
        vac = row.get("預測空位")
        try:
            vac_i = int(vac)
        except (TypeError, ValueError):
            vac_i = 0
        vacancies.append(vac_i)
        parts.append(f"{name}預計大約有{vac_i}個空位。")
    best = max(vacancies) if vacancies else 0
    parts.append(f"整體嚟講{vacancy_ease_label(best)}。")
    return prepare_spoken_text_for_tts("".join(parts))


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
    # 唔用裸「邊啲停車場」：會誤傷「邊啲停車場比較易泊」
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
  - "prediction"：想知邊度較易泊、某時某地空位、某地容唔容易泊、預測／推薦停車場等（即使沒寫「預測」二字）。「邊啲停車場比較易泊／好泊」屬 prediction，不是 info
  - "info"：與停車場有關，但不是在問空位預測。例如：某區有幾個停車場、係咪得一個、有冇停車場、叫咩名、收費／價錢、邊啲有／冇顯示收費、有收費資料、本系統可以問咩。單純「有邊啲停車場」（無時間、無易泊）才用 info
- district: 地點名稱，會交 Google Maps 定位。優先填用戶提到的具體地名（屋邨、商場、街道、地區，例如「彩雲邨」「栢慧豪園」「YOHO mall」「天水圍」）；沒有提到就填「全港」。related 為 false 時填「全港」。不要改寫成內部代號或加「／」合併名。若用戶問某商場／地標「容唔容易泊／易唔易泊／有冇位」，district 必須保留該具體地名（不要只填所在行政區）
- hour: 小時（0-23）。只在句子有具體鐘點時填數字（如「三點」「15:30」「晚上九點」「上午10:30」）；沒有具體鐘點則填 null（「後晚／聽晚／今晚／嚟緊」單獨出現不算具體鐘點）。問收費但無具體鐘點時填 null（系統會用當刻時間）
- minute: 分鐘，時間以半小時為單位，只允許 0 或 30。規則：
  - 「三點半／3:30／15:30／上午10:30」→ 30
  - 「三點／3:00／15:00」或只提到整點 → 0
  - 其他分鐘請四捨五入到最近的半小時（例如 15→30、45→下一小時的 0）
  - 如果沒有具體鐘點，填 null
  - 「嚟緊12點／嚟緊十二點」若未寫中午／午夜：hour 先填 0（稍後會規則校正為下一次 12 點）
- day: 星期（「週一」到「週日」）。必須依日期換算，今天是{today_day}：
  - 「今日／今晚／今天」→ {day_today}
  - 「聽日／聽晚／明天／明晚」→ {day_tomorrow}
  - 「後日／後晚／後天」→ {day_after}
  - 「琴日／昨日／前日／大前日」等已過去日子 → 仍填對應「週X」（系統會鎖定該過去日曆日並提示重新查詢）
  - 「星期日／週日／星期六」等直接寫星期 → 填對應「週X」
  - 「下星期二／下週三／下個星期一」等 → 填對應「週X」（系統會解成下一個日曆週，不是本週已過或今日）
  - 「11月29號／2026年11月29日」等絕對日期 → 填該日對應「週X」（不要填成今天；系統會再以規則鎖定該日曆日；無年份時今個月／上個月用今年，更早月份用明年）
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

範例（指定星期＋鐘點＋較易泊；勿當 info、勿用而家時間）：
{{"related": true, "query_type": "prediction", "district": "天水圍", "hour": 10, "minute": 30, "day": "週日", "top_n": null}}

範例（絕對日期 11月29號＝該日星期，勿當成今日）：
{{"related": true, "query_type": "prediction", "district": "屯門兆康", "hour": 15, "minute": 30, "day": "週日", "top_n": null}}

範例（屋邨／地標）：
{{"related": true, "query_type": "prediction", "district": "彩雲邨", "hour": 16, "minute": 30, "day": "{day_tomorrow}", "top_n": null}}

範例（某商場本身容唔容易泊；district 保留商場名，勿改成「元朗」）：
{{"related": true, "query_type": "prediction", "district": "YOHO mall", "hour": 10, "minute": 30, "day": "{day_tomorrow}", "top_n": null}}

範例（停車場資料／非預測；句子如「掃管笏有冇停車場」）：
{{"related": true, "query_type": "info", "district": "掃管笏", "hour": null, "minute": null, "day": "{day_today}", "top_n": null}}

範例（問收費、無具體鐘點 → 當刻收費；句子如「天水圍停車場收費如何」）：
{{"related": true, "query_type": "info", "district": "天水圍", "hour": null, "minute": null, "day": "{day_today}", "top_n": null}}

範例（問某日某時收費；句子如「天水圍聽日晚上八點收費」）：
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

    # 規則覆寫 query_type（資料查詢 vs 空位預測；預測關鍵字優先）
    raw_qtype = str(intent.get("query_type") or "").strip().lower()
    if raw_qtype in ("info", "parking_info", "faq", "data"):
        intent["query_type"] = "info"
    elif raw_qtype in ("prediction", "vacancy", "predict"):
        intent["query_type"] = "prediction"
    else:
        intent["query_type"] = "prediction"
    if looks_like_vacancy_prediction_query(query):
        intent["query_type"] = "prediction"
    elif looks_like_parking_info_query(query):
        intent["query_type"] = "info"

    # 規則覆寫日期：絕對日期（11月29號）優先；過去相對日子（琴日等）要鎖定 absolute_date
    abs_dt = parse_absolute_date_from_query(query, today)
    if abs_dt is None:
        abs_dt = parse_past_relative_calendar_date(query, today)
    if abs_dt is not None:
        intent["absolute_date"] = abs_dt.strftime("%Y-%m-%d")
        intent["day"] = WEEKDAY_CN[abs_dt.weekday()]
        intent["day_next_week"] = False
    else:
        intent.pop("absolute_date", None)
        resolved_day = resolve_relative_day(query, today)
        if resolved_day:
            intent["day"] = resolved_day
        intent["day_next_week"] = query_implies_next_week_weekday(query)

    # 保留 LLM 抽出的地點原文予 Google Geocoding；只把「全部」等正規成全港
    if _is_all_hong_kong(intent.get("district")):
        intent["district"] = "全港"

    if is_now_query(query):
        intent["day"] = WEEKDAY_CN[today.weekday()]
        intent["day_next_week"] = False
        intent.pop("absolute_date", None)

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
                extracted = extract_clock_time_from_query(query)
                if extracted is not None:
                    intent["hour"], intent["minute"] = extracted
                else:
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
    next_week = bool(intent.get("day_next_week"))
    future = is_future_day(
        day_cn,
        today,
        next_week=next_week,
        absolute_date=intent.get("absolute_date"),
    )
    has_clock = has_explicit_clock_time(query)
    coming_soon = is_coming_soon_query(query)
    has_absolute = bool(intent.get("absolute_date"))

    # LLM 漏填鐘點時，用規則從句子補回（例如「上午10:30」）
    if has_clock and not intent_has_clock_time(intent.get("hour")):
        extracted = extract_clock_time_from_query(query)
        if extracted is not None:
            intent["hour"], intent["minute"] = extracted

    # 「嚟緊」＋具體鐘點 → 下一次即將到來的該鐘點（可跨日；曖昧 12 點取最近未來）
    if coming_soon and has_clock and not has_absolute:
        intent = resolve_coming_soon_intent(intent, query, today)
    # 明天或之後／已指定絕對日期：未提供具體鐘點則保留 null（稍後提示重新輸入）
    elif (future or has_absolute) and not has_clock:
        intent["hour"] = None
        intent["minute"] = None
    # 「而家／現在／嚟緊」或今日未指明鐘點 → 下一個半小時（可跨日）
    elif is_now_query(query) or coming_soon or not has_clock:
        dt = next_half_hour_datetime(today)
        intent["hour"] = dt.hour
        intent["minute"] = dt.minute
        intent["day"] = WEEKDAY_CN[dt.weekday()]
        intent["day_next_week"] = False
        intent.pop("absolute_date", None)

    return intent


def _intent_display(value, fallback="未指定"):
    if value is None or str(value).strip() in ("", "null", "None"):
        return fallback
    return str(value)


def render_intent_summary(
    intent: dict,
    *,
    info_mode: bool = False,
    resolved_top_n: int | None = None,
    top_n_default: bool = False,
):
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
        info_focus = intent.get("_info_focus")
        if info_focus:
            intent_fields.append(("查詢焦點", str(info_focus)))
        fee_slot = intent.get("_fee_slot_label")
        if fee_slot:
            intent_fields.append(("收費時段", str(fee_slot)))
    else:
        if resolved_top_n is not None:
            qty_label = (
                f"{resolved_top_n}（預設）"
                if top_n_default
                else str(resolved_top_n)
            )
        else:
            qty_label = _intent_display(intent.get("top_n"))
        intent_fields = [
            ("地區", _intent_display(place_label)),
            (
                "日期",
                format_intent_date(
                    intent.get("day"),
                    next_week=bool(intent.get("day_next_week")),
                    absolute_date=intent.get("absolute_date"),
                ),
            ),
            ("時間", format_intent_time(intent.get("hour"), intent.get("minute"))),
            ("顯示數量", qty_label),
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
            load_carparks_meta()[
                ["park_id", "name", "district", "sub_district", "latitude", "longitude"]
            ]
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

    hour = intent.get("hour")
    minute_raw = intent.get("minute")
    if has_explicit_clock_time(query) and not intent_has_clock_time(hour):
        extracted = extract_clock_time_from_query(query)
        if extracted is not None:
            hour, minute_raw = extracted

    if has_explicit_clock_time(query) and intent_has_clock_time(hour):
        day = explicit_day or intent.get("day") or WEEKDAY_CN[now.weekday()]
        minute = (
            0
            if minute_raw is None or str(minute_raw).strip() in ("", "null", "None")
            else int(minute_raw)
        )
        hour, minute = normalize_half_hour(int(hour), minute)
        return str(day), hour, minute, False

    return WEEKDAY_CN[now.weekday()], int(now.hour), int(now.minute), True


def needs_fee_missing_time_reentry(intent: dict, query_text: str) -> bool:
    """收費查詢：非今天（聽日／某一日）且未提供時間 → 要求重新輸入。"""
    if not looks_like_fee_query(query_text):
        return False
    # 「有顯示收費／有收費資料」唔使指定鐘點
    if looks_like_fee_availability_query(query_text):
        return False
    if is_now_query(query_text):
        return False
    has_hour = intent_has_clock_time(intent.get("hour")) or (
        has_explicit_clock_time(query_text)
        and extract_clock_time_from_query(query_text) is not None
    )
    if has_explicit_clock_time(query_text) and has_hour:
        return False
    explicit_day = resolve_relative_day(query_text)
    if not explicit_day:
        return False
    # 「下星期二」即使今日亦係星期二，仍屬未來日，要具體時間
    if query_implies_next_week_weekday(query_text) or intent.get("day_next_week"):
        return True
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


def _parking_info_name_block(
    parks_df: pd.DataFrame,
    *,
    include_fees: bool,
    limit: int = INFO_TABLE_PREVIEW_MAX,
) -> str:
    """組 LLM／fallback 用嘅車場名單摘要（長名單都會抽樣，唔好交空白）。"""
    n = len(parks_df)
    if n == 0:
        return "（無）"
    lines = []
    for i, (_, row) in enumerate(parks_df.head(limit).iterrows(), start=1):
        district = row.get("地區")
        district_bit = f"（{district}）" if district not in (None, "") else ""
        line = f"{i}. {row.get('名稱', '未命名')}{district_bit}"
        if include_fees and "收費" in parks_df.columns:
            line += f"｜收費：{row.get('收費') or '未提供'}"
        lines.append(line)
    listed = "\n".join(lines)
    if n > limit:
        listed += f"\n…另有 {n - limit} 個未列出"
    return listed


def _fallback_parking_info_answer(
    query: str,
    district_label: str,
    parks_df: pd.DataFrame,
    *,
    fee_slot_label: str | None = None,
    fee_availability: bool = False,
    fee_availability_mode: str | None = None,
    scope_total: int | None = None,
) -> tuple[str, str]:
    """LLM 失敗時的規則回覆。"""
    n = len(parks_df)
    names = parks_df["名稱"].tolist() if "名稱" in parks_df.columns else []
    has_fees = bool(fee_slot_label) and "收費" in parks_df.columns and n > 0
    total = int(scope_total) if scope_total is not None else n
    missing_fees = fee_availability_mode == "missing"

    if n == 0 and not fee_availability:
        written = (
            f"本系統在「{district_label}」附近目前沒有停車場紀錄。"
            f"你可以改試其他地點，或直接問空位預測"
            f"（例如：天水圍今晚八點邊度較易泊？）。"
        )
    elif fee_availability and missing_fees:
        if n == 0:
            written = (
                f"以本系統資料，「{district_label}」目前錄得 {total} 個停車場，"
                f"全部都有顯示收費（政府開放資料私家車規則），"
                f"暫時沒有「冇顯示收費」的停車場。"
            )
        else:
            samples = [f"「{row['名稱']}」" for _, row in parks_df.head(8).iterrows()]
            sample_txt = "、".join(samples)
            more = f" 等共 {n} 個" if n > 8 else f"，共 {n} 個"
            written = (
                f"「{district_label}」範圍本系統共 {total} 個停車場，"
                f"其中 {n} 個冇顯示收費（政府開放資料暫無私家車收費規則）。"
                f"例如：{sample_txt}{more}。"
                f"詳細見下表；可指定地區再問以收窄名單。"
            )
    elif fee_availability:
        if n == 0:
            written = (
                f"以本系統資料，「{district_label}」目前錄得 {total} 個停車場，"
                f"但暫時未有政府開放資料可顯示嘅私家車收費規則。"
                f"你可以改問某區收費（例如：觀塘停車場收費如何？），"
                f"或問空位預測（例如：觀塘今晚八點邊度較易泊？）。"
            )
        else:
            samples = []
            for _, row in parks_df.head(8).iterrows():
                fee = str(row.get("收費") or "").strip()
                if fee and not fee_value_missing(fee):
                    samples.append(f"「{row['名稱']}」{fee}")
                else:
                    samples.append(f"「{row['名稱']}」")
            sample_txt = "、".join(samples)
            more = f" 等共 {n} 個" if n > 8 else f"，共 {n} 個"
            written = (
                f"「{district_label}」範圍本系統共 {total} 個停車場，"
                f"其中 {n} 個有顯示收費（政府開放資料私家車規則）。"
                f"例如：{sample_txt}{more}。"
                f"詳細見下表；想知某場幾錢可直接問該區／該場收費。"
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
    fee_availability: bool = False,
    fee_availability_mode: str | None = None,
    scope_total: int | None = None,
) -> tuple[str, str]:
    """根據本系統停車場名單，用 LLM 認真回答非預測問題。"""
    n = len(parks_df)
    total = int(scope_total) if scope_total is not None else n
    with_fee_in_scope, _ = fee_coverage_counts(parks_df)
    missing_fees = fee_availability_mode == "missing"
    if fee_availability and scope_total is not None:
        # parks_df 已按有／無收費篩過
        matched = n
        with_fee_in_scope = 0 if missing_fees else n
        without_fee_in_scope = n if missing_fees else max(total - n, 0)
    else:
        matched = n
        without_fee_in_scope = max(total - with_fee_in_scope, 0)
    # 「冇顯示收費」名單唔使逐行標收費；「有顯示」則附上價錢作證據
    include_fees = (
        bool(fee_slot_label or (fee_availability and not missing_fees))
        and "收費" in parks_df.columns
    )
    # 長名單／全港都要抽樣，避免 LLM 無事實可答
    name_block = _parking_info_name_block(parks_df, include_fees=include_fees)

    fee_fact = ""
    fee_guidance = ""
    if fee_availability and missing_fees:
        fee_fact = (
            f"- 冇顯示收費的停車場：{matched} 個（範圍總數 {total}；"
            f"有顯示收費 {max(total - matched, 0)} 個）\n"
            f"- 收費來源：政府停車場開放資料（私家車時租／日夜泊；無規則則視為冇顯示）\n"
        )
        fee_guidance = (
            "用戶在問邊啲停車場冇顯示收費／無收費資料："
            "請直接根據事實說明冇顯示收費的數量，並舉幾個名稱例子；"
            "不要改答成「有顯示收費」名單，也不要捏造價錢。"
            "可提醒下表列出無收費資料的車場；範圍太大時建議再收窄地區。\n"
        )
    elif fee_availability:
        fee_fact = (
            f"- 有顯示收費的停車場：{matched} 個（範圍總數 {total}；"
            f"冇顯示收費 {without_fee_in_scope} 個）\n"
            f"- 收費來源：政府停車場開放資料（私家車時租／日夜泊；無規則則無顯示）\n"
        )
        fee_guidance = (
            "用戶在問邊啲停車場有顯示收費／有收費資料："
            "請直接根據事實說明有顯示收費的數量，並舉幾個名稱例子；"
            "不要改答成「全港收費一覽」或捏造價錢。"
            "可提醒下表列出有收費資料的車場；範圍太大時建議再收窄地區。\n"
        )
    elif fee_slot_label and "收費" in parks_df.columns:
        fee_fact = (
            f"- 收費時段：{fee_slot_label}\n"
            f"- 有收費資料：{with_fee_in_scope}／{total}\n"
            f"- 收費來源：政府停車場開放資料（私家車時租／日夜泊；無資料則顯示未提供）\n"
        )
        fee_guidance = (
            "用戶在問收費：請依事實說明該時段收費概況，可舉數個有資料的例子，"
            "並提醒下表可見更多；不要捏造價錢。範圍太大時建議指定地區再問。\n"
        )

    prompt = f"""
你是「咪再兜」香港停車場空位預測系統的助手。用戶的問題與停車場有關，但不是在問空位預測。
請根據「本系統資料庫事實」認真、盡力正確回答，用繁體中文、簡潔清楚（可用粵語口語）。
只根據事實作答，不要捏造停車場或收費；若資料不足就誠實說明，並建議可怎樣收窄問題。
{fee_guidance}回答後可自然地提一句：若要查某時段較易泊／空位，可以再問預測問題。

用戶問題：{query}

事實：
- 查詢地區：{district_label}
- 該範圍停車場數量：{total}
{fee_fact}- 停車場名單（抽樣，已按相關性排序）：
{name_block}
- 本系統主要功能：預測香港各區停車場在指定時段的空位，並推薦較易泊的停車場

只回傳 JSON，不要其他文字：
{{"answer": "給用戶看的書面回覆", "spoken": "給語音朗讀的粵語口語版本，一句至數句即可"}}
"""
    text = call_deepseek(prompt)
    fallback_kwargs = dict(
        fee_slot_label=fee_slot_label,
        fee_availability=fee_availability,
        fee_availability_mode=fee_availability_mode,
        scope_total=total,
    )
    if not text:
        return _fallback_parking_info_answer(
            query, district_label, parks_df, **fallback_kwargs
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
                query, district_label, parks_df, **fallback_kwargs
            )
        return answer, spoken or answer
    except Exception:
        return _fallback_parking_info_answer(
            query, district_label, parks_df, **fallback_kwargs
        )


def render_saved_info_answer(payload: dict, *, auto_speak: bool = False) -> None:
    """重顯示已儲存的停車場資料／說明回覆。"""
    intent = dict(payload.get("intent") or {})
    fee_slot_label = payload.get("fee_slot_label")
    fee_availability = bool(payload.get("fee_availability"))
    avail_mode = payload.get("fee_availability_mode")
    if fee_availability and avail_mode not in ("present", "missing"):
        avail_mode = "present"
    missing_fees = avail_mode == "missing"
    scope_total = payload.get("scope_total")
    # 舊 session payload 無 wants_fee：有 fee_slot_label 即當明確問收費
    wants_fee = (
        bool(payload["wants_fee"])
        if "wants_fee" in payload
        else bool(fee_slot_label)
    )
    if fee_availability:
        intent["_info_focus"] = (
            "冇顯示收費／無收費資料" if missing_fees else "有顯示收費／收費資料"
        )
        intent.pop("_fee_slot_label", None)
    elif fee_slot_label:
        intent["_fee_slot_label"] = fee_slot_label
    render_intent_summary(intent, info_mode=True)
    district_label = payload.get("district_label") or district_display_name(
        intent.get("district")
    )
    count = int(payload.get("count") or 0)
    total = int(scope_total) if scope_total is not None else count
    nearby_suffix = "附近" if intent.get("geo_lat") is not None else ""
    if fee_availability and missing_fees:
        st.success(
            f"「{district_label}」{nearby_suffix}冇顯示收費的停車場"
            f"（{count}／{total} 個）"
        )
    elif fee_availability:
        st.success(
            f"「{district_label}」{nearby_suffix}有顯示收費的停車場"
            f"（{count}／{total} 個）"
        )
    elif wants_fee and fee_slot_label:
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
        show_written=SHOW_WRITTEN_INFO_ANSWER,
    )
    parks_df = parks_df_for_display(pd.DataFrame(payload.get("park_records") or []))
    if not parks_df.empty:
        # 表格顯示完整名單；地圖只取前 N 個標記，避免幾百點卡死瀏覽器
        map_n = min(len(parks_df), INFO_MAP_MARKER_MAX)
        map_df = parks_df.head(map_n)
        display_df = parks_df.drop(columns=list(_TABLE_HIDDEN_COLS), errors="ignore")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        render_results_map(map_df, intent=intent)
        map_note = (
            f"地圖先標示前 {map_n} 個（共 {count} 個）。"
            if count > map_n
            else ""
        )
        if fee_availability and missing_fees:
            st.caption(
                f"上表為政府開放資料暫無私家車收費規則的停車場"
                f"（共 {count} 個；範圍總數 {total}）。"
                f"{map_note}"
                "可指定地區再問以收窄名單。"
            )
        elif fee_availability:
            st.caption(
                f"上表為有政府開放資料私家車收費規則的停車場"
                f"（共 {count} 個；範圍總數 {total}）。"
                f"{map_note}"
                "想知某場幾錢可再問該區／該場收費。"
            )
        elif fee_slot_label and "收費" in parks_df.columns:
            st.caption(
                f"收費為「{fee_slot_label}」時段（政府開放資料私家車規則；無資料則顯示未提供）。"
                f"{map_note}"
            )
        elif map_note:
            st.caption(f"該範圍共有 {count} 個停車場。{map_note}可指定地區再問。")
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
    avail_mode = fee_availability_mode(query_text)
    fee_availability = avail_mode is not None
    wants_fee = looks_like_fee_query(query_text) or fee_availability
    fee_slot_label = None
    scope_total = int(len(parks_df)) if parks_df is not None else 0
    if not parks_df.empty:
        day_cn, hour, minute, is_current = resolve_fee_query_slot(intent, query_text)
        parks_df = attach_fees_to_parks_df(parks_df, day_cn, hour, minute)
        if is_current:
            fee_slot_label = f"現在 {format_time_ampm(hour, minute)}"
        else:
            fee_slot_label = f"{day_cn} {format_time_12h(hour, minute)}"
        # 地標置頂；問收費時把有規則嘅車場排前，避免成表「未提供」
        venue_label = resolve_venue_focus_label(query_text, str(district or ""))
        at_ids: set[str] | None = None
        if venue_label:
            at_place = filter_parks_at_queried_place(
                parks_df, venue_label, name_match_required=True
            )
            if not at_place.empty and "park_id" in at_place.columns:
                at_ids = set(at_place["park_id"].astype(str))
        if wants_fee or at_ids:
            parks_df = order_parks_for_fee_display(
                parks_df,
                at_place_ids=at_ids,
            )
        # 「有／冇顯示收費」→ 按極性篩選作答
        if avail_mode == "missing":
            parks_df = filter_parks_without_fee(parks_df)
        elif avail_mode == "present":
            parks_df = filter_parks_with_fee(parks_df)

    if match_type == "nearby" and scope_total == 0:
        intent_for_summary = dict(intent)
        if wants_fee and fee_slot_label and not fee_availability:
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
            fee_slot_label=(
                None
                if fee_availability
                else (fee_slot_label if wants_fee else None)
            ),
            fee_availability=fee_availability,
            fee_availability_mode=avail_mode,
            scope_total=scope_total,
        )

    payload = {
        "intent": intent,
        "district_label": district_label,
        "count": int(len(parks_df)),
        "scope_total": scope_total,
        "park_records": parks_df_for_display(parks_df).to_dict(orient="records"),
        "written": written,
        "spoken": spoken,
        "fee_slot_label": None if fee_availability else fee_slot_label,
        "wants_fee": wants_fee,
        "fee_availability": fee_availability,
        "fee_availability_mode": avail_mode,
    }
    _push_search_history(query_text)
    st.session_state.last_info_answer = payload
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_feedback", None)
    render_saved_info_answer(payload, auto_speak=True)


def resolve_top_n(query_text: str, intent: dict) -> tuple[int, bool]:
    """回傳 (top_n, used_default)。"""
    top_n = parse_top_n_from_query(query_text)
    used_default = False
    if top_n is None:
        raw_top_n = intent.get("top_n")
        if raw_top_n is not None and str(raw_top_n).strip() not in ("", "null", "None"):
            try:
                top_n = int(raw_top_n)
            except (TypeError, ValueError):
                top_n = DEFAULT_TOP_N
                used_default = True
        else:
            top_n = DEFAULT_TOP_N
            used_default = True
    if top_n < 1:
        top_n = DEFAULT_TOP_N
        used_default = True
    return top_n, used_default


def _push_search_history(query_text: str) -> None:
    q = (query_text or "").strip()
    if not q:
        return
    hist = [
        h
        for h in (st.session_state.get("search_history") or [])
        if h != q
    ]
    hist.insert(0, q)
    st.session_state.search_history = hist[:SEARCH_HISTORY_MAX]


def day_profile_for_parks(
    park_ids: list[str],
    names: dict[str, str],
    dow: int,
    baselines: tuple,
    *,
    highlight_half_hour: int,
) -> pd.DataFrame:
    """產生指定車場全日（半小時）預測曲線。"""
    rows: list[dict] = []
    for pid in park_ids:
        for hh in range(48):
            pred, meta = predict_park_vacancy_baseline(pid, hh, dow, baselines)
            if pred is None:
                continue
            hour = hh // 2
            minute = 30 if hh % 2 else 0
            rows.append(
                {
                    "park_id": pid,
                    "名稱": names.get(pid, pid),
                    "half_hour": hh,
                    "時間": f"{hour:02d}:{minute:02d}",
                    "預測空位": max(0, round(pred, 1)),
                    "查詢時段": hh == highlight_half_hour,
                    "信心": meta.get("confidence", "低"),
                }
            )
    return pd.DataFrame(rows)


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
        query_date = dt.replace(second=0, microsecond=0)
    else:
        hour = int(raw_hour)
        minute = (
            0
            if raw_minute is None or str(raw_minute).strip() in ("", "null", "None")
            else int(raw_minute)
        )
        hour, minute = normalize_half_hour(hour, minute)
        query_date = resolve_intent_calendar_date(
            day,
            now,
            next_week=bool(intent.get("day_next_week")),
            absolute_date=intent.get("absolute_date"),
        )
        if query_date is not None:
            query_date = query_date.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
        else:
            query_date = now.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
    half_hour = to_half_hour_slot(hour, minute)
    top_n, top_n_default = resolve_top_n(query_text, intent)
    requested_top_n = top_n
    dow_num = DAY_TO_NUM.get(day, 0)
    dist_map: dict[str, float] = {}
    match_type = "all"
    district_label = "全港"
    target_park_ids: list[str] | None = None

    # 非全港：先用靜態清單篩附近，再載入 baseline 摘要（方案 1）
    if _is_all_hong_kong(district):
        with st.spinner("正在載入全港空位預測基準…"):
            baselines = load_park_baselines()
        park_info = (
            load_carparks_meta()
            .drop_duplicates(subset=["park_id"])
            .set_index("park_id", drop=False)
        )
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
        target_park_ids = [str(p) for p in nearby["park_id"].tolist()]
        with st.spinner(f"正在載入「{district_label}」附近預測基準…"):
            park_key = tuple(sorted(target_park_ids))
            baselines = load_park_baselines(park_key)
        park_info = nearby.set_index("park_id", drop=False)
        if park_info.empty:
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
    # 確保 index／欄位一律字串 id，避免數字 id 整場消失
    if "park_id" in park_info.columns:
        park_info = park_info.copy()
        park_info["park_id"] = park_info["park_id"].map(coerce_park_id)
        park_info = park_info.set_index("park_id", drop=False)
    results = []
    for park_id in park_info.index:
        park_id = coerce_park_id(park_id)
        if not park_id:
            continue
        pred, meta = predict_park_vacancy_baseline(
            park_id, half_hour, dow_num, baselines
        )
        if pred is None:
            continue
        info = park_info.loc[park_id]
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        row = {
            "park_id": park_id,
            "name": info["name"],
            "latitude": info.get("latitude"),
            "longitude": info.get("longitude"),
            "名稱": info["name"],
            "地區": info.get("sub_district") or info.get("district"),
            "收費": format_carpark_fee(park_id, day, hour, minute, fee_map),
            "預測空位": max(0, round(pred)),
            "信心": meta.get("confidence", "低"),
            "樣本數": meta.get("n", 0),
            "依據": meta.get("source", ""),
        }
        if match_type == "nearby" and park_id in dist_map:
            row["距離(公里)"] = round(dist_map[park_id], 1)
        results.append(row)

    pred_df = pd.DataFrame(results)
    if not pred_df.empty:
        pred_df = dedupe_equivalent_carparks(
            pred_df, prefer_higher_col="預測空位"
        )
        # 保留 park_id／座標畀曲線同地圖；表格顯示時會隱藏
        pred_df = pred_df.drop(columns=["name"], errors="ignore")

    # 通用：原句問某商場／邨／地標「容唔容易泊」→ 只評該場所車場
    # （用地名抽取，避免 LLM 把 YOHO mall 收成「元朗」後誤走附近排名）
    place_label = resolve_place_ease_label(query_text, str(district or ""))
    place_ease = match_type == "nearby" and place_label is not None

    if place_ease:
        # 一律以名稱／別名認場（適用所有商場／邨／地標）；
        # 唔用距離，避免密集積偶把隔壁邨誤當「該場所」。
        at_place = filter_parks_at_queried_place(
            pred_df, place_label, name_match_required=True
        )
        if not at_place.empty:
            # 該場所全部車場（按空位）；唔再攤開附近其他場
            result_df = select_top_with_ties(
                at_place, max(len(at_place), 1)
            )
            district_label = place_label
            top_n = len(result_df)
            top_n_default = False
        else:
            # 名稱對唔上（資料缺該場名）→ 附近排名＋地標置頂（同樣只靠名稱）
            nearby_ranked = select_top_nearby_with_ties(pred_df, top_n)
            pinned = filter_parks_at_queried_place(
                pred_df, place_label, name_match_required=True
            )
            result_df = pin_venue_parks_into_ranking(
                pinned, nearby_ranked, top_n
            )
            district_label = f"{place_label}附近"
            place_ease = False
    elif match_type == "nearby":
        # 顯示優先用用戶原地名，其次 Google 地址
        display_place = (
            str(district or "").strip()
            or (geo_label or "").strip()
            or district_label
        )
        nearby_ranked = select_top_nearby_with_ties(pred_df, top_n)
        # 「德福廣場」等具體地標：該場所車場置頂，再補附近較易泊
        venue_label = resolve_venue_focus_label(query_text, str(district or ""))
        if venue_label:
            at_place = filter_parks_at_queried_place(
                pred_df, venue_label, name_match_required=True
            )
            result_df = pin_venue_parks_into_ranking(
                at_place, nearby_ranked, top_n
            )
        else:
            result_df = nearby_ranked
        district_label = f"{display_place}附近"
    else:
        result_df = select_top_with_ties(pred_df, top_n)

    # 置頂／合併後重算名次（避免名次欄 NaN → 建議文案 IndexError）
    if not result_df.empty and "預測空位" in result_df.columns:
        result_df = add_dense_ranks(
            result_df.drop(columns=[""], errors="ignore")
        )

    # 欄位順序：排名、名稱、地區、（距離）、收費、預測空位；（隱藏）信心／座標／內部欄
    preferred = ["", "名稱", "地區", "距離(公里)", "收費", "預測空位"]
    ordered = [c for c in preferred if c in result_df.columns]
    ordered += [
        c
        for c in result_df.columns
        if c not in ordered and c not in _TABLE_HIDDEN_COLS
    ]
    # 內部欄留喺 df 尾（表格會隱藏）
    for hidden in _TABLE_HIDDEN_COLS:
        if hidden in result_df.columns and hidden not in ordered:
            ordered.append(hidden)
    result_df = result_df[ordered]

    shown = len(result_df)
    # 用戶明確要求 N 個，但範圍內可用車場少過 N → 提示並列出全部
    available_count = int(len(pred_df)) if pred_df is not None else shown
    top_n_shortfall = (not place_ease) and top_n_exceeds_available(
        requested_top_n,
        available_count,
        used_default=top_n_default,
    )
    if top_n_shortfall:
        # select_top_* 喺 top_n ≥ 池大小時已回傳全部；標題改用實際數量
        available_count = shown
        top_n = shown

    # 有指明數量 → 朗讀該數量；未指明 → 只朗讀頭 SPEAK_DEFAULT_TOP_N 名
    speak_n = SPEAK_DEFAULT_TOP_N if top_n_default else int(top_n)
    if place_ease:
        written = format_top_suggestion(
            result_df, hour, minute, max_ranks=speak_n
        )
        spoken = format_spoken_place_ease(
            result_df,
            place_label,
            hour,
            minute,
            query_date=query_date,
        )
    else:
        written = format_top_suggestion(
            result_df, hour, minute, max_ranks=speak_n
        )
        spoken = format_spoken_suggestion(
            result_df,
            hour,
            minute,
            max_ranks=speak_n,
            query_date=query_date,
        )

    if not (spoken or "").strip():
        when = format_query_when_spoken_yue(query_date, hour, minute)
        spoken = prepare_spoken_text_for_tts(
            f"{when}。{district_label}暫時冇可用嘅預測結果。"
        )

    if top_n_shortfall:
        notice_w = format_top_n_shortfall_notice(
            requested_top_n,
            available_count,
            district_label,
            style="written",
        )
        notice_s = format_top_n_shortfall_notice(
            requested_top_n,
            available_count,
            district_label,
            style="spoken",
        )
        written = f"{notice_w}\n\n{written}" if written else notice_w
        # 保留開頭「年月日＋時間」，提示插喺第一句之後
        if spoken and "。" in spoken:
            lead, rest = spoken.split("。", 1)
            spoken = prepare_spoken_text_for_tts(
                f"{lead}。{notice_s}{rest}"
            )
        else:
            spoken = prepare_spoken_text_for_tts(
                f"{notice_s}{spoken}" if spoken else notice_s
            )

    profile_df = pd.DataFrame()
    if not result_df.empty and "park_id" in result_df.columns:
        top_ids = (
            result_df["park_id"].astype(str).drop_duplicates().head(3).tolist()
        )
        name_map = {
            str(r["park_id"]): str(r["名稱"])
            for _, r in result_df.drop_duplicates("park_id").iterrows()
        }
        profile_df = day_profile_for_parks(
            top_ids,
            name_map,
            dow_num,
            baselines,
            highlight_half_hour=half_hour,
        )

    payload = {
        "intent": intent,
        "asking_now": asking_now,
        "day": day,
        "hour": hour,
        "minute": minute,
        "district_label": district_label,
        "top_n": top_n,
        "top_n_default": top_n_default,
        "top_n_shortfall": top_n_shortfall,
        "requested_top_n": requested_top_n,
        "available_count": available_count,
        "shown": shown,
        "place_ease": place_ease,
        "result_records": result_df.to_dict(orient="records"),
        "profile_records": profile_df.to_dict(orient="records"),
        "written": written,
        "spoken": spoken,
    }
    _push_search_history(query_text)
    st.session_state.last_prediction = payload
    st.session_state.pop("last_info_answer", None)
    st.session_state.pop("last_feedback", None)
    render_saved_prediction(payload, auto_speak=True)


def needs_missing_time_reentry(intent: dict, query_text: str) -> bool:
    """明天或之後（或已指定未來絕對日期）且未提供時間 → 要求重新輸入。"""
    asking_now = is_now_query(query_text)
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    if asking_now:
        return False
    return is_future_day(
        day,
        next_week=bool(intent.get("day_next_week")),
        absolute_date=intent.get("absolute_date"),
    ) and not intent_has_clock_time(intent.get("hour"))


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
    """所指日子／時間已過去 → 不執行預測，要求重新輸入。

    含：琴日／前日等、無年份落喺今個月／上個月但已過嘅月日、今日但鐘點已過。
    """
    if is_now_query(query_text):
        return False
    # 句子明確講過去相對日子 → 一律當已過（即使 LLM 漏填 absolute_date）
    if parse_past_relative_calendar_date(query_text) is not None:
        return True
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    return is_past_intent_datetime(
        day,
        intent.get("hour"),
        intent.get("minute"),
        next_week=bool(intent.get("day_next_week")),
        absolute_date=intent.get("absolute_date"),
    )


def needs_beyond_horizon_followup(intent: dict, query_text: str) -> bool:
    """所指日子超過可預測範圍（今日起計 MAX_PREDICTION_HORIZON_DAYS 日）→ 要求重新輸入。"""
    if is_now_query(query_text):
        return False
    today = hong_kong_now()
    day = intent.get("day", WEEKDAY_CN[today.weekday()])
    date_part = resolve_intent_calendar_date(
        day,
        today,
        next_week=bool(intent.get("day_next_week")),
        absolute_date=intent.get("absolute_date"),
    )
    if date_part is None:
        return False
    return is_beyond_prediction_horizon(
        date_part,
        today=today,
        horizon_days=MAX_PREDICTION_HORIZON_DAYS,
    )


def render_missing_time_blocked(intent: dict, *, auto_speak: bool = True):
    """未來日期但未指明時間：顯示意圖並要求重新輸入完整查詢。"""
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    day_label = format_intent_date(
        day,
        next_week=bool(intent.get("day_next_week")),
        absolute_date=intent.get("absolute_date"),
    )
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
    abs_dt = (
        parse_absolute_date_from_query(query_text)
        or absolute_date_from_intent(intent)
        or parse_past_relative_calendar_date(query_text)
    )
    day = (
        (WEEKDAY_CN[abs_dt.weekday()] if abs_dt is not None else None)
        or resolve_relative_day(query_text)
        or intent.get("day")
        or WEEKDAY_CN[hong_kong_now().weekday()]
    )
    day_label = format_intent_date(
        day,
        next_week=bool(
            intent.get("day_next_week")
            or query_implies_next_week_weekday(query_text)
        ),
        absolute_date=(
            abs_dt.strftime("%Y-%m-%d")
            if abs_dt is not None
            else intent.get("absolute_date")
        ),
    )
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
    """查詢日子／時間已過：顯示意圖並要求重新輸入完整查詢。"""
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    date_label = format_intent_date(
        day,
        next_week=bool(intent.get("day_next_week")),
        absolute_date=intent.get("absolute_date"),
    )
    time_label = format_intent_time(intent.get("hour"), intent.get("minute"))
    if time_label == "未指定":
        when = date_label
        written = (
            f"你想查詢的是 **{when}**，但該日子已過去。請重新輸入查詢內容。"
        )
        spoken = f"你想查嘅係{when}，但該日子已經過去。請重新輸入查詢內容。"
    else:
        when = f"{date_label} {time_label}"
        written = (
            f"你想查詢的是 **{when}**，但該時間已過去。請重新輸入查詢內容。"
        )
        spoken = f"你想查嘅係{when}，但該時間已經過去。請重新輸入查詢內容。"
    render_feedback_with_speech(
        written,
        spoken=spoken,
        intent=intent,
        auto_speak=auto_speak,
    )


def render_beyond_horizon_blocked(intent: dict, *, auto_speak: bool = True):
    """查詢日子超過可預測範圍：顯示意圖並要求重新輸入。"""
    day = intent.get("day", WEEKDAY_CN[hong_kong_now().weekday()])
    date_label = format_intent_date(
        day,
        next_week=bool(intent.get("day_next_week")),
        absolute_date=intent.get("absolute_date"),
    )
    time_label = format_intent_time(intent.get("hour"), intent.get("minute"))
    when = date_label if time_label == "未指定" else f"{date_label} {time_label}"
    n = MAX_PREDICTION_HORIZON_DAYS
    render_feedback_with_speech(
        f"你想查詢的是 **{when}**，但本系統只提供未來 **{n} 日內** 的空位預測。"
        f"請重新輸入查詢內容（例如：天水圍大後日下午三點邊度較易泊？）。",
        spoken=(
            f"你想查嘅係{when}，但本系統只提供未來{n}日內嘅空位預測。"
            "請重新輸入查詢內容，例如天水圍大後日下午三點邊度較易泊。"
        ),
        intent=intent,
        auto_speak=auto_speak,
    )


def render_invalid_calendar_date_blocked(
    intent: dict | None, date_label: str, *, auto_speak: bool = True
):
    """無效月日（如 11月31號）：唔好讓 LLM 改寫成其他日子，要求重新輸入。"""
    # 清掉估錯嘅 absolute_date／星期，避免意圖卡顯示「11月9日」
    # 早攔截（未叫 LLM）時 intent 可為 None，唔顯示意圖卡
    display_intent = None
    if intent is not None:
        display_intent = {
            **intent,
            "day": None,
            "absolute_date": None,
            "day_next_week": False,
        }
    render_feedback_with_speech(
        f"你輸入的日期 **{date_label}** 不存在。"
        f"請重新輸入查詢內容（例如：11月30號朝早六點半彩雲邨邊度較易泊？）。",
        spoken=SPOKEN_INVALID_CALENDAR_DATE,
        intent=display_intent,
        auto_speak=auto_speak,
    )


def render_invalid_clock_time_blocked(
    intent: dict | None, time_label: str, *, auto_speak: bool = True
):
    """無效鐘點（如 68點30分）：唔好讓 LLM 改寫成其他時間，要求重新輸入。"""
    display_intent = None
    if intent is not None:
        display_intent = {**intent, "hour": None, "minute": None}
    render_feedback_with_speech(
        f"你輸入的時間 **{time_label}** 不存在。"
        f"請重新輸入查詢內容（例如：元朗後日上晝八點半邊度較易泊？）。",
        spoken=SPOKEN_INVALID_CLOCK_TIME,
        intent=display_intent,
        auto_speak=auto_speak,
    )


def render_hard_to_park_unsupported(intent: dict, *, auto_speak: bool = True):
    """用戶問較難／冇咁易泊：本系統只提供較易泊排名。"""
    district = (
        intent.get("geo_label")
        or intent.get("district")
        or "該地區"
    )
    if _is_all_hong_kong(intent.get("district")):
        district = "全港"
    example = f"{district}今晚八點邊度較易泊？"
    render_feedback_with_speech(
        "本系統只能夠顯示較易泊車的停車場，暫時未能排列「冇咁容易／較難泊」的名單。"
        f"請改問較易泊（例如：{example}）。",
        spoken=(
            "本系統只能夠顯示較易泊車嘅停車場，暫時未能排列冇咁容易或者較難泊嘅名單。"
            f"請改問較易泊，例如{example}"
        ),
        intent=intent,
        auto_speak=auto_speak,
    )


def render_non_half_hour_blocked(intent: dict | None = None, *, auto_speak: bool = True):
    """時間唔係半小時單位：不執行預測，要求用半小時方式重問。"""
    # 唔顯示被四捨五入後嘅鐘點，避免誤導（例如 4:45 → 4:30）
    display_intent = None
    if intent is not None:
        display_intent = {**intent, "hour": None, "minute": None}
    render_feedback_with_speech(
        "這樣是查詢不到的。本系統必須以**半小時**為單位查問"
        "（例如：六點半、十二點、4:30、11:00），"
        "不接受四點九、4:45、九點二十七分這類時間。請重新輸入查詢內容。",
        spoken=SPOKEN_NON_HALF_HOUR,
        intent=display_intent,
        auto_speak=auto_speak,
    )


auto_submit = st.session_state.pop("auto_submit_query", False)
should_query = submitted or auto_submit
query_text = (user_query or "").strip()
# 語音送出標記：結果頁要用更長自動播重試（唔好只 set 唔用）
voice_query_unlock = bool(st.session_state.pop("_voice_query_unlock", False))
# 文字輸入同樣校正（語音已喺辨識後改過；此處作雙重保險）
if query_text:
    query_text, typed_asr_note = normalize_asr_query(query_text)
    if typed_asr_note and not asr_note:
        st.caption(typed_asr_note)

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
    # 查詢掣／語音送出都喺手勢內 unlock：一律加固第一次自動播
    if voice_query_unlock or auto_submit or submitted:
        st.session_state["_voice_auto_speak_this_query"] = True
        inject_voice_auto_speak_prime()
    else:
        st.session_state.pop("_voice_auto_speak_this_query", None)

    # 無效月日／鐘點／非半小時：規則可即判，唔好等 LLM（縮短等待，保住手機 unlock）
    early_invalid_date = query_mentions_invalid_calendar_date(query_text)
    early_invalid_time = (
        None
        if early_invalid_date
        else query_mentions_invalid_clock_time(query_text)
    )
    if early_invalid_date:
        render_invalid_calendar_date_blocked(None, early_invalid_date)
    elif early_invalid_time:
        render_invalid_clock_time_blocked(None, early_invalid_time)
    elif query_specifies_non_half_hour_time(query_text):
        # 4:45／四點九等：唔好先四捨五入再當「已過去」
        render_non_half_hour_blocked(None)
    else:
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
        elif looks_like_hard_to_park_ranking_query(query_text):
            # 「冇咁容易泊／較難泊」：只提示可查較易泊，唔好誤顯示較易泊名單
            render_hard_to_park_unsupported(intent)
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
            elif needs_past_time_followup(intent, query_text):
                # 過去日子優先於「未提供時間／曖昧上下午」，直接要求重新查詢
                past_rel = parse_past_relative_calendar_date(query_text)
                if past_rel is not None and not intent.get("absolute_date"):
                    intent["absolute_date"] = past_rel.strftime("%Y-%m-%d")
                    intent["day"] = WEEKDAY_CN[past_rel.weekday()]
                    intent["day_next_week"] = False
                render_past_time_blocked(intent)
            elif needs_beyond_horizon_followup(intent, query_text):
                # 超過未來 14 日：唔進入預測（即使未填時間／曖昧上下午）
                render_beyond_horizon_blocked(intent)
            elif needs_missing_time_reentry(intent, query_text):
                render_missing_time_blocked(intent)
            elif needs_ambiguous_ampm_reentry(intent, query_text):
                render_ambiguous_ampm_blocked(intent)
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
elif not should_query:
    # 未搜尋：預設顯示附近即時空位
    render_nearby_live_slots()
