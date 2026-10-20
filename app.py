import streamlit as st
import sqlite3
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv
import html
import io
import os
import json
import re
import wave
import numpy as np
import requests
from datetime import datetime
from urllib.parse import urlencode
from pyflac import StreamEncoder

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

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# Google STT 需要 FLAC；不用 speech_recognition 內建的 x86_64 flac-mac。
GOOGLE_STT_URL = "http://www.google.com/speech-api/v2/recognize"
GOOGLE_STT_KEY = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"


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

st.title("香港停車場空位預測")
st.write("全港 18 區｜AI Agent 查詢")

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# 語音辨識結果先寫入 pending，再同步到查詢欄（避免 form widget 狀態不同步）
if st.session_state.get("pending_query"):
    st.session_state.user_query = st.session_state.pending_query

# 文字輸入 + 語音輸入（Enter 或按「查詢」皆可送出）
st.caption("輸入查詢（例如：元朗今晚八點邊度較易泊？）")
query_col, voice_col = st.columns([6, 1], vertical_alignment="bottom")
with query_col:
    with st.form("query_form", border=False):
        user_query = st.text_input(
            "輸入查詢",
            placeholder="輸入你的問題...",
            label_visibility="collapsed",
            key="user_query",
        )
        submitted = st.form_submit_button("查詢", use_container_width=False)
with voice_col:
    audio = mic_recorder(
        start_prompt="🎤",
        stop_prompt="⏹",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="voice_input",
    )
    if audio and audio.get("id") != st.session_state.get("_last_voice_audio_id"):
        st.session_state._last_voice_audio_id = audio["id"]
        with st.spinner("正在辨識語音..."):
            try:
                voice_text = transcribe_audio(audio)
            except Exception as e:
                voice_text = ""
                st.error(f"語音辨識服務暫時無法使用：{e}")
        if voice_text:
            st.session_state.pending_query = voice_text
            st.session_state.auto_submit_query = True
            st.rerun()
        else:
            st.warning("未能辨識語音，請再說一次或改用文字輸入。")

@st.cache_data
def load_data():
    conn = sqlite3.connect("parking.db")
    df = pd.read_sql_query("""
        SELECT v.park_id, v.vacancy, v.collected_at, c.name, c.district
        FROM vacancy_records v
        LEFT JOIN carparks c ON v.park_id = c.park_id
    """, conn)
    conn.close()
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df["hour"] = df["collected_at"].dt.hour
    df["dow"] = df["collected_at"].dt.dayofweek
    return df

df = load_data()

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


def format_top_suggestion(result_df: pd.DataFrame, hour: int) -> str:
    """依 1st 名次（含同分）產生建議文案。"""
    first_rank = result_df.iloc[0][""]
    top_df = result_df[result_df[""] == first_rank]
    names = [str(n) for n in top_df["名稱"].tolist()]
    vacancy = int(top_df.iloc[0]["預測空位"])

    if len(names) == 1:
        return f"建議於{hour}時前往{names[0]}，預計約有{vacancy}個空位。"

    names_text = " 或 ".join(names)
    count_text = NUM_CN.get(len(names), str(len(names)))
    return (
        f"建議於{hour}時前往{names_text}，"
        f"這{count_text}個停車場同樣分別預計約有{vacancy}個空位。"
    )


def render_ranked_table(df: pd.DataFrame) -> None:
    """以 HTML 表格顯示結果；同分排名用 rowspan 合併為同一格。"""
    if df.empty:
        st.info("沒有可顯示的停車場。")
        return

    rank_col = ""
    data_cols = [c for c in df.columns if c != rank_col]

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
<div style="width:100%;overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)


def parse_query_with_llm(query: str):
    """用 DeepSeek 理解查詢意圖"""
    today = datetime.now()
    weekday_map = {0: "週一", 1: "週二", 2: "週三", 3: "週四", 4: "週五", 5: "週六", 6: "週日"}
    today_day = weekday_map[today.weekday()]
    today_str = today.strftime("%Y-%m-%d")

    prompt = f"""
你是一個香港停車場查詢助手。今天是 {today_str}（{today_day}）。
請從以下句子提取資訊，並以 JSON 格式回傳。
只回傳 JSON，不要其他文字。

句子：{query}

請提取：
- district: 地區名稱（例如「元朗區」、「觀塘區」，如果沒有提到就填「全部」）
- hour: 小時（0-23 的數字，如果沒有提到就填 19）
- day: 星期（「週一」到「週日」）。如果句子沒有提到星期，就填今天的星期「{today_day}」
- top_n: 要顯示幾個較易泊車的停車場（整數）。規則：
  - 「邊個／哪個／邊一個／哪一個」→ 1
  - 「邊兩個／哪兩個」→ 2、「邊三個／哪三個」→ 3，如此類推
  - 如果沒有提到要幾個，填 null

範例輸出：
{{"district": "元朗區", "hour": 20, "day": "{today_day}", "top_n": 2}}
"""
    text = call_deepseek(prompt)
    try:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        st.error(f"解析失敗：{e}")
        return {"district": "全部", "hour": 19, "day": today_day, "top_n": None}

auto_submit = st.session_state.pop("auto_submit_query", False)
should_query = submitted or auto_submit
if should_query:
    query_text = (st.session_state.pop("pending_query", None) or user_query or "").strip()
else:
    query_text = (user_query or "").strip()

if should_query and query_text:
    with st.spinner("AI 正在理解你的問題..."):
        intent = parse_query_with_llm(query_text)
        st.write("**AI 理解的意圖：**", intent)
        
        district = intent.get("district", "全部")
        hour = int(intent.get("hour", 19))
        day = intent.get("day", "週一")

        # 優先用規則解析「邊個／哪兩個」等；否則用 LLM；再否則預設 10
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
        
        # 轉成數字
        day_map = {"週一": 0, "週二": 1, "週三": 2, "週四": 3, "週五": 4, "週六": 5, "週日": 6}
        dow_num = day_map.get(day, 0)
        
        # 預測
        le = LabelEncoder()
        df["park_code"] = le.fit_transform(df["park_id"])
        
        features = ["hour", "dow", "park_code"]
        X = df[features]
        y = df["vacancy"]
        
        model = GradientBoostingRegressor(random_state=42, n_estimators=50)
        model.fit(X, y)
        
        if district != "全部" and district in df["district"].values:
            df_filtered = df[df["district"] == district]
        else:
            df_filtered = df
        
        results = []
        for park_id in df_filtered["park_id"].unique():
            park_code = le.transform([park_id])[0]
            pred = model.predict([[hour, dow_num, park_code]])[0]
            info = df[df["park_id"] == park_id].iloc[0]
            results.append({
                "名稱": info["name"],
                "地區": info["district"],
                "預測空位": max(0, round(pred))
            })
        
        # 取前 top_n；邊界同分一併顯示，並加上 1st/2nd… 排名欄
        result_df = select_top_with_ties(pd.DataFrame(results), top_n)
        shown = len(result_df)
        if shown > top_n:
            st.success(
                f"{day} {hour}:00 {district} 較易泊車的停車場"
                f"（前 {top_n} 名，含同分共 {shown} 個）"
            )
        else:
            st.success(f"{day} {hour}:00 {district} 較易泊車的停車場（前 {top_n} 名）")
        render_ranked_table(result_df)
        
        if not result_df.empty:
            st.subheader("建議")
            st.write(format_top_suggestion(result_df, hour))
elif should_query:
    st.warning("請先輸入查詢內容。")