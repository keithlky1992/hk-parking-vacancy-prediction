"""結果表／查詢輔助：收費缺漏、地標置頂、ASR 常見誤辨校正等。"""

from __future__ import annotations

import math
import re
from typing import Callable

import pandas as pd

FEE_MISSING_TOKENS = frozenset(
    {"", "未提供", "—", "-", "–", "n/a", "na", "none", "null", "nan"}
)


def normalize_asr_query_text(query: str) -> tuple[str, bool]:
    """校正常見粵語語音誤辨。回傳 (校正後句子, 是否有改動)。"""
    original = (query or "").strip()
    if not original:
        return original, False

    compact = re.sub(r"\s+", "", original)
    # 「易泊」常被聽成 200／二百
    yibo_mishear = r"(?:200|二百|兩百|两百|贰佰|貳佰|2百)"
    # 「易唔易」常被聽成 252／250（例如「…九點半252泊車」）
    yimyi_mishear = r"(?:252|250|二百五十二|兩百五十二|两百五十二)"
    # 「比較二百蚊」＝價錢，唔好改成易泊
    if re.search(
        rf"(?:比較|較|较|邊度|哪[里裡]|好|易唔易){yibo_mishear}(?:蚊|元|塊|美金|港幣|hkd)",
        compact,
        flags=re.IGNORECASE,
    ):
        return original, False
    if re.search(
        rf"{yimyi_mishear}(?:蚊|元|塊|美金|港幣|hkd)",
        compact,
        flags=re.IGNORECASE,
    ):
        return original, False

    replacements = (
        # 容唔容易 → 用唔容易（粵語語音常把「容」聽成「用」）
        (r"用唔容易", "容唔容易"),
        (r"用不容易", "容不容易"),
        # 易唔易 → 252／250（必須喺「泊／泊車」前，避免誤傷門牌）
        (rf"{yimyi_mishear}\s*泊車", "易唔易泊車"),
        (rf"{yimyi_mishear}\s*泊", "易唔易泊"),
        # 要唔要泊 ≈ 易唔易泊（語音相近）
        (r"要唔要\s*泊車", "易唔易泊車"),
        (r"要唔要\s*泊", "易唔易泊"),
        # 易泊 → 200／二百
        (rf"比較\s*{yibo_mishear}", "比較易泊"),
        (rf"较\s*{yibo_mishear}", "較易泊"),  # 簡體「较」
        (rf"較\s*{yibo_mishear}", "較易泊"),
        (rf"邊度\s*{yibo_mishear}", "邊度易泊"),
        (rf"哪里\s*{yibo_mishear}", "哪里易泊"),
        (rf"哪裡\s*{yibo_mishear}", "哪裡易泊"),
        (rf"好\s*{yibo_mishear}", "好泊"),
        (rf"易唔易\s*{yibo_mishear}", "易唔易泊"),
    )

    text = original
    changed = False
    for pattern, repl in replacements:
        new_text, n = re.subn(pattern, repl, text, flags=re.IGNORECASE)
        if n:
            text = new_text
            changed = True

    if not changed or text == original:
        return original, False
    return text, True


def fee_value_missing(value) -> bool:
    """收費欄是否視為無資料。"""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return text in FEE_MISSING_TOKENS


def first_available_fee(values) -> str | None:
    """由一組收費值取第一個有效者。"""
    for value in values:
        if not fee_value_missing(value):
            return str(value).strip()
    return None


def top_n_exceeds_available(
    requested_top_n: int,
    available_count: int,
    *,
    used_default: bool = False,
) -> bool:
    """用戶明確要求 N 個，但查詢範圍內可用車場少過 N。"""
    if used_default:
        return False
    try:
        req = int(requested_top_n)
        avail = int(available_count)
    except (TypeError, ValueError):
        return False
    return req > 0 and avail >= 0 and avail < req


def format_top_n_shortfall_notice(
    requested_top_n: int,
    available_count: int,
    place_label: str = "",
    *,
    style: str = "written",
) -> str:
    """要求數量超過實際可用時的提示（written／spoken）。"""
    try:
        req = int(requested_top_n)
        avail = int(available_count)
    except (TypeError, ValueError):
        return ""
    place = (place_label or "").strip() or "該範圍"
    if style == "spoken":
        return (
            f"你想查詢嘅地方冇{req}個停車場咁多，"
            f"「{place}」而家得{avail}個，以下係呢度全部停車場。"
        )
    return (
        f"你想查詢嘅地方沒有 {req} 個停車場那麼多；"
        f"「{place}」目前共有 {avail} 個，以下列出該範圍全部停車場。"
    )


def pin_venue_parks_into_ranking(
    at_place: pd.DataFrame,
    nearby_ranked: pd.DataFrame,
    top_n: int,
    *,
    id_col: str = "park_id",
    vacancy_col: str = "預測空位",
) -> pd.DataFrame:
    """把「該場所」車場置頂，再用附近排名補滿至 top_n。"""
    if top_n <= 0:
        return nearby_ranked.iloc[0:0].copy() if nearby_ranked is not None else pd.DataFrame()
    if at_place is None or at_place.empty:
        return nearby_ranked.copy() if nearby_ranked is not None else pd.DataFrame()

    venue = at_place.copy()
    if vacancy_col in venue.columns:
        venue = venue.sort_values(vacancy_col, ascending=False, kind="mergesort")
    venue = venue.reset_index(drop=True)

    seen = set(venue[id_col].astype(str)) if id_col in venue.columns else set()
    extras = (
        nearby_ranked.copy()
        if nearby_ranked is not None and not nearby_ranked.empty
        else pd.DataFrame(columns=venue.columns)
    )
    if id_col in extras.columns and seen:
        extras = extras[~extras[id_col].astype(str).isin(seen)]

    # 該場所車場全數保留（通常只有數個）；其餘用附近排名補位
    # 丟掉舊名次欄，交由呼叫端 add_dense_ranks 重算
    venue = venue.drop(columns=[""], errors="ignore")
    extras = extras.drop(columns=[""], errors="ignore")
    need = max(int(top_n) - len(venue), 0)
    parts = [venue]
    if need > 0 and not extras.empty:
        parts.append(extras.head(need))
    return pd.concat(parts, ignore_index=True)


def order_parks_for_fee_display(
    parks_df: pd.DataFrame,
    *,
    at_place_ids: set[str] | None = None,
    id_col: str = "park_id",
    fee_col: str = "收費",
) -> pd.DataFrame:
    """收費問句／地標：該場所有費 → 該場無費 → 其他有費 → 其他無費。"""
    if parks_df is None or parks_df.empty or fee_col not in parks_df.columns:
        return parks_df.copy() if parks_df is not None else pd.DataFrame()

    work = parks_df.copy().reset_index(drop=True)
    has_fee = ~work[fee_col].map(fee_value_missing)
    if at_place_ids and id_col in work.columns:
        at_mask = work[id_col].astype(str).isin({str(x) for x in at_place_ids})
        parts = [
            work[at_mask & has_fee],
            work[at_mask & ~has_fee],
            work[~at_mask & has_fee],
            work[~at_mask & ~has_fee],
        ]
    else:
        parts = [work[has_fee], work[~has_fee]]
    return pd.concat(parts, ignore_index=True)


def filter_parks_with_fee(
    parks_df: pd.DataFrame,
    *,
    fee_col: str = "收費",
) -> pd.DataFrame:
    """只保留有收費資料嘅車場。"""
    if parks_df is None or parks_df.empty or fee_col not in parks_df.columns:
        return parks_df.copy() if parks_df is not None else pd.DataFrame()
    work = parks_df.copy()
    return work[~work[fee_col].map(fee_value_missing)].reset_index(drop=True)


def filter_parks_without_fee(
    parks_df: pd.DataFrame,
    *,
    fee_col: str = "收費",
) -> pd.DataFrame:
    """只保留無收費資料嘅車場。"""
    if parks_df is None or parks_df.empty or fee_col not in parks_df.columns:
        return parks_df.copy() if parks_df is not None else pd.DataFrame()
    work = parks_df.copy()
    return work[work[fee_col].map(fee_value_missing)].reset_index(drop=True)


def fee_coverage_counts(
    parks_df: pd.DataFrame,
    *,
    fee_col: str = "收費",
) -> tuple[int, int]:
    """回傳 (有收費資料數, 總數)。"""
    if parks_df is None or parks_df.empty:
        return 0, 0
    total = int(len(parks_df))
    if fee_col not in parks_df.columns:
        return 0, total
    with_fee = int((~parks_df[fee_col].map(fee_value_missing)).sum())
    return with_fee, total


def fee_availability_mode(query: str) -> str | None:
    """收費資料有／無查詢：回傳 'present'｜'missing'｜None。

    「冇／沒有顯示收費」要優先於「顯示收費」判斷，避免誤當成有資料。
    """
    q = re.sub(r"\s+", "", query or "").lower()
    if not q:
        return None
    # 明確問價錢／點收費 → 當收費查詢，唔當「有冇顯示」
    if re.search(
        r"收費如何|收費怎樣|收費怎麼|收費點樣|收費點算|"
        r"幾錢|多少錢|幾多錢|價錢|價格|時租|日泊|夜泊",
        q,
    ):
        return None

    missing_patterns = (
        r"冇顯示收費",
        r"沒有顯示收費",
        r"未有顯示收費",
        r"無顯示收費",
        r"冇收費資料",
        r"沒有收費資料",
        r"未有收費資料",
        r"無收費資料",
        r"冇收費規則",
        r"沒有收費規則",
        r"未有收費規則",
        r"無收費規則",
        r"冇收費資訊",
        r"沒有收費資訊",
        r"未提供收費",
        r"收費未提供",
        r"邊[啲個].*冇(?:顯示)?收費",
        r"邊[啲個].*沒有(?:顯示)?收費",
        r"邊[啲個].*無(?:顯示)?收費",
        r"哪[些個].*沒有(?:顯示)?收費",
        r"哪些.*沒有(?:顯示)?收費",
        r"有邊[啲個].*冇(?:顯示)?收費",
        r"有邊[啲個].*沒有(?:顯示)?收費",
        r"有邊[啲個].*無(?:顯示)?收費",
    )
    if any(re.search(p, q) for p in missing_patterns):
        return "missing"

    present_patterns = (
        r"有顯示收費",
        r"顯示收費",
        r"顯示.*收費",
        r"有收費資料",
        r"有收費規則",
        r"有收費資訊",
        r"有收費顯示",
        r"收費資料",
        r"收費規則",
        r"有冇.*收費.*(資料|顯示|規則|資訊)",
        r"有沒有.*收費.*(資料|顯示|規則|資訊)",
        r"邊[啲個].*有(?:顯示)?收費",
        r"哪[些個].*有(?:顯示)?收費",
        r"哪些.*有(?:顯示)?收費",
        r"有邊[啲個].*(?:有)?(?:顯示)?收費",
    )
    if any(re.search(p, q) for p in present_patterns):
        return "present"
    return None


def looks_like_fee_availability_query(query: str) -> bool:
    """問邊啲車場有／冇顯示收費資料（而非問價錢本身）。"""
    return fee_availability_mode(query) is not None


def looks_like_hard_to_park_ranking_query(query: str) -> bool:
    """問邊啲較難／冇咁易泊（本系統只支援較易泊排名，應攔截提示）。

    「難唔難泊／容唔容易／易唔易」屬鬆緊評估，唔當難泊名單。
    """
    q = re.sub(r"\s+", "", query or "")
    if not q:
        return False
    if re.search(
        r"難唔難|难不难|容唔容易|用唔容易|容不容易|用不容易|"
        r"容易唔容易|容易不容易|易唔易|好唔好泊",
        q,
    ):
        return False
    hard_patterns = (
        r"冇咁容易",
        r"冇咁易",
        r"沒那麼容易",
        r"沒有那麼容易",
        r"不太容易",
        r"不那麼容易",
        r"較難泊",
        r"比较难泊",
        r"比較難泊",
        r"最難泊",
        r"最难泊",
        r"邊[度啲個].*難泊",
        r"哪[裡里些個].*難泊",
        r"哪些.*難泊",
        r"難泊.*邊[度啲個]",
        r"唔好泊",
        r"不好泊",
        r"最唔易泊",
        r"最不易泊",
        r"較唔易泊",
        r"比較唔易泊",
        r"比較難",
        r"較難",
    )
    return any(re.search(p, q) for p in hard_patterns)


_CN_DIGIT = {
    "零": 0,
    "〇": 0,
    "○": 0,
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
}

# 「邊十七個／哪廿一個」等顯示數量用；含粵語廿／卅
_CN_COUNT_CHARS = r"一二两三兩四五六七八九十廿卅零〇○壹\d"


def cn_count_token_to_int(token: str) -> int | None:
    """中文／阿拉伯數量詞 → 正整數（約 1–99）；無法解析則 None。

    支援：17、十七、廿、廿一、三十、三十二 等；唔處理「半／正」等時間用語。
    """
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        n = int(token)
        return n if n > 0 else None

    if token in _CN_DIGIT and _CN_DIGIT[token] > 0:
        return _CN_DIGIT[token]
    if token == "十":
        return 10
    if token in ("廿", "二十"):
        return 20
    if token in ("卅", "三十"):
        return 30

    # 十一～十九
    if token.startswith("十") and len(token) == 2 and token[1] in _CN_DIGIT:
        ones = _CN_DIGIT[token[1]]
        return 10 + ones if ones > 0 else None

    # 廿一～廿九／卅一～卅九
    if token.startswith("廿") and len(token) == 2 and token[1] in _CN_DIGIT:
        ones = _CN_DIGIT[token[1]]
        return 20 + ones if ones > 0 else None
    if token.startswith("卅") and len(token) == 2 and token[1] in _CN_DIGIT:
        ones = _CN_DIGIT[token[1]]
        return 30 + ones if ones > 0 else None

    # X十／X十Y（二～九）
    for tens_char, tens_val in (
        ("二", 20),
        ("兩", 20),
        ("两", 20),
        ("三", 30),
        ("四", 40),
        ("五", 50),
        ("六", 60),
        ("七", 70),
        ("八", 80),
        ("九", 90),
    ):
        prefix = tens_char + "十"
        if token == prefix:
            return tens_val
        if token.startswith(prefix) and len(token) == len(prefix) + 1:
            ones = _CN_DIGIT.get(token[-1])
            if ones and ones > 0:
                return tens_val + ones

    return None


def parse_top_n_from_query(query: str) -> int | None:
    """從查詢解析要顯示幾個停車場；找不到則回傳 None。"""
    m = re.search(rf"[邊哪]([{_CN_COUNT_CHARS}]+)個", query or "")
    if m:
        n = cn_count_token_to_int(m.group(1))
        if n is not None and n > 0:
            return n

    # 「邊個／哪個」（未指明數量，視為 1）
    if re.search(r"[邊哪]個", query or ""):
        return 1

    return None


def _cn_minute_token_to_int(token: str) -> int | None:
    """中文分鐘詞 → 0–59；無法解析則 None。"""
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        value = int(token)
        return value if 0 <= value <= 59 else None
    if token in ("半",):
        return 30
    if token in ("正", "整"):
        return 0
    if token in ("一刻",):
        return 15
    if token in ("三刻",):
        return 45
    if token in ("廿", "二十"):
        return 20
    if token in ("卅", "三十"):
        return 30
    if token.startswith("廿") and len(token) == 2 and token[1] in _CN_DIGIT:
        return 20 + _CN_DIGIT[token[1]]
    if token.startswith("卅") and len(token) == 2 and token[1] in _CN_DIGIT:
        return 30 + _CN_DIGIT[token[1]]
    if token == "十":
        return 10
    if token.startswith("十") and len(token) == 2 and token[1] in _CN_DIGIT:
        return 10 + _CN_DIGIT[token[1]]
    if token.startswith("二十") and len(token) == 3 and token[2] in _CN_DIGIT:
        return 20 + _CN_DIGIT[token[2]]
    if token.startswith("三十") and len(token) == 3 and token[2] in _CN_DIGIT:
        return 30 + _CN_DIGIT[token[2]]
    if token.startswith("四十") and len(token) == 3 and token[2] in _CN_DIGIT:
        return 40 + _CN_DIGIT[token[2]]
    if token.startswith("五十") and len(token) == 3 and token[2] in _CN_DIGIT:
        return 50 + _CN_DIGIT[token[2]]
    if token in ("四十",):
        return 40
    if token in ("五十",):
        return 50
    if token in _CN_DIGIT:
        return _CN_DIGIT[token]
    return None


def extract_explicit_clock_minute(query: str) -> int | None:
    """若句子寫明具體分鐘，回傳 0–59；未寫分鐘（如「四點」）則 None。

    「四點半／4:30／4:00」會回傳 30／30／0；「4:45／四點九／九點二十七分」回傳實際分鐘。
    """
    q = re.sub(r"\s+", "", query or "")
    if not q:
        return None

    m = re.search(r"(?<!\d)([0-2]?\d)[:：]([0-5]?\d)", q)
    if m:
        minute = int(m.group(2))
        return minute if 0 <= minute <= 59 else None

    # 點半／時半
    if re.search(r"[點时時]半", q):
        return 30

    # 點一刻／三刻
    m = re.search(r"[點时時](一刻|三刻)", q)
    if m:
        return 15 if m.group(1) == "一刻" else 45

    # 點 + 數字／中文分鐘（可選「分」）
    m = re.search(
        r"[點时時]([0-5]?\d|[一二兩三两四五六七八九十廿卅零〇○]+|[正整])分?",
        q,
    )
    if not m:
        return None
    return _cn_minute_token_to_int(m.group(1))


def is_half_hour_minute(minute: int | None) -> bool:
    """分鐘是否為半小時單位（0 或 30）。"""
    if minute is None:
        return True
    try:
        return int(minute) in (0, 30)
    except (TypeError, ValueError):
        return False


def query_specifies_non_half_hour_time(query: str) -> bool:
    """用戶查詢時間唔係半小時單位（例如 4:45、四點九、九點二十七分）。"""
    minute = extract_explicit_clock_minute(query)
    if minute is None:
        return False
    return not is_half_hour_minute(minute)


def build_fee_park_id_aliases(
    parks_df: pd.DataFrame,
    fee_ids: set[str] | frozenset[str],
    *,
    max_distance_km: float,
    normalize_name: Callable[[str], str],
    haversine_km: Callable[[float, float, float, float], float] | None = None,
    id_col: str = "park_id",
    name_col: str = "name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> dict[str, str]:
    """
    將無收費規則嘅 park_id 映射到名稱＋距離等效、且有收費規則嘅 id。
    有收費嘅 id 映射到自己。
    """
    fee_ids = {str(x) for x in fee_ids}
    aliases = {fid: fid for fid in fee_ids}
    if parks_df is None or parks_df.empty or not fee_ids:
        return aliases

    work = parks_df.dropna(subset=[lat_col, lon_col]).copy()
    if work.empty:
        return aliases
    work[id_col] = work[id_col].astype(str)
    work["_norm"] = work[name_col].map(normalize_name)

    fee_rows = work[work[id_col].isin(fee_ids)]
    if fee_rows.empty:
        return aliases

    def _dist(lat1, lon1, lat2, lon2) -> float:
        if haversine_km is not None:
            return float(haversine_km(lat1, lon1, lat2, lon2))
        return _haversine_km_fallback(lat1, lon1, lat2, lon2)

    for _, row in work.iterrows():
        pid = str(row[id_col])
        if pid in fee_ids:
            continue
        norm = row["_norm"]
        if not norm:
            continue
        lat1, lon1 = float(row[lat_col]), float(row[lon_col])
        best_id = None
        best_key = None
        for _, fee_row in fee_rows.iterrows():
            if fee_row["_norm"] != norm:
                continue
            dist = _dist(lat1, lon1, float(fee_row[lat_col]), float(fee_row[lon_col]))
            if dist > max_distance_km:
                continue
            fid = str(fee_row[id_col])
            # 偏好純數字 id
            key = (0 if fid.isdigit() else 1, dist, fid)
            if best_key is None or key < best_key:
                best_key = key
                best_id = fid
        if best_id:
            aliases[pid] = best_id
    return aliases


def _haversine_km_fallback(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def normalize_park_name_basic(name: str) -> str:
    """同 app._normalize_park_name 對齊嘅基本正規化（測試／別名用）。"""
    s = (name or "").strip().lower()
    s = re.sub(r"[\s\-_/·．.]+", "", s)
    s = s.replace("停車場", "").replace("車場", "")
    s = re.sub(r"parking|carpark", "", s, flags=re.I)
    return s


def park_name_matches_place_tokens(
    park_name: str,
    tokens: list[str] | tuple[str, ...],
    *,
    normalize_name: Callable[[str], str] | None = None,
) -> bool:
    """車場名稱是否命中地點 token（名稱／別名匹配；唔靠距離）。"""
    norm_fn = normalize_name or normalize_park_name_basic
    name_l = (park_name or "").lower()
    name_n = norm_fn(park_name or "")
    for token in tokens:
        t = (token or "").strip().lower()
        if len(t) < 2:
            continue
        if t in name_l or t in name_n:
            return True
    return False


def coerce_park_id(value) -> str:
    """統一 park_id 為字串，避免 int／str key 對唔上。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def merge_baseline_packs(
    primary: tuple,
    extra: tuple,
) -> tuple:
    """合併兩套 baseline dict（後者補前者缺 key；適用 parquet 缺部分 park）。"""
    if not extra or not any(extra):
        return primary
    if not primary or not any(primary):
        return extra
    merged = []
    for left, right in zip(primary, extra):
        out = dict(left or {})
        out.update(right or {})
        merged.append(out)
    return tuple(merged)


def parks_present_in_baselines(packed: tuple) -> set[str]:
    """由 baseline pack 抽出已有資料嘅 park_id。"""
    have: set[str] = set()
    if not packed:
        return have
    by_slot_dow, by_slot_weekend, by_slot, by_park = packed[:4]
    have.update(str(k) for k in (by_park or {}))
    for key in by_slot_dow or {}:
        have.add(str(key[0]))
    for key in by_slot_weekend or {}:
        have.add(str(key[0]))
    for key in by_slot or {}:
        have.add(str(key[0]))
    return have


def build_equivalent_park_id_map(
    parks_df: pd.DataFrame,
    *,
    max_distance_km: float,
    normalize_name: Callable[[str], str],
    haversine_km: Callable[[float, float, float, float], float] | None = None,
    id_col: str = "park_id",
    name_col: str = "name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> dict[str, tuple[str, ...]]:
    """
    名稱＋距離等效車場：park_id -> 其他等效 id（唔包自己）。
    用於 baseline／收費互相借資料（運輸署雙編號）。
    """
    if parks_df is None or parks_df.empty:
        return {}
    work = parks_df.dropna(subset=[lat_col, lon_col]).copy()
    if work.empty:
        return {}
    work = work.reset_index(drop=True)
    work[id_col] = work[id_col].map(coerce_park_id)
    work["_norm"] = work[name_col].map(normalize_name)
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

    def _dist(lat1, lon1, lat2, lon2) -> float:
        if haversine_km is not None:
            return float(haversine_km(lat1, lon1, lat2, lon2))
        return _haversine_km_fallback(lat1, lon1, lat2, lon2)

    lats = work[lat_col].astype(float).tolist()
    lons = work[lon_col].astype(float).tolist()
    norms = work["_norm"].tolist()
    for i in range(n):
        if not norms[i]:
            continue
        for j in range(i + 1, n):
            if norms[i] != norms[j]:
                continue
            if _dist(lats[i], lons[i], lats[j], lons[j]) <= max_distance_km:
                union(i, j)

    clusters: dict[int, list[str]] = {}
    ids = work[id_col].tolist()
    for i in range(n):
        clusters.setdefault(find(i), []).append(ids[i])

    out: dict[str, tuple[str, ...]] = {}
    for members in clusters.values():
        uniq = []
        seen = set()
        for pid in members:
            if pid and pid not in seen:
                seen.add(pid)
                uniq.append(pid)
        if len(uniq) < 2:
            continue
        for pid in uniq:
            out[pid] = tuple(x for x in uniq if x != pid)
    return out
