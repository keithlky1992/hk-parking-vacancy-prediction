"""全港十八區實用子區對照與停車場指派。"""

from __future__ import annotations

import math
import sqlite3

# 十八區 → 子區顯示名 → 匹配關鍵字（同區內最長優先）
DISTRICT_SUBDISTRICTS: dict[str, dict[str, list[str]]] = {
    "中西區": {
        "中環": ["中環", "愛丁堡廣場", "金融街", "大會堂", "天星"],
        "上環": ["上環", "林士街"],
        "西營盤": ["西營盤", "中山紀念"],
        "堅尼地城": ["堅尼地城", "西市街"],
        "金鐘": ["金鐘"],
        "山頂": ["山頂"],
    },
    "灣仔區": {
        "灣仔": ["灣仔", "皇后大道東", "港灣道", "胡忠", "合和", "新鴻基中心"],
        "銅鑼灣": ["銅鑼灣", "希慎", "利園", "告士打道", "天后"],
        "跑馬地": ["跑馬地", "黃泥涌", "禮頓"],
    },
    "東區": {
        "筲箕灣": ["筲箕灣", "耀東", "愛東", "興東"],
        "柴灣": ["柴灣", "環翠", "興華", "永泰道"],
        "小西灣": ["小西灣", "翠灣"],
        "北角": ["北角"],
        "鰂魚涌": ["鰂魚涌", "太古"],
    },
    "南區": {
        "香港仔": ["香港仔", "石排灣", "瀑布灣"],
        "黃竹坑": ["黃竹坑", "海洋公園"],
        "數碼港": ["數碼港", "鋼綫灣"],
        "赤柱": ["赤柱"],
        "深水灣": ["深水灣"],
        "鴨脷洲": ["鴨脷洲"],
        "華富": ["華富"],
    },
    "油尖旺區": {
        "尖沙咀": ["尖沙咀", "圓方", "戲曲中心", "柯士甸"],
        "旺角": ["旺角", "MOKO", "新世紀廣場", "始創", "西洋菜"],
        "油麻地": ["油麻地", "海庭"],
        "西九龍": ["西九", "高鐵", "九龍站", "M+", "藝術公園"],
    },
    "深水埗區": {
        "深水埗": ["深水埗", "南昌", "石硤尾"],
        "長沙灣": ["長沙灣", "元州", "海麗"],
        "荔枝角": ["荔枝角"],
    },
    "九龍城區": {
        "啟德": ["啟德", "啟晴", "德朗", "體育園", "沐虹", "承啟"],
        "何文田": ["何文田", "愛民", "佛光街"],
        "紅磡": ["紅磡"],
        "土瓜灣": ["土瓜灣", "馬頭圍"],
        "九龍塘": ["九龍塘", "九龍仔"],
    },
    "黃大仙區": {
        "黃大仙": ["黃大仙", "竹園", "雙鳳", "東頭"],
        "彩雲／牛池灣": [
            "彩雲(二)邨",
            "彩雲邨",
            "彩雲商場",
            "彩雲A",
            "彩雲B",
            "牛池灣公園",
            "牛池灣",
            "彩雲",
        ],
        "鑽石山": ["鑽石山", "鳳德"],
        "新蒲崗": ["新蒲崗", "AIRSIDE", "寶光", "五芳街"],
        "樂富": ["樂富", "橫頭磡"],
        "慈雲山": ["慈雲山"],
    },
    "觀塘區": {
        "九龍灣": ["九龍灣", "德福", "淘大", "宏照", "宏泰", "Manhattan Place"],
        "牛頭角": ["牛頭角", "彩德", "彩盈", "彩福"],
        # 不用裸「觀塘」（觀塘道橫跨多處）；用地標／街道
        "觀塘": ["巧明", "海濱道", "Landmark East", "綠景", "宏利廣場", "觀塘碼頭", "開源道", "鴻圖道", "成業街"],
        "藍田": ["藍田", "匯景", "德田", "廣田"],
        "油塘": ["油塘", "油麗", "大本型", "高超"],
        "秀茂坪": ["秀茂坪", "寶達"],
        "順利": ["順利"],
        "茶果嶺": ["茶果嶺"],
    },
    "荃灣區": {
        "荃灣": ["荃灣", "如心", "悅來", "海盛路", "城門谷", "蕙荃"],
        "梨木樹": ["梨木樹"],
        "象山": ["象山"],
    },
    "屯門區": {
        # 不用裸「屯門」；以街道／地標為主，其餘最近鄰
        "屯門市中心": ["屯門市中心", "河田街", "新安街", "青賢街", "海華路", "屯門政府", "屯門公園", "屯門市廣場", "屯門站", "杯渡路", "屯喜路"],
        "良景／建生": ["良景", "建生", "田景", "良運"],
        "大興": ["大興"],
        "蝴蝶灣": ["蝴蝶", "湖翠"],
        "三聖": ["三聖"],
        "藍地": ["藍地", "富泰", "屯貴"],
        "掃管笏": ["掃管笏"],
        "兆康": ["兆康", "兆翠", "恆富"],
    },
    "元朗區": {
        "天水圍": ["天水圍", "天晴", "天華", "天澤", "天逸", "天盛", "天耀", "天慈", "濕地公園", "T Town"],
        # 不用裸「元朗」（常與天水圍等同現）；未命中再靠最近鄰或別名查詢
        "元朗市中心": ["元朗市中心", "元朗民政", "青山公路(元朗", "青山公路－元朗", "元朗廣場", "元朗劇院"],
        "朗屏": ["朗屏"],
        "洪水橋": ["洪水橋"],
        "屏山／廈村": ["屏山", "廈村", "屏廈"],
        "錦田": ["錦田", "石崗"],
        "新田／落馬洲": ["新田", "落馬洲"],
        "流浮山": ["流浮山"],
    },
    "北區": {
        "上水": ["上水", "彩園", "寶石湖"],
        "粉嶺": ["粉嶺", "祥華", "聯和墟", "和興"],
        "打鼓嶺": ["打鼓嶺", "香園圍"],
        "沙頭角": ["沙頭角"],
    },
    "大埔區": {
        "大埔市中心": ["大埔市中心", "大埔", "廣福", "太和", "富善", "大元", "富亨", "寶鄉", "汀角"],
        "大美督": ["大美督"],
        "西貢北": ["西貢北"],
    },
    "西貢區": {
        "將軍澳": ["將軍澳", "調景嶺", "彩明", "健明", "唐明", "TKO", "寶林", "欣明", "英明", "坑口", "尚德", "翠林"],
        # 「將軍澳」較長，同現時會優先；只寫西貢者歸西貢市
        "西貢市": ["西貢市", "西貢苑", "西貢政府", "西貢"],
        "清水灣": ["清水灣"],
    },
    "沙田區": {
        # 不用裸「沙田」（地址常寫「新界沙田大圍…」）；未命中再靠最近鄰
        "沙田": ["沙田市中心", "小瀝源", "愉翠", "廣源", "文禮", "沙田正街", "沙田鄉事會"],
        "大圍": ["大圍", "顯徑", "隆亨", "新翠", "美林", "車公廟", "田心"],
        "火炭": ["火炭", "山尾街", "穗禾"],
        "馬鞍山": ["馬鞍山", "頌安", "恆安", "耀安", "西沙路"],
    },
    "葵青區": {
        "葵涌": ["葵涌", "葵芳", "葵盛", "荔景", "賢麗", "華荔", "寧峰"],
        "青衣": ["青衣", "長發", "長安", "青華", "長宏"],
    },
    "離島區": {
        "東涌": ["東涌", "逸東", "富東", "東薈城", "文東路"],
        "機場": ["機場", "暢達路", "暢順路", "翔天"],
        "大嶼山": ["大嶼山", "迪士尼", "昂坪"],
        "長洲": ["長洲"],
        "坪洲": ["坪洲"],
        "南丫島": ["南丫"],
    },
}

# 使用者口語／簡稱 → 正式子區名（跨區時靠 resolve 優先子區精確名）
LOCATION_ALIASES: dict[str, str] = {
    "元朗": "元朗市中心",
    "屏山": "屏山／廈村",
    "廈村": "屏山／廈村",
    "新田": "新田／落馬洲",
    "落馬洲": "新田／落馬洲",
    "良景": "良景／建生",
    "建生": "良景／建生",
    "蝴蝶": "蝴蝶灣",
    "屯門": "屯門市中心",
    "大埔": "大埔市中心",
    "西貢": "西貢市",
    "將軍澳": "將軍澳",
    "TKO": "將軍澳",
    "調景嶺": "將軍澳",
    "彩雲": "彩雲／牛池灣",
    "彩雲邨": "彩雲／牛池灣",
    "牛池灣": "彩雲／牛池灣",
}


def all_sub_district_names() -> set[str]:
    names: set[str] = set()
    for subs in DISTRICT_SUBDISTRICTS.values():
        names.update(subs.keys())
    return names


def all_district_names() -> set[str]:
    return set(DISTRICT_SUBDISTRICTS.keys())


def sub_district_prompt_summary() -> str:
    """給 LLM 的子區摘要（按十八區）。"""
    lines = []
    for district, subs in DISTRICT_SUBDISTRICTS.items():
        lines.append(f"- {district}：{'、'.join(subs.keys())}")
    return "\n".join(lines)


def _match_sub_district(district: str, text: str) -> str | None:
    """在指定十八區內，以最長關鍵字匹配子區。"""
    subs = DISTRICT_SUBDISTRICTS.get(district) or {}
    if not text or not subs:
        return None
    best_name = None
    best_len = -1
    for name, keywords in subs.items():
        for kw in keywords:
            if kw and kw in text and len(kw) > best_len:
                best_len = len(kw)
                best_name = name
    return best_name


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def ensure_carpark_sub_districts(db_path: str = "parking.db") -> int:
    """確保 carparks.sub_district 存在並填滿；回傳更新列數。"""
    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(carparks)")}
        if "sub_district" not in cols:
            conn.execute("ALTER TABLE carparks ADD COLUMN sub_district TEXT")
            conn.commit()

        rows = conn.execute(
            "SELECT park_id, name, address, district, latitude, longitude, sub_district "
            "FROM carparks"
        ).fetchall()

        assigned: dict[str, str] = {}
        pending: list[tuple] = []

        for park_id, name, address, district, lat, lon, existing in rows:
            district = district or ""
            text = f"{name or ''} {address or ''}"
            matched = _match_sub_district(district, text)
            if matched:
                assigned[park_id] = matched
            else:
                pending.append((park_id, district, lat, lon, existing))

        # 未命中：同區內最近已標註車場
        labeled_by_district: dict[str, list[tuple[str, float, float]]] = {}
        for park_id, name, address, district, lat, lon, _existing in rows:
            if park_id not in assigned:
                continue
            if lat is None or lon is None:
                continue
            labeled_by_district.setdefault(district or "", []).append(
                (assigned[park_id], float(lat), float(lon))
            )

        for park_id, district, lat, lon, existing in pending:
            if existing and existing in (DISTRICT_SUBDISTRICTS.get(district) or {}):
                assigned[park_id] = existing
                continue
            best = None
            best_dist = float("inf")
            if lat is not None and lon is not None:
                for sub_name, plat, plon in labeled_by_district.get(district, []):
                    dist = _haversine_km(float(lat), float(lon), plat, plon)
                    if dist < best_dist:
                        best_dist = dist
                        best = sub_name
            if best is None:
                # 後備：該區第一個子區，或十八區名去掉「區」
                subs = list((DISTRICT_SUBDISTRICTS.get(district) or {}).keys())
                if subs:
                    best = subs[0]
                elif district.endswith("區") and len(district) > 1:
                    best = district[:-1]
                else:
                    best = district or "未知"
            assigned[park_id] = best

        updated = 0
        for park_id, sub in assigned.items():
            conn.execute(
                "UPDATE carparks SET sub_district = ? WHERE park_id = ?",
                (sub, park_id),
            )
            updated += 1
        conn.commit()
        return updated
    finally:
        conn.close()


def normalize_location_intent(raw: str | None) -> tuple[str, str]:
    """
    將意圖地區字串正規化。
    回傳 (match_type, value)：
      - ("all", "全港")
      - ("sub_district", 子區名)
      - ("district", 十八區名)
      - ("unknown", 原字串)
    優先：子區精確名 → 別名 → 十八區。
    """
    if raw is None:
        return "all", "全港"
    text = str(raw).strip()
    if text in ("", "全部", "全港", "香港"):
        return "all", "全港"

    sub_names = all_sub_district_names()
    if text in sub_names:
        return "sub_district", text

    if text in LOCATION_ALIASES:
        return "sub_district", LOCATION_ALIASES[text]

    districts = all_district_names()
    if text in districts:
        return "district", text

    # 「觀塘」→ 先當子區；「觀塘區」已在 districts
    if text + "區" in districts:
        # 若同時是子區名（如沙田、觀塘、荃灣），優先子區
        if text in sub_names:
            return "sub_district", text
        return "district", text + "區"

    # 模糊：關鍵字命中唯一子區
    hits = []
    for district, subs in DISTRICT_SUBDISTRICTS.items():
        for name, keywords in subs.items():
            for kw in keywords:
                if kw == text or text == name:
                    hits.append(name)
                    break
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return "sub_district", hits[0]

    # 長地址／geocode 結果（例如「香港新界天水圍」）內含已知子區名 → 取最長命中
    contained_subs = [n for n in sub_names if n and n in text]
    if contained_subs:
        best = max(contained_subs, key=len)
        return "sub_district", best

    # 「彩雲邨」「牛池灣」等別名出現在長字串中
    contained_aliases = [alias for alias in LOCATION_ALIASES if alias and alias in text]
    if contained_aliases:
        best_alias = max(contained_aliases, key=len)
        return "sub_district", LOCATION_ALIASES[best_alias]

    contained_districts = [n for n in districts if n and n in text]
    if contained_districts:
        return "district", max(contained_districts, key=len)

    return "unknown", text
