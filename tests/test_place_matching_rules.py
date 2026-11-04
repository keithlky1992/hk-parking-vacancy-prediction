"""通用地標名稱匹配規則：適用所有商場／邨／地標，唔靠單一例子。"""

from __future__ import annotations

import unittest

from parking_result_utils import (
    coerce_park_id,
    normalize_park_name_basic,
    park_name_matches_place_tokens,
)


def _tokens_like_app(place: str) -> list[str]:
    """精簡版 app._place_match_tokens（測試用；覆蓋主要規則）。"""
    import re

    raw = (place or "").strip()
    compact = re.sub(r"\s+", "", raw).lower()
    stripped = re.sub(
        r"(mall|商場|廣場|中心|center|centre|parking|carpark|停車場)$",
        "",
        compact,
        flags=re.I,
    )
    aliases = {
        "yoho": ("yoho", "形點"),
        "yohomall": ("yoho", "形點"),
        "apm": ("apm",),
        "airside": ("airside",),
        "telford": ("telford", "德福"),
        "telfordplaza": ("telford", "德福"),
        "德福": ("德福", "telford"),
        "德福廣場": ("德福", "telford"),
    }
    tokens: list[str] = []
    if compact:
        tokens.append(compact)
    if stripped and stripped != compact:
        tokens.append(stripped)
    for eng in re.findall(r"[A-Za-z]{2,}", raw):
        tokens.append(eng.lower())
    for key, als in aliases.items():
        if key in compact or compact in key or key in stripped:
            tokens.extend(als)
    for part in re.findall(r"[\u4e00-\u9fff]{2,}", raw):
        tokens.append(part)
        if len(part) >= 3 and part[-1] in "邨苑園城灣莊村":
            tokens.append(part[:-1])
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        t = (t or "").strip().lower()
        if len(t) < 2 or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# (查詢地點, 車場名, 應否視為該場所)
PLACE_PARK_CASES = [
    # 德福：應中自己、唔中隔壁邨
    ("德福廣場", "德福廣場一期停車場", True),
    ("德福廣場", "德福廣場二期停車場", True),
    ("德福廣場", "德福廣場D車場", True),
    ("德福廣場", "牛頭角下邨車場", False),
    ("德福廣場", "牛頭角下邨停車場", False),
    ("德福廣場", "創紀之城 2 & 3期", False),
    # 其他商場／地標
    ("淘大商場", "淘大商場", True),
    ("淘大商場", "德福廣場一期停車場", False),
    ("AIRSIDE", "AIRSIDE", True),
    ("AIRSIDE", "Airside", True),
    ("AIRSIDE", "Mikiki", False),
    ("YOHO mall", "YOHO Mix", True),
    ("YOHO mall", "形點", True),
    ("YOHO mall", "元朗廣場", False),
    ("彩雲邨", "彩雲A停車場", True),
    ("彩雲邨", "彩雲(二)邨停車場", True),
    ("彩雲邨", "坪石邨停車場", False),
    ("apm", "apm 停車場", True),
    ("apm", "泓富廣場停車場", False),
    ("啟晴邨", "啟晴邨停車場", True),
    ("啟晴邨", "德朗邨停車場A", False),
]


class PlaceNameMatchTest(unittest.TestCase):
    def test_generic_venue_vs_neighbor_rules(self):
        for place, park, expect in PLACE_PARK_CASES:
            tokens = _tokens_like_app(place)
            hit = park_name_matches_place_tokens(
                park, tokens, normalize_name=normalize_park_name_basic
            )
            self.assertEqual(
                hit,
                expect,
                msg=f"place={place!r} park={park!r} tokens={tokens} → {hit} (want {expect})",
            )


class ParkIdCoerceTest(unittest.TestCase):
    def test_int_and_str_same_key(self):
        self.assertEqual(coerce_park_id(31), "31")
        self.assertEqual(coerce_park_id("31"), "31")
        self.assertEqual(coerce_park_id("tdc130p1"), "tdc130p1")


if __name__ == "__main__":
    unittest.main()
