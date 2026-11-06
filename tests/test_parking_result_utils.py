"""parking_result_utils 單元測試（唔依賴 Streamlit）。"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from parking_result_utils import (
    build_equivalent_park_id_map,
    build_fee_park_id_aliases,
    cn_count_token_to_int,
    extract_explicit_clock_minute,
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
    normalize_park_name_basic,
    order_parks_for_fee_display,
    parks_present_in_baselines,
    parse_past_relative_calendar_date,
    parse_top_n_from_query,
    pin_venue_parks_into_ranking,
    query_specifies_non_half_hour_time,
    relative_day_offset_from_query,
    resolve_yearless_calendar_date,
    top_n_exceeds_available,
)


class FeeHelpersTest(unittest.TestCase):
    def test_fee_value_missing(self):
        self.assertTrue(fee_value_missing("未提供"))
        self.assertTrue(fee_value_missing(""))
        self.assertTrue(fee_value_missing(None))
        self.assertFalse(fee_value_missing("$22／小時"))

    def test_first_available_fee(self):
        self.assertEqual(
            first_available_fee(["未提供", "$18／小時", "$20／小時"]),
            "$18／小時",
        )
        self.assertIsNone(first_available_fee(["未提供", ""]))


class PinVenueTest(unittest.TestCase):
    def test_pins_venue_parks_first_then_fills(self):
        at_place = pd.DataFrame(
            {
                "park_id": ["31", "32"],
                "名稱": ["德福一期", "德福二期"],
                "預測空位": [5, 8],
                "收費": ["$22／小時", "$22／小時"],
            }
        )
        nearby = pd.DataFrame(
            {
                "park_id": ["x1", "31", "x2", "x3"],
                "名稱": ["遠場A", "德福一期", "遠場B", "遠場C"],
                "預測空位": [99, 5, 58, 40],
                "收費": ["未提供", "$22／小時", "未提供", "未提供"],
            }
        )
        out = pin_venue_parks_into_ranking(at_place, nearby, top_n=4)
        self.assertEqual(out["park_id"].tolist(), ["32", "31", "x1", "x2"])
        self.assertEqual(out.iloc[0]["名稱"], "德福二期")  # 較高空位置頂

    def test_empty_venue_returns_nearby(self):
        nearby = pd.DataFrame({"park_id": ["a"], "預測空位": [1]})
        out = pin_venue_parks_into_ranking(pd.DataFrame(), nearby, 3)
        self.assertEqual(out["park_id"].tolist(), ["a"])


class OrderFeeDisplayTest(unittest.TestCase):
    def test_venue_with_fee_comes_first(self):
        df = pd.DataFrame(
            {
                "park_id": ["a", "31", "b", "32"],
                "收費": ["未提供", "$22／小時", "$10／小時", "未提供"],
            }
        )
        out = order_parks_for_fee_display(df, at_place_ids={"31", "32"})
        self.assertEqual(out["park_id"].tolist(), ["31", "32", "b", "a"])


class FeeAvailabilityQueryTest(unittest.TestCase):
    def test_detects_which_lots_show_fees(self):
        self.assertEqual(
            fee_availability_mode("全香港既停車場有邊啲有顯示收費"),
            "present",
        )
        self.assertTrue(looks_like_fee_availability_query("邊啲停車場有收費資料"))
        self.assertTrue(looks_like_fee_availability_query("觀塘哪些有顯示收費"))

    def test_detects_which_lots_lack_fees(self):
        self.assertEqual(
            fee_availability_mode("全香港有邊啲停車場冇顯示收費"),
            "missing",
        )
        self.assertEqual(
            fee_availability_mode("邊啲停車場沒有收費資料"),
            "missing",
        )
        self.assertTrue(
            looks_like_fee_availability_query("全香港有邊啲停車場冇顯示收費")
        )

    def test_price_questions_are_not_availability(self):
        self.assertFalse(looks_like_fee_availability_query("天水圍停車場收費如何"))
        self.assertFalse(looks_like_fee_availability_query("彩雲邨今晚收費幾錢"))
        self.assertFalse(looks_like_fee_availability_query("德福廣場時租"))
        self.assertIsNone(fee_availability_mode("德福廣場時租"))


class NonHalfHourTimeTest(unittest.TestCase):
    def test_rejects_non_half_hour_examples(self):
        cases = {
            "彩雲邨下晝4點45分邊度比較易泊": 45,
            "四點九": 9,
            "4:45": 45,
            "四點四十五分": 45,
            "12:05": 5,
            "九點二十七分": 27,
            "下晝四點一刻": 15,
        }
        for query, minute in cases.items():
            with self.subTest(query=query):
                self.assertEqual(extract_explicit_clock_minute(query), minute)
                self.assertTrue(query_specifies_non_half_hour_time(query))

    def test_allows_half_hour_examples(self):
        ok_queries = [
            "黃大仙下晝四點邊度比較易泊",
            "六點半",
            "十二點",
            "4:30",
            "11:00",
            "下晝三點半邊度比較易泊",
            "今晚八點",
            "彩雲邨附近停車場收費如何",
        ]
        for query in ok_queries:
            with self.subTest(query=query):
                self.assertFalse(query_specifies_non_half_hour_time(query))


class TopNShortfallTest(unittest.TestCase):
    def test_detects_when_request_exceeds_available(self):
        self.assertTrue(top_n_exceeds_available(100, 9))
        self.assertFalse(top_n_exceeds_available(100, 100))
        self.assertFalse(top_n_exceeds_available(100, 120))
        self.assertFalse(top_n_exceeds_available(100, 9, used_default=True))
        self.assertFalse(top_n_exceeds_available(0, 9))

    def test_shortfall_notice_mentions_requested_and_available(self):
        written = format_top_n_shortfall_notice(100, 9, "啟德附近", style="written")
        spoken = format_top_n_shortfall_notice(100, 9, "啟德附近", style="spoken")
        self.assertIn("100", written)
        self.assertIn("9", written)
        self.assertIn("啟德附近", written)
        self.assertIn("冇100個停車場咁多", spoken)
        self.assertIn("9", spoken)


class PastRelativeAndYearlessDateTest(unittest.TestCase):
    def setUp(self):
        self.today = datetime(2026, 11, 6, 17, 50, tzinfo=ZoneInfo("Asia/Hong_Kong"))

    def test_relative_past_offsets(self):
        self.assertEqual(relative_day_offset_from_query("天水圍琴日邊啲易泊"), -1)
        self.assertEqual(relative_day_offset_from_query("天水圍昨日邊啲易泊"), -1)
        self.assertEqual(relative_day_offset_from_query("天水圍前日邊啲易泊"), -2)
        self.assertEqual(relative_day_offset_from_query("天水圍大前日邊啲易泊"), -3)
        self.assertEqual(relative_day_offset_from_query("天水圍聽日邊啲易泊"), 1)
        self.assertIsNone(relative_day_offset_from_query("天水圍邊啲易泊"))

    def test_parse_past_relative_calendar_date(self):
        self.assertEqual(
            parse_past_relative_calendar_date("天水圍琴日兩點半", self.today).date(),
            datetime(2026, 11, 5).date(),
        )
        self.assertEqual(
            parse_past_relative_calendar_date("天水圍前日", self.today).date(),
            datetime(2026, 11, 4).date(),
        )
        self.assertEqual(
            parse_past_relative_calendar_date("天水圍大前日", self.today).date(),
            datetime(2026, 11, 3).date(),
        )
        self.assertIsNone(
            parse_past_relative_calendar_date("天水圍聽日", self.today)
        )

    def test_yearless_current_and_prev_month_use_this_year(self):
        # 今日 11/6：今個月／上個月 → 2026（即使日子已過）
        nov5 = resolve_yearless_calendar_date(11, 5, today=self.today)
        self.assertEqual(nov5.date(), datetime(2026, 11, 5).date())
        self.assertLess(nov5.date(), self.today.date())
        oct1 = resolve_yearless_calendar_date(10, 1, today=self.today)
        self.assertEqual(oct1.date(), datetime(2026, 10, 1).date())

    def test_yearless_earlier_months_use_next_year(self):
        # 1–9 月（早於上個月）→ 2027
        self.assertEqual(
            resolve_yearless_calendar_date(9, 30, today=self.today).date(),
            datetime(2027, 9, 30).date(),
        )
        self.assertEqual(
            resolve_yearless_calendar_date(1, 3, today=self.today).date(),
            datetime(2027, 1, 3).date(),
        )

    def test_yearless_later_month_this_year(self):
        # 12 月仍係今年未到
        dt = resolve_yearless_calendar_date(12, 1, today=self.today)
        self.assertEqual(dt.date(), datetime(2026, 12, 1).date())

    def test_yearless_january_previous_month_is_last_december(self):
        jan = datetime(2027, 1, 10, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        dt = resolve_yearless_calendar_date(12, 25, today=jan)
        self.assertEqual(dt.date(), datetime(2026, 12, 25).date())

    def test_explicit_year_honoured(self):
        dt = resolve_yearless_calendar_date(11, 5, today=self.today, year=2027)
        self.assertEqual(dt.date(), datetime(2027, 11, 5).date())


class CnCountAndTopNParseTest(unittest.TestCase):
    def test_cn_count_token_to_int(self):
        cases = {
            "17": 17,
            "十七": 17,
            "十": 10,
            "十一": 11,
            "廿": 20,
            "廿一": 21,
            "二十": 20,
            "三十二": 32,
            "兩": 2,
            "九十九": 99,
        }
        for token, expected in cases.items():
            with self.subTest(token=token):
                self.assertEqual(cn_count_token_to_int(token), expected)

    def test_parse_top_n_arabic_and_chinese_seventeen(self):
        self.assertEqual(
            parse_top_n_from_query("啟德邊17個停車場比較易泊"),
            17,
        )
        self.assertEqual(
            parse_top_n_from_query("啟德邊十七個停車場比較易泊"),
            17,
        )

    def test_parse_top_n_common_forms(self):
        self.assertEqual(parse_top_n_from_query("邊三個好泊"), 3)
        self.assertEqual(parse_top_n_from_query("哪兩個停車場"), 2)
        self.assertEqual(parse_top_n_from_query("邊個最易泊"), 1)
        self.assertIsNone(parse_top_n_from_query("天水圍今晚邊度較易泊"))


class HardToParkRankingQueryTest(unittest.TestCase):
    def test_detects_not_so_easy_queries(self):
        self.assertTrue(
            looks_like_hard_to_park_ranking_query("啟德有邊啲停車場冇咁容易泊車")
        )
        self.assertTrue(looks_like_hard_to_park_ranking_query("觀塘邊啲難泊"))
        self.assertTrue(looks_like_hard_to_park_ranking_query("天水圍較難泊嘅停車場"))

    def test_ease_assessment_is_not_hard_ranking(self):
        self.assertFalse(looks_like_hard_to_park_ranking_query("YOHO mall 容唔容易泊"))
        self.assertFalse(looks_like_hard_to_park_ranking_query("APM 易唔易泊"))
        self.assertFalse(looks_like_hard_to_park_ranking_query("彩雲邨難唔難泊"))
        self.assertFalse(looks_like_hard_to_park_ranking_query("天水圍今晚八點邊度較易泊"))


class FilterParksWithFeeTest(unittest.TestCase):
    def test_keeps_only_rows_with_fee(self):
        df = pd.DataFrame(
            {
                "park_id": ["a", "b", "c"],
                "收費": ["未提供", "$18／小時", ""],
            }
        )
        out = filter_parks_with_fee(df)
        self.assertEqual(out["park_id"].tolist(), ["b"])
        self.assertEqual(fee_coverage_counts(df), (1, 3))

    def test_keeps_only_rows_without_fee(self):
        df = pd.DataFrame(
            {
                "park_id": ["a", "b", "c"],
                "收費": ["未提供", "$18／小時", ""],
            }
        )
        out = filter_parks_without_fee(df)
        self.assertEqual(out["park_id"].tolist(), ["a", "c"])


class FeeAliasTest(unittest.TestCase):
    def test_maps_tdc_twin_to_fee_id(self):
        parks = pd.DataFrame(
            {
                "park_id": ["100", "tdc50p8", "999"],
                "name": ["泓富廣場停車場", "泓富廣場停車場", "其他"],
                "latitude": [22.3106, 22.3106, 22.4],
                "longitude": [114.2257, 114.2257, 114.1],
            }
        )
        aliases = build_fee_park_id_aliases(
            parks,
            {"100"},
            max_distance_km=0.06,
            normalize_name=normalize_park_name_basic,
        )
        self.assertEqual(aliases["100"], "100")
        self.assertEqual(aliases["tdc50p8"], "100")
        self.assertNotIn("999", aliases)


class BaselineMergeTest(unittest.TestCase):
    def test_merge_and_presence(self):
        primary = ({}, {}, {}, {"tdc1": 10.0}, {}, {}, {}, {"tdc1": 3})
        extra = ({}, {}, {}, {"31": 5.0}, {}, {}, {}, {"31": 8})
        merged = merge_baseline_packs(primary, extra)
        self.assertEqual(merged[3]["tdc1"], 10.0)
        self.assertEqual(merged[3]["31"], 5.0)
        self.assertEqual(parks_present_in_baselines(merged), {"tdc1", "31"})

    def test_equivalent_map_bidirectional(self):
        parks = pd.DataFrame(
            {
                "park_id": ["121", "tdc129p4"],
                "name": ["安基苑停車場", "安基苑車場"],
                "latitude": [22.32, 22.32],
                "longitude": [114.21, 114.21],
            }
        )
        eq = build_equivalent_park_id_map(
            parks,
            max_distance_km=0.06,
            normalize_name=normalize_park_name_basic,
        )
        self.assertIn("tdc129p4", eq["121"])
        self.assertIn("121", eq["tdc129p4"])


if __name__ == "__main__":
    unittest.main()
