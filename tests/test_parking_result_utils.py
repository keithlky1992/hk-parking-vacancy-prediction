"""parking_result_utils 單元測試（唔依賴 Streamlit）。"""

from __future__ import annotations

import unittest

import pandas as pd

from parking_result_utils import (
    build_equivalent_park_id_map,
    build_fee_park_id_aliases,
    fee_availability_mode,
    fee_coverage_counts,
    fee_value_missing,
    filter_parks_with_fee,
    filter_parks_without_fee,
    first_available_fee,
    looks_like_fee_availability_query,
    merge_baseline_packs,
    normalize_park_name_basic,
    order_parks_for_fee_display,
    parks_present_in_baselines,
    pin_venue_parks_into_ranking,
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
