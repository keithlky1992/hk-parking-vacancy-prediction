"""語音誤辨校正單元測試。"""

from __future__ import annotations

import unittest

from parking_result_utils import normalize_asr_query_text


class AsrNormalizeTest(unittest.TestCase):
    def test_yong_to_rong_ease(self):
        text, changed = normalize_asr_query_text("YOHO mall 聽朝十點半用唔容易泊")
        self.assertEqual(text, "YOHO mall 聽朝十點半容唔容易泊")
        self.assertTrue(changed)

    def test_simplified_yong_bu_rongyi(self):
        text, changed = normalize_asr_query_text("彩雲邨用不容易泊車")
        self.assertEqual(text, "彩雲邨容不容易泊車")
        self.assertTrue(changed)

    def test_unchanged_when_already_correct(self):
        text, changed = normalize_asr_query_text("APM 容唔容易泊")
        self.assertEqual(text, "APM 容唔容易泊")
        self.assertFalse(changed)

    def test_yibo_number_mishear(self):
        text, changed = normalize_asr_query_text("天水圍邊度200")
        self.assertEqual(text, "天水圍邊度易泊")
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
