# Demo Script · Slide 8（中英對照）· 約 3–3.5 分鐘

**產品：** 「咪再兜」  
**連結：** https://hk-parking-vacancy-prediction.streamlit.app/  
**用法：** 每句英文下一行中文；邊講邊操作。括號為現場動作提示。  
**示範重點：** 只示範語音輸入（先說明亦可手動打字）。

**建議語音查詢：**
- `天水圍今晚八點邊度較易泊？`
- 備援文字（語音失敗時貼上同一句）

---

## 0. 開場與進入 App（約 35 秒）

This is our live product demo — 「咪再兜」.  
這是我們的產品現場演示——「咪再兜」。

Please scan the QR code, or open the Streamlit Cloud link on the slide.  
請掃描 QR code，或打開投影片上的 Streamlit Cloud 連結。

〔打開瀏覽器／投影 App；等頁面載入〕

What you see is the real deployed system, not a mockup.  
大家看到的是已部署的真實系統，不是 mockup。

---

## 1. 輸入方式說明（約 30 秒）

Drivers can ask in Cantonese for a future time, and get ranked parking suggestions.  
司機能用粵語查詢未來時段，並取得排序泊車建議。

〔指著輸入框與「語音」掣〕

You can type manually in the query box, or use voice input with the microphone.  
你可以在查詢欄手動打字，或用麥克風語音輸入。

Today I’ll demonstrate voice input.  
今天我示範語音輸入。

---

## 2. 語音示範（約 100 秒）

〔按「語音」；清楚說：`天水圍今晚八點邊度較易泊？`；再按停止〕

I’ll say in Cantonese: “Where is it easier to park in Tin Shui Wai around 8 tonight?”  
我會用粵語說：「天水圍今晚八點邊度較易泊？」

〔等語音辨識 → 自動或手動查詢；若失敗，改貼同一句文字〕

Speech Recognition turns the audio into text.  
語音識別把聲音轉成文字。

Then DeepSeek parses place and time; Google Geocoding locates the place.  
接著 DeepSeek 抽出地點與時間；Google Geocoding 定位該地。

We score nearby parks with live data plus our vacancy prediction.  
我們用即時數據加上空位預測，為附近車場評分。

〔結果出現後，停 2 秒讓觀眾看〕

Here are the ranked recommendations: park names, distance, predicted vacancies, and often fees.  
這裡是排序建議：車場名稱、距離、預測空位，通常也有收費。

〔指排名／預測空位；再指地圖〕

On the map, parks sit near the destination, so drivers can trade off walking distance versus more spaces.  
地圖上的車場都在目的地附近，司機可在「行近啲」與「多啲位」之間取捨。

〔若自動播粵語，讓它播完一句；否則可按「🔊 播放」〕

The app can also read the suggestion aloud in Cantonese — useful while driving.  
應用亦可用粵語讀出建議——開車時很有用。

---

## 3. 收結（約 25 秒）

So in one voice query, we connect language understanding, maps, prediction, and recommendation.  
因此一次語音查詢，就串起語言理解、地圖、預測與建議。

It is live on Streamlit Cloud — please scan the QR code and try it yourself.  
它已在 Streamlit Cloud 上線——歡迎掃描 QR code 親自試用。

Thank you. We are happy to take questions.  
謝謝。歡迎提問。

---

## 時間總覽（約 3–3.5 分鐘）

| 段落 | 內容 | 建議時間 |
|------|------|----------|
| 0 | 開場／開 App | ~35 秒 |
| 1 | 說明可打字或語音 | ~30 秒 |
| 2 | 語音示範＋解讀結果 | ~100 秒 |
| 3 | 收結 | ~25 秒 |
| **合計** | | **約 3.0–3.5 分鐘** |

---

## Demo 應急備註

- **網路慢／首次載入：** 說「首次載入可能需數秒」，邊等邊講 pipeline。
- **語音失敗：** 改貼同一句文字；說明「嘈雜環境下文字是備援」。
- **粵語播報被擋：** 按一次「🔊 播放」；或只展示文字建議。
- **時間偏緊：** 結果出現後只指排名與地圖各一句，略過粵語播報（約 2.5 分鐘）。
