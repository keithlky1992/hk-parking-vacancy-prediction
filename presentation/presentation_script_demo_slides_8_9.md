# Demo Script · Slides 8–9（中英對照）· 約 3–3.5 分鐘

**產品：** 「咪再兜」  
**連結：** https://hk-parking-vacancy-prediction.streamlit.app/  
**用法：** 每句英文下一行中文；邊講邊操作。括號為現場動作提示。  
**示範重點：** 只示範語音輸入（先說明亦可手動打字）。  
**流程：** Slide 8 過渡句見 `presentation_script_slides_1_8.md`；本檔由 live demo 開始；Demo 完先翻去 Slide 9 畀聽眾掃 QR。

**建議語音查詢：**
- `天水圍今晚八點邊度較易泊？`
- 備援文字（語音失敗時貼上同一句）

---

## 1. 輸入方式說明（約 30 秒）

This is our app 「咪再兜」— it helps drivers stop circling for a parking space.  
呢個係我哋嘅 app「咪再兜」——幫司機唔使再兜圈搵車位。

Drivers can ask in Cantonese for a future time, and get ranked parking suggestions.  
司機能用粵語查詢未來時段，並取得排序泊車建議。

〔指著輸入框與「語音」掣〕

You can type manually in the query box, or use voice input with the microphone.  
你可以在查詢欄手動打字，或用麥克風語音輸入。

Now we’ll demonstrate voice input.  
現在我們示範語音輸入。

---

## 2. 語音示範（約 100 秒）

〔按「語音」；清楚說：`天水圍今晚八點邊度較易泊？`；再按停止〕

Keith’ll say in Cantonese: “Where is it easier to park in Tin Shui Wai around 8 tonight?”  
Keith 會用粵語說：「天水圍今晚八點邊度較易泊？」

Please wait a moment — the computer is loading.  
現在請稍等一下，電腦 load 緊。

After the query finishes, voice playback starts automatically — useful while driving.  
喺查詢完結後，會有語音自動播出——開車時很有用。

You can also replay anytime, or stop while it’s playing.  
你亦可以隨時播多次或播放時停止。

Now let me briefly walk through the flow.  
而家簡單講吓個流程。

First, Speech Recognition turns the audio into text.  
首先，語音識別把聲音轉成文字。

Then DeepSeek parses place and time; Google Geocoding locates the place.  
接著 DeepSeek 抽出地點與時間；Google Geocoding 定位該地。

We score nearby parks with live data plus our vacancy prediction.  
我們用即時數據加上空位預測，為附近車場評分。

Here are the ranked recommendations: park names, distance, predicted vacancies, and often fees.  
這裡是排序建議：車場名稱、距離、預測空位，通常也有收費。

On the map, parks sit near the destination, so drivers can trade off walking distance versus more spaces.  
地圖上的車場都在目的地附近，司機可在「行近啲」與「多啲位」之間取捨。

---

## 3. 收結 + QR（約 30 秒）· 翻去 Slide 9

So in one voice query, we connect language understanding, maps, prediction, and recommendation.  
因此一次語音查詢，就串起語言理解、地圖、預測與建議。

〔翻去 Slide 9 · Try it yourself〕

It is live on Streamlit Cloud — if you’re interested in this app, please scan the QR code and try it yourself.  
它已在 Streamlit Cloud 上線——如果對呢個app有興趣，歡迎掃描 QR code 親自試用。

That’s the end of our presentation — thank you.  
呢個就係我哋 Presentation 嘅結尾——謝謝。

---

## 時間總覽（約 3–3.5 分鐘）

| 段落 | 內容 | 建議時間 |
|------|------|----------|
| 1 | 說明可打字或語音 | ~30 秒 |
| 2 | 語音示範＋解讀結果 | ~100 秒 |
| 3 | 收結＋翻 Slide 9 出 QR | ~30 秒 |
| **合計** | | **約 2.5–3.0 分鐘**（不含 Slide 8 過渡句） |

---

## Demo 應急備註

- **網路慢／首次載入：** 說「首次載入可能需數秒」，邊等邊講 pipeline。
- **語音失敗：** 改貼同一句文字；說明「嘈雜環境下文字是備援」。
- **粵語播報被擋／無聲：** 結果出現後本應自動朗讀；若無聲，按一次「🔊 播放」；或只展示文字建議。
- **時間偏緊：** 結果出現後只指排名與地圖各一句，略過粵語播報；仍記得 Demo 完先出 QR。
- **QR 時機：** Demo 完、翻到 Slide 9 先邀請下載／試用。
