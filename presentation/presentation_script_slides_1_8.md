# Presentation Script · Slides 1–8（中英對照）· 約 7 分鐘

**範圍：** Slide 1–7 簡報內容 + Slide 8 Demo 過渡句（不含詳細 Demo／QR）。  
**用法：** 每句英文下一行中文；講一種語言約 7 分鐘。  
**對應簡報：** `Group4_HK_Parking_Vacancy_Prediction.pptx`（共 9 頁；詳細 Demo／Slide 9 QR 見 `presentation_script_demo_slides_8_9.md`）

---

## Slide 1 — Title（約 40 秒）

Good afternoon, everyone. We are Group 4.  
大家好，我們是第四組。

I’m Keith, and this is Kelvin.  
我是 Keith，旁邊是 Kelvin。

Our capstone project is Hong Kong Parking Vacancy Prediction.  
我們的畢業專題是香港停車場空位預測。

We call the product 「咪再兜」— stop circling for parking.  
產品名稱叫「咪再兜」——意思是：唔使再兜圈搵車位。

We predict private-car vacancies 30 to 60 minutes ahead.  
我們預測未來 30 到 60 分鐘的私家車空位。

We combine official open data, Machine Learning, and an AI Agent app.  
我們結合官方開放數據、機器學習，以及一個 AI Agent 應用。

---

## Slide 2 — Problem & Research Question（約 70 秒）

Finding parking in Hong Kong is time-consuming and stressful.  
在香港找車位既費時又令人有壓力。

Drivers often circle without knowing where spaces will open next.  
司機經常兜圈，卻不知道下一刻哪裡會有空位。

Official vacancy data already exists from the Transport Department.  
運輸署其實已有官方空位數據。

But that data is hard to use in real time while driving.  
但開車時很難即時使用這些數據。

Most tools only show “now,” not what will happen in half an hour.  
多數工具只顯示「現在」，不會告訴你半小時後會怎樣。

Drivers need forward-looking guidance.  
司機需要的是前瞻指引。

So our research question is:  
因此我們的研究問題是：

Can we predict whether a car park will have private-car vacancies — classification —  
我們能否預測某停車場會否有私家車空位——這是分類——

and how many vacancies — regression — 30 to 60 minutes ahead?  
以及會有多少個空位——這是迴歸——在未來 30 到 60 分鐘？

And can Machine Learning beat a same-weekday, same-hour historical baseline?  
機器學習又能否勝過「同一星期幾、同一小時」的歷史基準？

If we beat that baseline, the prediction is useful beyond simple averages.  
若能勝過該基準，預測就比單純平均更有實用價值。

---

## Slide 3 — Dataset Description（約 55 秒）

Here is our data scope.  
以下是我們的數據範圍。

We cover about 592 car parks across roughly 18 Hong Kong districts.  
我們涵蓋全港約 592 個停車場，覆蓋約 18 個地區。

We built about 34.9 thousand historical snapshots over one year.  
我們建立了約 3.49 萬筆歷史快照，為期一年。

Collection runs about every 10 minutes.  
收集節奏大約每 10 分鐘一次。

Main sources are the Transport Department vacancy API for live spaces and fees,  
主要數據源是運輸署空位 API——提供即時空位與收費——

and the Hong Kong Observatory Weather API for temperature and humidity.  
以及香港天文台天氣 API——提供溫度與濕度。

We store data in Supabase Postgres.  
資料存放在 Supabase Postgres。

On the product side, we use DeepSeek for intent, Google Maps for location,  
產品端我們用 DeepSeek 做意圖理解、Google Maps 做定位、

Speech Recognition for voice in, and Edge Text-to-Speech for Cantonese out.  
語音識別做輸入，以及 Edge 文字轉語音做粵語播報。

Key features include hour, day of week, weekend, park, district, and temperature.  
主要特徵包括小時、星期幾、是否週末、車場、地區與溫度。

We focus on car-park vacancy, not on-street parking meters.  
我們聚焦停車場空位，不包含路旁咪錶。

---

## Slide 4 — Methodology & Pipeline（約 70 秒）

Our pipeline has five stages: data sources, ETL, storage, models, and the Streamlit app.  
我們的流程分為五步：數據源、ETL、儲存、模型，以及 Streamlit 應用。

We collect Transport Department vacancy and Observatory weather about every 10 minutes.  
我們大約每 10 分鐘收集運輸署空位與天文台天氣。

Then we join and store them in SQLite locally and Supabase in the cloud.  
再合併，並存到本地 SQLite 與雲端 Supabase。

For modeling, we start with a strong baseline:  
建模方面，我們先建立一個強基準：

the mean vacancy for the same park, same hour, and same weekday,  
同一車場、同一小時、同一星期幾的平均空位，

built from 365 days of history.  
來自 365 天歷史。

Then we train Gradient Boosting models.  
接著我們訓練 Gradient Boosting 模型。

A regressor predicts vacancy count; a classifier predicts whether the park has vacancy.  
迴歸預測空位數量；分類預測是否有空位。

We use a time-based 70/30 train-test split, so there is no future leakage.  
我們採用按時間的 70/30 切分，避免未來資訊洩漏。

Finally, the GenAI layer turns predictions into natural-language answers for drivers.  
最後，GenAI 層把預測變成司機可用的自然語言答案。

DeepSeek parses intent; Maps handles location; speech supports hands-busy driving.  
DeepSeek 解析意圖；Maps 處理位置；語音支援開車時使用。

---

## Slide 5 — EDA Insights（約 55 秒）

From exploratory analysis, clear patterns appear by district and hour.  
從探索性分析可見，地區與小時有明顯規律。

We look at capacity-weighted vacancy rate by district times hour.  
我們檢視「地區 × 小時」的容量加權空置率。

Midday is tighter; night is freer — especially in Central Business District areas.  
中午較緊；晚上較鬆——尤其在商業核心區。

Central and Western, Yau Tsim Mong, and Wan Chai show lower average vacancy rates.  
中西區、油尖旺與灣仔的平均空置率較低。

Some New Territories and Islands districts stay freer on average.  
部分新界與離島地區平均仍較寬鬆。

These patterns justify using time and location features in the models.  
這些規律支持我們在模型中使用時間與地點特徵。

Parking vacancy in Hong Kong is not random — it has structure.  
香港停車場空位並非隨機——它有結構。

That structure is what Machine Learning can learn beyond a simple average.  
正是這結構，讓機器學習能在簡單平均之外學到訊號。

---

## Slide 6 — Analysis & Model Results（約 80 秒）

Now to the model results — the key evidence for our research question.  
接著看模型結果——回答研究問題的關鍵證據。

Gradient Boosting beats the same-hour historical baseline.  
Gradient Boosting 勝過同小時的歷史基準。

On regression, mean absolute error improves from about 17.8 vacancies to 10.40.  
迴歸方面，平均絕對誤差由約 17.8 個空位改善至 10.40。

That is a substantial reduction in average error.  
這是平均誤差的明顯下降。

On classification, F1 is approximately 0.957.  
分類方面，F1 約為 0.957。

So the model is strong at predicting whether a park has private-car vacancy.  
也就是說，預測「有沒有私家車空位」的表現很強。

On the right, you can see a Friday 7 p.m. district ranking example.  
右邊可見一個星期五晚上 7 時的地區排名例子。

That ranking helps drivers decide where to head next — not only whether a space exists.  
這類排名幫助司機決定下一步往哪裡——不只告訴你有沒有位。

Overall, Machine Learning adds real lift over a strong, interpretable baseline.  
總體而言，機器學習相對一個已很強、可解釋的基準，仍帶來實質提升。

That answers our research question in the affirmative.  
這肯定地回答了我們的研究問題。

---

## Slide 7 — Challenges, Limitations & Future Work（約 55 秒）

We also want to be honest about limits.  
我們也想誠實說明限制。

Early live collection had gaps when the machine slept.  
早期即時收集在電腦休眠時會有缺口。

We cover car parks only, not on-street meters.  
我們只涵蓋停車場，不包含路旁咪錶。

Weather signals were collected but are still underused in the models.  
天氣訊號已收集，但模型中仍使用不足。

The early training window was short; later we expanded with one-year baselines.  
早期訓練視窗較短；其後我們用一年基準擴大歷史基礎。

Cloud disk limits also constrained raw historical imports.  
雲端磁碟限制也制約了原始歷史資料匯入。

For future work: cloud cron for 24-by-7 collection,  
未來工作包括：雲端 cron 做全天候收集、

richer weather and holiday features, persisted models,  
更豐富的天氣與假日特徵、模型持久化、

vacancy level classes, stronger Large Language Model suggestions, and deeper error analysis.  
空位等級分類、更強的 Large Language Model 建議，以及更深入的誤差分析。

---

## Slide 8 — Product Demo — 咪再兜（約 10 秒）

Next, we will demo the live product — 「咪再兜」.  
接下來我們會演示線上產品——「咪再兜」。

〔一句帶過；隨即切去 App 做 live demo。詳細 Demo 講稿見 `presentation_script_demo_slides_8_9.md`；QR 喺 Slide 9。〕

---

## 時間總覽（約 7 分鐘）

| 投影片    | 主題           | 建議時間             |
| ------ | ------------ | ---------------- |
| 1      | Title        | ~40 秒            |
| 2      | Problem & RQ | ~70 秒            |
| 3      | Dataset      | ~55 秒            |
| 4      | Methodology  | ~70 秒            |
| 5      | EDA          | ~55 秒            |
| 6      | Results      | ~80 秒            |
| 7      | Challenges   | ~55 秒            |
| 8      | Demo 過渡     | ~10 秒            |
| **合計** |              | **約 7.0–7.2 分鐘** |

**提示：** 若時間偏緊，可壓縮 Slide 3（數據源只點 TD／HKO／Supabase）與 Slide 7（Future work 只講 2 點）。  
若時間偏鬆，可在 Slide 6 多停 5–10 秒指圖。
