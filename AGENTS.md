# Agent Instructions

## Language

Always communicate with the user in Traditional Chinese (繁體中文).

## Product query scope（泊車查詢範圍）

兩類意圖要分開處理（適用所有地點，唔限單一商場例子）：

1. **該場所容唔容易泊**（如「YOHO mall／APM／彩雲邨…聽朝十點半容唔容易／易唔易／有冇位」）  
   → 只顯示／朗讀**該場所**停車場空位，唔好攤開附近其他場做「邊度較易泊」排名。

2. **附近邊度較易泊**（如「天水圍今晚八點邊度較易泊」「邊啲／邊三個」）  
   → 先定位再按附近車場做空位排名。

廣域地區名（觀塘、天水圍、元朗等）即使講「容唔容易」，仍用附近排名，避免只剩地心附近幾場。
