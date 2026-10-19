import streamlit as st
import sqlite3
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

st.title("香港停車場空位預測")
st.write("觀塘區 + 沙田區｜私家車")

@st.cache_data
def load_data():
    conn = sqlite3.connect("parking.db")
    df = pd.read_sql_query("""
        SELECT v.park_id, v.vacancy, v.collected_at, c.name, c.district
        FROM vacancy_records v
        LEFT JOIN carparks c ON v.park_id = c.park_id
    """, conn)
    conn.close()
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df["hour"] = df["collected_at"].dt.hour
    df["dow"] = df["collected_at"].dt.dayofweek
    return df

df = load_data()

hour = st.slider("選擇小時", 0, 23, 19)
dow = st.selectbox("星期", ["週一", "週二", "週三", "週四", "週五", "週六", "週日"], index=0)
dow_num = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"].index(dow)

if st.button("查詢較易泊車的地方"):
    with st.spinner("正在預測..."):
        # 簡單模型訓練（每次查詢時訓練，數據量小可接受）
        le = LabelEncoder()
        df["park_code"] = le.fit_transform(df["park_id"])
        
        features = ["hour", "dow", "park_code"]
        X = df[features]
        y = df["vacancy"]
        
        model = GradientBoostingRegressor(random_state=42, n_estimators=50)
        model.fit(X, y)
        
        # 預測所有停車場
        results = []
        for park_id in df["park_id"].unique():
            park_code = le.transform([park_id])[0]
            pred = model.predict([[hour, dow_num, park_code]])[0]
            info = df[df["park_id"] == park_id].iloc[0]
            results.append({
                "名稱": info["name"],
                "地區": info["district"],
                "預測空位": max(0, round(pred))
            })
        
        result_df = pd.DataFrame(results).sort_values("預測空位", ascending=False).head(10)
        st.success(f"{dow} {hour}:00 較易泊車的停車場（前 10 名）")
        st.dataframe(result_df, use_container_width=True)