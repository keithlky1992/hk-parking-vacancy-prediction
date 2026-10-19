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
district = st.selectbox("選擇地區", ["全部", "觀塘區", "沙田區"])
dow_num = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"].index(dow)

if st.button("查詢較易泊車的地方"):
    with st.spinner("正在預測..."):
        le = LabelEncoder()
        df["park_code"] = le.fit_transform(df["park_id"])
        
        features = ["hour", "dow", "park_code"]
        X = df[features]
        y = df["vacancy"]
        
        model = GradientBoostingRegressor(random_state=42, n_estimators=50)
        model.fit(X, y)
        
        # 根據地區過濾
        if district != "全部":
            df_filtered = df[df["district"] == district]
        else:
            df_filtered = df
        
        results = []
        for park_id in df_filtered["park_id"].unique():
            park_code = le.transform([park_id])[0]
            pred = model.predict([[hour, dow_num, park_code]])[0]
            info = df[df["park_id"] == park_id].iloc[0]
            results.append({
                "名稱": info["name"],
                "地區": info["district"],
                "預測空位": max(0, round(pred))
            })
        
        result_df = pd.DataFrame(results).sort_values("預測空位", ascending=False).head(10)
        st.success(f"{dow} {hour}:00 {district} 較易泊車的停車場（前 10 名）")
        st.dataframe(result_df, use_container_width=True)
        
        # 加入簡單建議
        if not result_df.empty:
            st.subheader("建議")
            top = result_df.iloc[0]
            st.write(f"建議優先考慮 **{top['名稱']}**（{top['地區']}），預計約有 {top['預測空位']} 個空位。")
            
            if top["預測空位"] > 50:
                st.success("空位較充足，較易泊車。")
            elif top["預測空位"] > 10:
                st.info("可能有空位，建議盡快前往。")
            else:
                st.warning("空位可能較少，建議考慮其他選擇。")