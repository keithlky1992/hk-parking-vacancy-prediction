import streamlit as st
import sqlite3
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

st.title("香港停車場空位預測")
st.write("全港 18 區")

@st.cache_data
def load_data():
    conn = sqlite3.connect("parking.db")
    df = pd.read_sql_query("""
        SELECT v.park_id, v.vacancy, v.collected_at, c.name, c.district, c.latitude, c.longitude
        FROM vacancy_records v
        LEFT JOIN carparks c ON v.park_id = c.park_id
    """, conn)
    conn.close()
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df["hour"] = df["collected_at"].dt.hour
    df["dow"] = df["collected_at"].dt.dayofweek
    return df

df = load_data()

hour = st.slider("查詢時間", 0, 23, 19)
dow = st.selectbox("星期", ["週一", "週二", "週三", "週四", "週五", "週六", "週日"], index=0)
district = st.selectbox("選擇地區", ["全部"] + sorted(df["district"].dropna().unique().tolist()))
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
        st.success(f"{dow} {hour}:00 {district} 較易泊車的停車場")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # # 加入地圖
        # st.subheader("地圖位置")
        # map_df = result_df.merge(
        #     df[["name", "latitude", "longitude"]].drop_duplicates(),
        #     left_on="名稱",
        #     right_on="name",
        #     how="left"
        # )
        # map_df = map_df.dropna(subset=["latitude", "longitude"])
        # if not map_df.empty:
        #     st.map(map_df[["latitude", "longitude"]], size=20)
        
        # # 加入橫向長條圖
        # st.subheader("預測空位比較")
        # chart_df = result_df.set_index("名稱")["預測空位"].sort_values()
        # st.bar_chart(chart_df, horizontal=True)
        
        # 加入簡單建議
        if not result_df.empty:
            st.subheader("建議")
            top = result_df.iloc[0]
            name = top["名稱"]
            district_name = top["地區"]
            vacancy = top["預測空位"]
            
            if vacancy > 50:
                suggestion = f"根據目前預測，{district_name}的「{name}」在 {hour}:00 左右預計仍有較多空位（約 {vacancy} 個），相對容易找到車位，建議可優先考慮前往。"
            elif vacancy > 10:
                suggestion = f"{district_name}的「{name}」預計約有 {vacancy} 個空位。雖然不算很多，但仍有機會，建議盡快前往確認。"
            else:
                suggestion = f"目前預測 {district_name}的「{name}」空位較少（約 {vacancy} 個），可能較難泊車。建議考慮清單中的其他停車場，或稍後再查詢。"
            
            st.write(suggestion)