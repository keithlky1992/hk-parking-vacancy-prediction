import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

st.title("香港停車場空位預測")
st.write("觀塘區 + 沙田區｜私家車")

hour = st.slider("選擇小時", 0, 23, 19)
dow = st.selectbox("星期", ["週一", "週二", "週三", "週四", "週五", "週六", "週日"], index=0)
dow_num = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"].index(dow)

if st.button("查詢較易泊車的地方"):
    st.write(f"正在預測 {dow} {hour}:00 ...")
    # 這裡之後會呼叫你的預測函數
    st.info("預測功能將在下一步連接模型")