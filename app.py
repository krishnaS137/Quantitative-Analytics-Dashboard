import asyncio
import json
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import websockets
from statsmodels.tsa.stattools import adfuller

# -----------------------------
# GLOBAL IN-MEMORY BUFFER
# -----------------------------
BUFFER = deque(maxlen=4000)

# -----------------------------
# BINANCE WEBSOCKET INGESTION
# -----------------------------
async def binance_ws():
    url = "wss://fstream.binance.com/stream?streams=btcusdt@trade/ethusdt@trade"
    while True:
        try:
            async with websockets.connect(url) as ws:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)["data"]
                    BUFFER.append({
                        "ts": data["T"],
                        "symbol": data["s"],
                        "price": float(data["p"]),
                        "qty": float(data["q"])
                    })
        except Exception as e:
            print("WebSocket error, reconnecting:", e)
            time.sleep(2)

def start_ws():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(binance_ws())

threading.Thread(target=start_ws, daemon=True).start()

# -----------------------------
# DATA HELPERS
# -----------------------------
def get_df(symbol):
    buffer_copy = list(BUFFER)  # snapshot

    rows = [x for x in buffer_copy if x["symbol"] == symbol]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df


def resample_price(df, tf):
    rule = {"1s": "1S", "1m": "1T", "5m": "5T"}[tf]
    return df["price"].resample(rule).last().dropna()

# -----------------------------
# ANALYTICS
# -----------------------------
def hedge_ratio(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.params[1]

def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def adf_test(series):
    try:
        return adfuller(series.dropna())[1]
    except:
        return None

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("📈 Real-Time Quantitative Analytics Dashboard")

st.sidebar.header("Controls")
symbol_x = st.sidebar.selectbox("Asset X", ["BTCUSDT", "ETHUSDT"])
symbol_y = st.sidebar.selectbox("Asset Y", ["ETHUSDT", "BTCUSDT"])
tf = st.sidebar.selectbox("Timeframe", ["1s", "1m", "5m"])
window = st.sidebar.slider("Rolling Window", 20, 100, 30)
z_thresh = st.sidebar.slider("Z-Score Alert", 1.0, 3.0, 2.0)
run_adf = st.sidebar.button("Run ADF Test")

uploaded = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])

placeholder = st.empty()

# -----------------------------
# MAIN LIVE LOOP
# -----------------------------
while True:
    if uploaded:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)

        # convert first column to datetime index
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df.set_index(df.iloc[:, 0], inplace=True)

        # keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            st.error("CSV must contain at least two numeric price columns")
            time.sleep(1)
            continue

        px = numeric_df.iloc[:, 0]
        py = numeric_df.iloc[:, 1]
    else:
        df_x = get_df(symbol_x)
        df_y = get_df(symbol_y)

        if df_x.empty or df_y.empty:
            time.sleep(1)
            continue

        px = resample_price(df_x, tf)
        py = resample_price(df_y, tf)

    df_pair = pd.concat([px, py], axis=1).dropna()
    df_pair.columns = ["x", "y"]

    if len(df_pair) < window:
        time.sleep(1)
        continue

    beta = hedge_ratio(df_pair["x"].tail(window), df_pair["y"].tail(window))
    spread = df_pair["y"] - beta * df_pair["x"]
    z = zscore(spread, window)

    with placeholder.container():
        st.subheader("Key Metrics")
        c1, c2 = st.columns(2)
        c1.metric("Hedge Ratio (β)", round(beta, 4))
        c2.metric("Current Z-Score", round(z.iloc[-1], 2))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=z.tail(200), name="Z-Score"))
        fig.add_hline(y=z_thresh, line_color="red")
        fig.add_hline(y=-z_thresh, line_color="red")
        fig.update_layout(title="Z-Score Spread")
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"zplot-{symbol_x}-{symbol_y}-{tf}-{int(time.time())}"
        )


        if abs(z.iloc[-1]) > z_thresh:
            st.error("🚨 Z-Score Alert Triggered")

        if run_adf:
            pval = adf_test(spread)
            st.write("ADF p-value:", pval)

        st.download_button(
            "Download Analytics CSV",
            df_pair.tail(500).to_csv(),
            file_name="analytics.csv",
            key=f"download-{symbol_x}-{symbol_y}-{tf}-{int(time.time())}"
        )

    time.sleep(1)
