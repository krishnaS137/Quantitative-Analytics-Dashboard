# Real-Time Quantitative Analytics Platform

## Overview
This project is a real-time quantitative analytics dashboard built as part of a Quant Developer evaluation assignment. It ingests live tick-level market data from Binance via WebSocket, processes and resamples the data, computes statistical arbitrage analytics, and visualizes results through an interactive dashboard.

The system is designed as a lightweight prototype reflecting the tooling used in market microstructure and statistical arbitrage research.

---

## Features

### Data Ingestion
- Live WebSocket ingestion from Binance (Spot market)
- Tick-level data: timestamp, symbol, price, quantity
- In-memory buffering for low-latency access

### Sampling & Processing
- Resampling into selectable timeframes:
  - 1 second
  - 1 minute
  - 5 minutes

### Quantitative Analytics
- Hedge Ratio estimation using OLS regression
- Spread computation
- Rolling Z-score
- Augmented Dickey-Fuller (ADF) test for stationarity
- Real-time alerting when Z-score crosses threshold

### Frontend Dashboard
- Built using Streamlit and Plotly
- Interactive controls:
  - Asset selection
  - Timeframe selection
  - Rolling window size
  - Z-score alert threshold
  - ADF test trigger
- Zoom, pan, and hover-enabled plots

### Data Export
- Download processed analytics as CSV

### CSV Upload
- Upload historical OHLC / price CSV
- Automatically detects numeric price columns
