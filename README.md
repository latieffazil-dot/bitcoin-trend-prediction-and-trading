# 🤖 AI-Driven Cryptocurrency Algorithmic Trading Bot

An end-to-end automated paper-trading system designed to execute short-term cryptocurrency trades (`BTC-USD`) on 15-minute candle intervals. The platform leverages a deep neural network engine built with TensorFlow and Keras to output multi-variable market direction probabilities while applying institutional risk management frameworks.

---

## 🚀 System Architecture & Core Features

### 1. Deep Learning Predictive Engine
* **Model Framework:** Built a Sequential Deep Neural Network (DNN) utilizing dense fully-connected layers to output a Sigmoid-activated probability ($P(\text{up})$) representing whether the next price candle will close higher than the current one.
* **Feature Engineering Pipeline:** Custom pipeline extracting multi-dimensional financial metrics from the Yahoo Finance API (`yfinance`), mapping raw metrics next to key technical indicators:
  * **Trend & Momentum:** 9 & 21 Simple Moving Averages (SMA), 50 & 200 Exponential Moving Averages (EMA), and a custom Relative Strength Index (RSI-14) routine[cite: 2].
  * **Market Dynamics:** Calculated asset volatility windows, volume metrics, and trailing percentage changes[cite: 2].
* **Continuous Online Re-training:** Programmed an autonomous checkpoint loop checking system states to trigger model and scaler re-fitting every 24 hours, actively fighting market feature drift[cite: 2].

### 2. Execution Strategy & Risk Mitigation
* **Confidence Threshold Bands:** Designed execution filters that ignore weak signals, triggering BUY orders only at $>60\%$ confidence and closing trades below $<40\%$ confidence[cite: 2].
* **Trend Filtering:** Integrates an EMA 50/200 crossover confirmation logic that acts as a guardrail, blocking long entries when the macro market trends downwards[cite: 2].
* **Risk Management Matrix:** Hardcoded capital preservation modules including a severe **2% Stop-Loss** and a **4% Take-Profit target** checked before every execution loop[cite: 2].

### 3. Production & Live Simulation Infrastructure
* **State Persistence Layer:** Leverages `joblib` binary serialization to dynamically cache financial ledgers, open trading positions, and technical training scalers across system restarts[cite: 2].
* **Real-World Calibration:** Tracks chronological prediction accuracies independently from historical backtests using a time-matched query queue to compare old forecast logs against actual candle arrivals[cite: 2].
* **Live Dashboard UI:** Implemented a real-time tracking interface utilizing `matplotlib.gridspec` to display current pricing lines, colored prediction histories, and an account ledger simultaneously[cite: 2].

---

## 🛠️ Technology Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras[cite: 2]
* **Data Processing & Analytics:** Pandas, NumPy, Scikit-Learn (`StandardScaler`)[cite: 2]
* **Persistence & Serialization:** Joblib[cite: 2]
* **Visualization Engine:** Matplotlib (`gridspec`)[cite: 2]

---

## 📁 Repository Map
* `crypto_brain_v3.py` - Core application module containing the live ingest engine, DNN training routines, backtest logic, and execution dashboard[cite: 2].
* `README.md` - Technical specification and architecture breakdown.
