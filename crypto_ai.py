import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
import json
import joblib
import warnings
import requests
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG
# =====================================================================
# Use a path relative to this script by default so it works on any machine.
# If you want a fixed location instead, set an absolute path, e.g.:
#   MODEL_FILE = "C:/Users/Fazil/crypto_brain_v3.keras"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "crypto_brain_v3.keras")
SCALER_FILE = os.path.join(SCRIPT_DIR, "crypto_brain_v3_scaler.joblib")
STATE_FILE = os.path.join(SCRIPT_DIR, "crypto_brain_v3_state.joblib")
ACCOUNT_FILE = os.path.join(SCRIPT_DIR, "crypto_brain_v3_account.joblib")
TRACKER_FILE = os.path.join(SCRIPT_DIR, "crypto_brain_v3_tracker.joblib")
TICKER = "BTC-USD"

INTERVAL = "15m"
HIST_PERIOD = "60d"       # max-ish lookback yfinance allows for 15m candles
LIVE_PERIOD = "10d"       # smaller window pulled every loop for live inference

STARTING_BALANCE = 10000.0

# Feature #6 (updated for classification model): act only when the model's
# predicted UP-probability is confidently above/below 50%, not just barely.
BUY_CONFIDENCE_THRESHOLD = 0.60   # predicted P(up) must exceed this to buy
SELL_CONFIDENCE_THRESHOLD = 0.40  # predicted P(up) must drop below this to sell on signal

# Feature #9: risk management
STOP_LOSS_PCT = 0.02       # -2%
TAKE_PROFIT_PCT = 0.04     # +4%

# Feature #4: periodic retraining
RETRAIN_EVERY_HOURS = 24
RETRAIN_EVERY_N_LOOPS = None   # set an int to retrain every N loops instead/also

# Neural net training params
NN_EPOCHS = 50
NN_BATCH_SIZE = 32

LOOP_SLEEP_SECONDS = 60

# ---- Self-improvement suggestions (read-only; never auto-applied) ----
# After each retrain cycle, the script can ask an LLM to look at recent
# performance and propose a code change. This is written to a markdown
# file for YOU to review - it never edits crypto_brain_v3.py itself.
ENABLE_SELF_IMPROVEMENT_SUGGESTIONS = True
SUGGESTIONS_FILE = os.path.join(SCRIPT_DIR, "suggestions.md")
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"   # set this env var on your machine
ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# =====================================================================
# INDICATORS  (Feature #8)
# =====================================================================
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_features(df):
    df = df.copy()
    df['Day_Minus_1'] = df['Close'].shift(1)
    df['Day_Minus_2'] = df['Close'].shift(2)
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI_14'] = calculate_rsi(df)
    df['Volatility'] = df['Close'].rolling(window=14).std()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    # Kept for human-readable reporting (actual next price, used by the
    # PredictionTracker to show real $ movement) - NOT used as the model
    # training target anymore.
    df['Target_Next_Candle'] = df['Close'].shift(-1)
    # Classification target: did the next candle close higher than this one?
    # 1 = up, 0 = down/flat. This is what the model actually trains on now.
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


FEATURES = [
    'Close', 'Day_Minus_1', 'Day_Minus_2', 'Volume',
    'SMA_9', 'SMA_21', 'EMA_50', 'EMA_200',
    'RSI_14', 'Volatility', 'Price_Change_Pct'
]

# =====================================================================
# TRADING ACCOUNT  (Feature #1, #9, #12)
# =====================================================================
class TradingAccount:
    """
    A single paper-trading account used both for the historical backtest
    and for live simulation. Tracks balance, position, and round-trip
    trade stats (wins/losses) under stop-loss / take-profit rules.
    """

    def __init__(self, starting_balance, stop_loss_pct=None, take_profit_pct=None, label="ACCOUNT"):
        self.label = label
        self.starting_balance = starting_balance
        self.cash = starting_balance
        self.btc = 0.0
        self.holding = False
        self.entry_price = None

        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.total_trades = 0      # completed round-trips
        self.wins = 0
        self.losses = 0
        self.trade_log = []        # list of dicts per completed trade

    # ---- derived stats ----
    @property
    def equity(self):
        """Total account value if we mark-to-market at last known price."""
        if self.holding and self.entry_price is not None:
            return self.cash + self.btc * self.entry_price
        return self.cash

    def equity_at(self, price):
        if self.holding:
            return self.cash + self.btc * price
        return self.cash

    @property
    def profit(self):
        return self.cash - self.starting_balance if not self.holding else None

    def profit_at(self, price):
        return self.equity_at(price) - self.starting_balance

    def roi_at(self, price):
        return (self.profit_at(price) / self.starting_balance) * 100

    @property
    def win_rate(self):
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100

    @property
    def open_trade_pl_pct(self, current_price=None):
        return None  # placeholder, real one computed where price is known

    def open_trade_unrealized(self, current_price):
        """Unrealized P/L % on the currently open position, or None if flat."""
        if not self.holding or self.entry_price is None:
            return None
        return ((current_price - self.entry_price) / self.entry_price) * 100

    # ---- actions ----
    def buy(self, price):
        if self.holding:
            return False
        self.btc = self.cash / price
        self.cash = 0.0
        self.holding = True
        self.entry_price = price
        return True

    def sell(self, price, reason="signal"):
        if not self.holding:
            return False
        proceeds = self.btc * price
        pnl = proceeds - (self.btc * self.entry_price)
        self.cash = proceeds
        self.btc = 0.0
        self.holding = False
        self.total_trades += 1
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self.trade_log.append({
            "entry": self.entry_price,
            "exit": price,
            "pnl": pnl,
            "reason": reason
        })
        self.entry_price = None
        return True

    def check_risk_management(self, current_price):
        """
        Returns 'stop_loss', 'take_profit', or None.
        Call this BEFORE evaluating the model's signal each loop/step.
        """
        if not self.holding or self.entry_price is None:
            return None
        change_pct = (current_price - self.entry_price) / self.entry_price
        if self.stop_loss_pct is not None and change_pct <= -self.stop_loss_pct:
            return "stop_loss"
        if self.take_profit_pct is not None and change_pct >= self.take_profit_pct:
            return "take_profit"
        return None

    def summary_dict(self, current_price):
        return {
            "label": self.label,
            "starting_balance": self.starting_balance,
            "current_balance": self.cash,
            "equity": self.equity_at(current_price),
            "profit_loss": self.profit_at(current_price),
            "roi_pct": self.roi_at(current_price),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "holding": self.holding,
            "btc_held": self.btc,
            "entry_price": self.entry_price,
            "open_trade_pl_pct": self.open_trade_unrealized(current_price),
        }

    # ---- persistence ----
    def to_dict(self):
        """Serialize all state needed to fully restore this account later."""
        return {
            "label": self.label,
            "starting_balance": self.starting_balance,
            "cash": self.cash,
            "btc": self.btc,
            "holding": self.holding,
            "entry_price": self.entry_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "trade_log": self.trade_log,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild an account exactly as it was when to_dict() was called."""
        acc = cls(
            starting_balance=data["starting_balance"],
            stop_loss_pct=data.get("stop_loss_pct"),
            take_profit_pct=data.get("take_profit_pct"),
            label=data.get("label", "ACCOUNT"),
        )
        acc.cash = data["cash"]
        acc.btc = data["btc"]
        acc.holding = data["holding"]
        acc.entry_price = data["entry_price"]
        acc.total_trades = data["total_trades"]
        acc.wins = data["wins"]
        acc.losses = data["losses"]
        acc.trade_log = data.get("trade_log", [])
        return acc


# =====================================================================
# PREDICTION ERROR TRACKER (now direction/confidence based)
# =====================================================================
class PredictionTracker:
    """
    Tracks how often the model's live UP/DOWN predictions have actually
    been correct, candle by candle. This is a real-world accuracy signal,
    distinct from the historical backtest and distinct from the chart.

    Each loop, we have a prediction made FOR a specific future timestamp
    (the next candle after the data we had at the time), based on the
    price AT prediction time. Once that timestamp's actual close is
    available, we compare it to the reference price to get the actual
    direction, and check whether the model's call was right.
    Matching by timestamp (not just "whatever came next") avoids silently
    misaligning predictions if a loop is skipped or data has a gap.
    """

    MAX_HISTORY = 500  # cap memory/disk growth; keeps the most recent N records

    def __init__(self):
        self.pending_timestamp = None      # timestamp the last prediction was FOR
        self.pending_up_prob = None        # predicted P(up) for that timestamp
        self.pending_reference_price = None  # price AT the time the prediction was made
        self.records = []                  # list of dicts: timestamp, up_prob, predicted_dir, actual_dir, correct

    def set_pending(self, timestamp, predicted_up_prob, reference_price):
        self.pending_timestamp = timestamp
        self.pending_up_prob = predicted_up_prob
        self.pending_reference_price = reference_price

    def try_resolve(self, live_data):
        """
        Call this each loop with the freshly fetched live_data. If the
        timestamp we were waiting on is now present, record whether the
        predicted direction was correct, and clear the pending prediction.
        Returns the resolved record dict, or None if nothing was resolved.
        """
        if self.pending_timestamp is None:
            return None
        if self.pending_timestamp not in live_data.index:
            return None  # not available yet (or was skipped) - keep waiting

        actual_price = float(np.squeeze(live_data.loc[self.pending_timestamp, 'Close']))
        reference_price = self.pending_reference_price
        up_prob = self.pending_up_prob

        actual_direction = "UP" if actual_price > reference_price else "DOWN"
        predicted_direction = "UP" if up_prob >= 0.5 else "DOWN"
        correct = actual_direction == predicted_direction

        record = {
            "timestamp": self.pending_timestamp,
            "up_prob": up_prob,
            "predicted_direction": predicted_direction,
            "reference_price": reference_price,
            "actual_price": actual_price,
            "actual_direction": actual_direction,
            "correct": correct,
        }
        self.records.append(record)
        if len(self.records) > self.MAX_HISTORY:
            self.records = self.records[-self.MAX_HISTORY:]

        self.pending_timestamp = None
        self.pending_up_prob = None
        self.pending_reference_price = None
        return record

    def stats(self, window=20):
        """Summary stats over the most recent `window` resolved predictions."""
        recent = self.records[-window:]
        if not recent:
            return None
        n_correct = sum(1 for r in recent if r["correct"])
        avg_confidence = float(np.mean([r["up_prob"] if r["predicted_direction"] == "UP"
                                         else (1 - r["up_prob"]) for r in recent]))
        return {
            "n": len(recent),
            "hit_rate_pct": (n_correct / len(recent)) * 100,
            "n_correct": n_correct,
            "n_wrong": len(recent) - n_correct,
            "avg_confidence_pct": avg_confidence * 100,
        }

    # ---- persistence ----
    def to_dict(self):
        return {
            "pending_timestamp": self.pending_timestamp,
            "pending_up_prob": self.pending_up_prob,
            "pending_reference_price": self.pending_reference_price,
            "records": self.records,
        }

    @classmethod
    def from_dict(cls, data):
        tracker = cls()
        tracker.pending_timestamp = data.get("pending_timestamp")
        tracker.pending_up_prob = data.get("pending_up_prob")
        tracker.pending_reference_price = data.get("pending_reference_price")
        tracker.records = data.get("records", [])
        return tracker


# =====================================================================
# BACKTEST ENGINE  (Feature #3, #6, #7, #9)
# =====================================================================
def run_backtest(price_series, predicted_up_prob_series, ema50_series, ema200_series,
                  starting_balance=10000,
                  buy_confidence=BUY_CONFIDENCE_THRESHOLD, sell_confidence=SELL_CONFIDENCE_THRESHOLD,
                  stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                  use_trend_filter=True):
    """
    price_series, predicted_up_prob_series, ema50_series, ema200_series must
    all be aligned (same index/order). predicted_up_prob_series holds the
    model's predicted probability (0.0-1.0) that the NEXT candle closes
    higher than the current one. Returns (TradingAccount, list_of_equity_curve).
    """
    account = TradingAccount(starting_balance, stop_loss_pct, take_profit_pct, label="BACKTEST")
    equity_curve = []

    n = len(price_series)
    for i in range(n):
        current_price = price_series.iloc[i]
        up_prob = predicted_up_prob_series[i]
        ema50 = ema50_series.iloc[i]
        ema200 = ema200_series.iloc[i]

        # Risk management checks first (stop-loss / take-profit) - Feature #9
        risk_action = account.check_risk_management(current_price)
        if risk_action:
            account.sell(current_price, reason=risk_action)

        uptrend = ema50 > ema200

        # Trend filter - Feature #7: only allow buys in an uptrend
        can_buy = (not use_trend_filter) or uptrend

        # Trend-reversal exit: if we're holding and the trend has flipped
        # against us, exit regardless of what the model predicts.
        trend_reversed_against_position = use_trend_filter and account.holding and not uptrend

        if trend_reversed_against_position:
            account.sell(current_price, reason="trend_reversal")
        elif up_prob > buy_confidence and can_buy and not account.holding:
            account.buy(current_price)
        elif up_prob < sell_confidence and account.holding:
            account.sell(current_price, reason="signal")
        # else: HOLD or WAIT, no action (Feature #6 - confidence band avoids noise trades)

        equity_curve.append(account.equity_at(current_price))

    # Close out any open position at the final price so stats are complete
    if account.holding:
        final_price = price_series.iloc[-1]
        account.sell(final_price, reason="end_of_backtest")
        equity_curve[-1] = account.equity_at(final_price)

    return account, equity_curve


def print_backtest_report(account, final_price):
    s = account.summary_dict(final_price)
    print("\n" + "=" * 46)
    print("📈 HISTORICAL BACKTEST REPORT 📈")
    print("=" * 46)
    print(f"Starting Balance:   ${s['starting_balance']:,.2f}")
    print(f"Final Balance:      ${s['current_balance']:,.2f}")
    pl = s['current_balance'] - s['starting_balance']
    roi = (pl / s['starting_balance']) * 100
    if pl >= 0:
        print(f"Total Profit:       +${pl:,.2f} (+{roi:.2f}%) 🟢")
    else:
        print(f"Total Loss:         -${abs(pl):,.2f} ({roi:.2f}%) 🔴")
    print(f"Total Trades:       {s['total_trades']}")
    print(f"Winning Trades:     {s['wins']}")
    print(f"Losing Trades:      {s['losses']}")
    print(f"Win Rate:           {s['win_rate']:.2f}%")
    print("=" * 46 + "\n")


# =====================================================================
# MODEL TRAINING / PERSISTENCE  (Feature #4, #11) - TensorFlow version
# =====================================================================
def build_model(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # outputs probability 0.0-1.0 that price goes UP
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(X_train, y_train):
    """
    Fits a fresh StandardScaler on X_train and trains a new Keras model
    on the scaled data. Returns (model, scaler) - both are needed for
    every future prediction, so they always travel together.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = build_model(X_train.shape[1])
    model.fit(X_train_scaled, y_train, epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE, verbose=0)

    return model, scaler


def predict_one(model, scaler, X_row):
    """
    X_row: 2D array-like of shape (1, n_features), UNSCALED.
    Scales with the model's own scaler before predicting.
    """
    X_scaled = scaler.transform(X_row)
    return float(model.predict(X_scaled, verbose=0)[0][0])


def predict_many(model, scaler, X):
    """Scales and predicts for a full DataFrame/array of rows at once."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled, verbose=0).reshape(-1)


def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        print(f"Loading existing model from {MODEL_FILE} ...")
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        last_trained = get_state().get("last_trained")
        return model, scaler, last_trained
    else:
        print("No saved model found. Training a new one...")
        model, scaler = train_model(X_train, y_train)
        model.save(MODEL_FILE)              # TensorFlow's way of saving the brain
        joblib.dump(scaler, SCALER_FILE)     # scaler must travel with the model
        now = datetime.now()
        save_state({"last_trained": now})
        return model, scaler, now


def get_state():
    if os.path.exists(STATE_FILE):
        try:
            return joblib.load(STATE_FILE)
        except Exception:
            return {}
    return {}


def save_state(state):
    joblib.dump(state, STATE_FILE)


def load_or_create_account(starting_balance, stop_loss_pct, take_profit_pct, label="LIVE"):
    """
    Feature: persist the live trading account across restarts.
    If a saved account exists, restore it EXACTLY (balance, BTC held,
    open position, trade history, win/loss stats). Otherwise start fresh.
    """
    if os.path.exists(ACCOUNT_FILE):
        try:
            data = joblib.load(ACCOUNT_FILE)
            account = TradingAccount.from_dict(data)
            print(f"Restored live account from {ACCOUNT_FILE}: "
                  f"balance=${account.cash:,.2f}, holding={account.holding}, "
                  f"trades={account.total_trades} (W:{account.wins} L:{account.losses})")
            return account
        except Exception as e:
            print(f"Could not restore saved account ({e}); starting a fresh one.")

    print("No saved live account found. Starting fresh.")
    return TradingAccount(starting_balance, stop_loss_pct, take_profit_pct, label=label)


def save_account(account):
    try:
        joblib.dump(account.to_dict(), ACCOUNT_FILE)
    except Exception as e:
        print(f"Warning: failed to save live account state ({e})")


def load_or_create_tracker():
    if os.path.exists(TRACKER_FILE):
        try:
            data = joblib.load(TRACKER_FILE)
            tracker = PredictionTracker.from_dict(data)
            print(f"Restored prediction tracker from {TRACKER_FILE}: "
                  f"{len(tracker.records)} past predictions on record.")
            return tracker
        except Exception as e:
            print(f"Could not restore saved prediction tracker ({e}); starting fresh.")
    return PredictionTracker()


def save_tracker(tracker):
    try:
        joblib.dump(tracker.to_dict(), TRACKER_FILE)
    except Exception as e:
        print(f"Warning: failed to save prediction tracker state ({e})")


def _gather_self_improvement_context(account, tracker, backtest_stats):
    """
    Collects the real, current performance numbers to hand to the LLM.
    No guessing or fabricated context - only what we can actually measure.
    """
    tracker_stats = tracker.stats(window=20)
    return {
        "backtest": backtest_stats,
        "live_account": {
            "starting_balance": account.starting_balance,
            "current_cash": account.cash,
            "total_trades": account.total_trades,
            "wins": account.wins,
            "losses": account.losses,
            "win_rate_pct": account.win_rate,
            "holding": account.holding,
        },
        "live_prediction_accuracy": tracker_stats,  # may be None if no data yet
        "current_config": {
            "BUY_CONFIDENCE_THRESHOLD": BUY_CONFIDENCE_THRESHOLD,
            "SELL_CONFIDENCE_THRESHOLD": SELL_CONFIDENCE_THRESHOLD,
            "STOP_LOSS_PCT": STOP_LOSS_PCT,
            "TAKE_PROFIT_PCT": TAKE_PROFIT_PCT,
            "RETRAIN_EVERY_HOURS": RETRAIN_EVERY_HOURS,
        },
    }


def _build_suggestion_prompt(context):
    return f"""You are reviewing the performance of a paper-trading bot (no real money,
fake balance) that combines a small neural network price predictor with
fixed, hand-written trading rules (trend filter, threshold, stop-loss/take-profit).

Here is its REAL recent performance data (not hypothetical):

{json.dumps(context, indent=2, default=str)}

Based ONLY on this data, propose exactly ONE specific, focused improvement.

Requirements for your response:
- Identify ONE concrete thing the data suggests is underperforming (e.g. a
  specific config constant, a missing feature, a logic gap) - point to the
  actual number that supports this, don't speculate beyond the data given.
- Propose ONE specific code change to address it. Show it as a small,
  reviewable diff-style snippet (a few lines, clearly marked old vs new),
  not a full file rewrite.
- Explain in 2-3 sentences WHY this change should help, referencing the
  specific number(s) above.
- Explicitly state any risk or downside of making this change.
- Do NOT propose removing or weakening stop-loss, take-profit, or the trend
  filter. Do NOT propose increasing position size or leverage.
- Keep your entire response under 250 words.

This proposal will be read by a human and NEVER auto-applied. Be concrete
and specific, not generic advice."""


def generate_self_improvement_suggestion(account, tracker, backtest_stats):
    """
    Calls the Anthropic API to get ONE proposed code improvement based on
    real recent performance. Writes it to SUGGESTIONS_FILE for human review.
    NEVER modifies this script's own source code - this is read-only output.

    Fails silently (with a printed message) if no API key is configured,
    so this optional feature can never break the actual trading bot.
    """
    if not ENABLE_SELF_IMPROVEMENT_SUGGESTIONS:
        return

    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
    if not api_key:
        print(f"[{time.strftime('%H:%M:%S')}] Self-improvement suggestions skipped: "
              f"no {ANTHROPIC_API_KEY_ENV} environment variable set.")
        return

    context = _gather_self_improvement_context(account, tracker, backtest_stats)
    prompt = _build_suggestion_prompt(context)

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        suggestion_text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        ).strip()

        if not suggestion_text:
            print(f"[{time.strftime('%H:%M:%S')}] Self-improvement suggestion came back empty; skipping write.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n\n---\n\n## Suggestion logged {timestamp}\n\n{suggestion_text}\n"

        with open(SUGGESTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(entry)

        print(f"[{time.strftime('%H:%M:%S')}] 💡 New self-improvement suggestion written to {SUGGESTIONS_FILE}")

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Self-improvement suggestion failed ({e}); continuing without it.")


def maybe_retrain(model, scaler, last_trained, loop_count, X_train, y_train,
                   live_account=None, tracker=None, backtest_stats=None):
    """
    Feature #4: retrain periodically — either every RETRAIN_EVERY_HOURS,
    or every RETRAIN_EVERY_N_LOOPS, whichever is configured.
    Refits a NEW scaler on the new data too (markets drift, so reusing an
    old scaler on a new data range would quietly distort predictions).
    """
    should_retrain = False

    if RETRAIN_EVERY_HOURS is not None and last_trained is not None:
        if datetime.now() - last_trained >= timedelta(hours=RETRAIN_EVERY_HOURS):
            should_retrain = True

    if RETRAIN_EVERY_N_LOOPS is not None and loop_count > 0:
        if loop_count % RETRAIN_EVERY_N_LOOPS == 0:
            should_retrain = True

    if should_retrain:
        print(f"[{time.strftime('%H:%M:%S')}] ⏳ Retraining model on fresh data...")
        new_model, new_scaler = train_model(X_train, y_train)
        new_model.save(MODEL_FILE)
        joblib.dump(new_scaler, SCALER_FILE)
        now = datetime.now()
        save_state({"last_trained": now})
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Retraining complete.")

        # Optional: ask an LLM to review recent performance and propose
        # ONE reviewable code change. Writes to SUGGESTIONS_FILE only -
        # never touches this script. Skips silently if no API key is set.
        if live_account is not None and tracker is not None:
            generate_self_improvement_suggestion(live_account, tracker, backtest_stats)

        return new_model, new_scaler, now

    return model, scaler, last_trained


# =====================================================================
# DATA FETCH / PREP
# =====================================================================
def fetch_and_prepare(period, interval):
    df = yf.download(TICKER, period=period, interval=interval, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = add_features(df)
    df = df.dropna()
    return df


# =====================================================================
# DASHBOARD RENDERING  (Feature #2, #10)
# =====================================================================
def render_dashboard(fig, ax_chart, ax_panel,
                      price_index, actual_prices, predicted_up_probs,
                      live_summary, current_price, predicted_up_prob):

    ax_chart.clear()
    ax_chart.plot(price_index, actual_prices, label="Actual Price", color="blue", linewidth=2, zorder=2)

    # Color each point green/red based on whether the model predicted UP or
    # DOWN at that time, so direction calls are visible directly on the
    # price line (a dashed "predicted price" line no longer makes sense
    # once the model outputs a 0-1 probability rather than a price).
    up_mask = predicted_up_probs >= 0.5
    down_mask = ~up_mask
    if up_mask.any():
        ax_chart.scatter(price_index[up_mask], actual_prices[up_mask],
                          color="green", s=18, label="Model said UP", zorder=3)
    if down_mask.any():
        ax_chart.scatter(price_index[down_mask], actual_prices[down_mask],
                          color="red", s=18, label="Model said DOWN", zorder=3)

    ax_chart.set_title(f"LIVE: {TICKER} - AI Trading Dashboard")
    ax_chart.set_xlabel("Time")
    ax_chart.set_ylabel("Price (USD)")
    ax_chart.grid(True, alpha=0.3)
    for label in ax_chart.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    # Secondary axis: the model's raw predicted UP-probability over time,
    # so you can see confidence trending, not just the binary call.
    ax_prob = ax_chart.twinx()
    ax_prob.plot(price_index, predicted_up_probs, color="purple", alpha=0.4,
                 linewidth=1, linestyle="dotted", label="P(up)")
    ax_prob.axhline(BUY_CONFIDENCE_THRESHOLD, color="green", alpha=0.3, linestyle="--", linewidth=0.8)
    ax_prob.axhline(SELL_CONFIDENCE_THRESHOLD, color="red", alpha=0.3, linestyle="--", linewidth=0.8)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("P(up)", color="purple")
    ax_prob.tick_params(axis='y', labelcolor="purple")

    handles1, labels1 = ax_chart.get_legend_handles_labels()
    handles2, labels2 = ax_prob.get_legend_handles_labels()
    ax_chart.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)

    ax_panel.clear()
    ax_panel.axis("off")

    pl = live_summary["profit_loss"]
    pl_color = "green" if pl >= 0 else "red"
    pos_text = "HOLDING BTC" if live_summary["holding"] else "NO POSITION"
    pos_color = "darkorange" if live_summary["holding"] else "gray"

    lines = [
        ("LIVE PAPER TRADING", "header"),
        ("", "spacer"),
        (f"Starting Balance:  ${live_summary['starting_balance']:,.2f}", "normal"),
        (f"Current Balance:   ${live_summary['current_balance']:,.2f}", "normal"),
        (f"Profit / Loss:     ${pl:,.2f}", pl_color),
        (f"ROI:               {live_summary['roi_pct']:.2f}%", pl_color),
        ("", "spacer"),
        (f"Trades:            {live_summary['total_trades']}", "normal"),
        (f"Wins:              {live_summary['wins']}", "green"),
        (f"Losses:            {live_summary['losses']}", "red"),
        (f"Win Rate:          {live_summary['win_rate']:.2f}%", "normal"),
        ("", "spacer"),
        (f"Position:          {pos_text}", pos_color),
        (f"BTC Held:          {live_summary['btc_held']:.6f}", "normal"),
    ]
    if live_summary["open_trade_pl_pct"] is not None:
        otp = live_summary["open_trade_pl_pct"]
        otp_color = "green" if otp >= 0 else "red"
        lines.append((f"Open Trade P/L:    {otp:.2f}%", otp_color))

    lines.append(("", "spacer"))
    lines.append((f"Current Price:     ${current_price:,.2f}", "normal"))
    direction_text = "UP" if predicted_up_prob >= 0.5 else "DOWN"
    direction_color = "green" if predicted_up_prob >= 0.5 else "red"
    lines.append((f"AI Predicts:       {direction_text} ({predicted_up_prob*100:.1f}% confidence)", direction_color))

    y = 0.97
    for text, style in lines:
        if style == "spacer":
            y -= 0.035
            continue
        if style == "header":
            ax_panel.text(0.02, y, text, fontsize=13, fontweight="bold", transform=ax_panel.transAxes)
        else:
            color = style if style in ("green", "red", "darkorange", "gray") else "black"
            ax_panel.text(0.02, y, text, fontsize=10.5, color=color,
                           family="monospace", transform=ax_panel.transAxes)
        y -= 0.058


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 46)
    print("STARTUP DIAGNOSTICS")
    print(f"Running file:        {os.path.abspath(__file__)}")
    print(f"TensorFlow version:  {tf.__version__}")
    print(f"Model file:          {MODEL_FILE}")
    print(f"Scaler file:         {SCALER_FILE}")
    print(f"State file:          {STATE_FILE}")
    print(f"Account file:        {ACCOUNT_FILE}")
    print(f"Tracker file:        {TRACKER_FILE}")
    print("=" * 46 + "\n")

    print("Fetching historical data...")
    data = fetch_and_prepare(HIST_PERIOD, INTERVAL)

    X = data[FEATURES]
    y = data['Target_Direction']            # classification label: 1=up, 0=down
    y_price = data['Target_Next_Candle']    # actual next price, for reporting/backtest only

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    _, _, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, shuffle=False)

    model, scaler, last_trained = load_or_train_model(X_train, y_train)

    # ---- Diagnostic: show exactly what was loaded for last_trained ----
    # This settles, with certainty, whether the 24h retrain clock is
    # actually persisting across restarts or silently resetting.
    if last_trained is None:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️  No last_trained timestamp found in {STATE_FILE} "
              f"(file missing, unreadable, or key absent).")
        print(f"[{time.strftime('%H:%M:%S')}] Treating model as OVERDUE for retraining (conservative "
              f"fallback) rather than resetting the 24h clock to now.")
        # Conservative fallback: backdate so a retrain is considered due
        # immediately, instead of granting a fresh RETRAIN_EVERY_HOURS
        # grace period the script never earned. This is the fix for the
        # silent-reset issue - a missing/corrupt timestamp no longer buys
        # the model 24 more hours before it's checked again.
        if RETRAIN_EVERY_HOURS is not None:
            last_trained = datetime.now() - timedelta(hours=RETRAIN_EVERY_HOURS)
        else:
            last_trained = datetime.now()
        save_state({"last_trained": last_trained})
    else:
        hours_since = (datetime.now() - last_trained).total_seconds() / 3600
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Loaded last_trained = {last_trained} "
              f"({hours_since:.2f} hours ago).")
        if RETRAIN_EVERY_HOURS is not None:
            hours_remaining = RETRAIN_EVERY_HOURS - hours_since
            if hours_remaining <= 0:
                print(f"[{time.strftime('%H:%M:%S')}] Model is OVERDUE for retraining "
                      f"(retrain cycle is {RETRAIN_EVERY_HOURS}h) - will retrain on the next loop check.")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Next scheduled retrain in ~{hours_remaining:.2f} hours "
                      f"(cycle is every {RETRAIN_EVERY_HOURS}h).")

    # ---- Backtest (Feature #3, #6, #7, #9) ----
    # current_price_series: the price AT THE TIME each prediction was made
    # (aligned with X_test), not the next-candle price - this is what the
    # backtest engine should treat as "now" at each step.
    current_price_series = data.loc[X_test.index, 'Close']
    historical_up_probs = predict_many(model, scaler, X_test)
    test_ema50 = data.loc[X_test.index, 'EMA_50']
    test_ema200 = data.loc[X_test.index, 'EMA_200']

    backtest_account, equity_curve = run_backtest(
        current_price_series, historical_up_probs, test_ema50, test_ema200,
        starting_balance=STARTING_BALANCE
    )
    final_test_price = float(current_price_series.iloc[-1])
    print_backtest_report(backtest_account, final_test_price)

    backtest_stats = {
        "starting_balance": backtest_account.starting_balance,
        "final_balance": backtest_account.cash,
        "total_trades": backtest_account.total_trades,
        "wins": backtest_account.wins,
        "losses": backtest_account.losses,
        "win_rate_pct": backtest_account.win_rate,
    }

    # ---- Live paper trading account (Feature #1, #12) ----
    # Persists across restarts: balance, BTC held, open position, trade history.
    live_account = load_or_create_account(
        STARTING_BALANCE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, label="LIVE"
    )

    # ---- Prediction error tracker (live accuracy diagnostic) ----
    # Persists across restarts: a running log of how far the model's
    # live predictions have actually been from reality.
    tracker = load_or_create_tracker()

    plt.ion()
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax_chart = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])

    loop_count = 0

    try:
        while True:
            loop_count += 1
            print(f"\n[{time.strftime('%H:%M:%S')}] Analyzing live market data... (loop {loop_count})")

            try:
                live_data = fetch_and_prepare(LIVE_PERIOD, INTERVAL)

                # Handle empty downloads safely - Yahoo Finance occasionally
                # returns nothing for a cycle; this is usually transient,
                # not an actual delisting, so just retry next loop.
                if live_data is None or live_data.empty:
                    print("No live market data received. Retrying next loop...")
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                if len(live_data) < 5:
                    print("Not enough live data available. Retrying...")
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

            except Exception as e:
                print(f"Data fetch failed: {e}")
                time.sleep(LOOP_SLEEP_SECONDS)
                continue

            live_X = live_data[FEATURES]

            if live_X.empty:
                print("No valid feature rows available. Retrying next loop...")
                time.sleep(LOOP_SLEEP_SECONDS)
                continue

            # ---- Feature #4: periodic retraining ----
            retrain_X = live_X.iloc[:-1]
            retrain_y = live_data['Target_Next_Candle'].iloc[:-1]
            model, scaler, last_trained = maybe_retrain(
                model, scaler, last_trained, loop_count, retrain_X, retrain_y,
                live_account=live_account, tracker=tracker, backtest_stats=backtest_stats
            )

            # ---- Resolve last loop's pending prediction against reality ----
            # (must happen BEFORE we overwrite predicted_up_prob with the new one)
            resolved = tracker.try_resolve(live_data)
            if resolved:
                result_icon = "✅" if resolved["correct"] else "❌"
                print(f"\n📊 PREDICTION CHECK: for {resolved['timestamp']}, predicted "
                      f"{resolved['predicted_direction']} (P(up)={resolved['up_prob']*100:.1f}%) -> "
                      f"actual was {resolved['actual_direction']} "
                      f"(${resolved['reference_price']:,.2f} -> ${resolved['actual_price']:,.2f}) {result_icon}")
                stats = tracker.stats(window=20)
                if stats:
                    print(f"   Last {stats['n']} predictions -> hit rate: {stats['hit_rate_pct']:.1f}% "
                          f"({stats['n_correct']} correct, {stats['n_wrong']} wrong), "
                          f"avg confidence: {stats['avg_confidence_pct']:.1f}%")

            latest_row = live_X.iloc[-1].values.reshape(1, -1)
            predicted_up_prob = predict_one(model, scaler, latest_row)
            current_price = float(np.squeeze(live_data['Close'].iloc[-1]))
            ema50 = float(np.squeeze(live_data['EMA_50'].iloc[-1]))
            ema200 = float(np.squeeze(live_data['EMA_200'].iloc[-1]))

            # This prediction is FOR the next candle after the latest one we have.
            # Infer the candle spacing from the data itself rather than assuming
            # 15 minutes, so this stays correct even if INTERVAL changes.
            if len(live_data.index) >= 2:
                candle_spacing = live_data.index[-1] - live_data.index[-2]
            else:
                candle_spacing = timedelta(minutes=15)
            target_timestamp = live_data.index[-1] + candle_spacing
            tracker.set_pending(target_timestamp, predicted_up_prob, reference_price=current_price)
            save_tracker(tracker)

            uptrend = ema50 > ema200
            predicted_direction = "UP" if predicted_up_prob >= 0.5 else "DOWN"

            print("-" * 32)
            print(f"CURRENT PRICE:   ${current_price:,.2f}")
            print(f"AI PREDICTION:   {predicted_direction}  (P(up)={predicted_up_prob*100:.1f}%)")
            print(f"TREND (EMA50/200): {'UP' if uptrend else 'DOWN'}")

            # ---- Risk management first (Feature #9) ----
            risk_action = live_account.check_risk_management(current_price)
            if risk_action:
                live_account.sell(current_price, reason=risk_action)
                label = "STOP-LOSS" if risk_action == "stop_loss" else "TAKE-PROFIT"
                print(f"\n🚨 RISK MANAGEMENT TRIGGERED: {label} HIT — POSITION CLOSED 🚨")

            # ---- Trading signal (Feature #6, #7, #13) ----
            print("\n🚨 TRADING SIGNAL 🚨")
            can_buy = uptrend  # Feature #7: only buy in an uptrend

            # Trend-reversal exit: if we're holding and the trend has
            # flipped against us, exit regardless of what the model
            # predicts. Otherwise a stale/wrong prediction can keep
            # firing HOLD indefinitely through a confirmed downtrend.
            trend_reversed_against_position = live_account.holding and not uptrend

            if trend_reversed_against_position:
                live_account.sell(current_price, reason="trend_reversal")
                print("SIGNAL: SELL")
                print("  -> Trend flipped to DOWN while holding. Exiting regardless of prediction.")
            elif predicted_up_prob > BUY_CONFIDENCE_THRESHOLD and can_buy and not live_account.holding:
                live_account.buy(current_price)
                print("SIGNAL: BUY")
                print(f"  -> AI is {predicted_up_prob*100:.1f}% confident of a rise and trend confirms uptrend.")
            elif predicted_up_prob > BUY_CONFIDENCE_THRESHOLD and live_account.holding:
                print("SIGNAL: HOLD")
                print(f"  -> Already in position; AI is {predicted_up_prob*100:.1f}% confident price will keep rising.")
            elif predicted_up_prob < SELL_CONFIDENCE_THRESHOLD and live_account.holding:
                live_account.sell(current_price, reason="signal")
                print("SIGNAL: SELL")
                print(f"  -> AI is {(1-predicted_up_prob)*100:.1f}% confident of a drop. Position closed.")
            elif predicted_up_prob > BUY_CONFIDENCE_THRESHOLD and not can_buy:
                print("SIGNAL: WAIT")
                print("  -> AI predicts a rise, but trend filter (EMA50<EMA200) blocks new buys.")
            else:
                print("SIGNAL: WAIT")
                print(f"  -> AI confidence ({predicted_up_prob*100:.1f}%) is within the neutral band; standing by.")
            print("-" * 32)

            # ---- Live stats ----
            summary = live_account.summary_dict(current_price)
            print(f"\nLIVE STATS | Balance: ${summary['current_balance']:,.2f} | "
                  f"Equity: ${summary['equity']:,.2f} | P/L: ${summary['profit_loss']:,.2f} "
                  f"({summary['roi_pct']:.2f}%) | Trades: {summary['total_trades']} "
                  f"(W:{summary['wins']} L:{summary['losses']}, {summary['win_rate']:.1f}% win rate)")
            if summary["open_trade_pl_pct"] is not None:
                print(f"Open trade unrealized P/L: {summary['open_trade_pl_pct']:.2f}%")

            # ---- Persist live account state every loop (Feature: remember everything) ----
            save_account(live_account)

            # ---- Dashboard (Feature #2, #10) ----
            # Use LIVE data here, not the one-time backtest snapshot — otherwise
            # the chart replays the same 50 points every loop and never updates.
            chart_window = live_data.tail(50)
            chart_index = chart_window.index
            chart_actual = chart_window['Close'].values
            chart_pred_probs = predict_many(model, scaler, chart_window[FEATURES])

            render_dashboard(
                fig, ax_chart, ax_panel,
                chart_index, chart_actual, chart_pred_probs,
                summary, current_price, predicted_up_prob
            )

            plt.draw()
            plt.pause(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C).")
    finally:
        # Save no matter how the loop exits (clean stop, Ctrl+C, or crash),
        # so resuming later picks up exactly where this run left off.
        save_account(live_account)
        save_tracker(tracker)
        print(f"Live account state saved to {ACCOUNT_FILE}. Safe to resume later.")


if __name__ == "__main__":
    main()