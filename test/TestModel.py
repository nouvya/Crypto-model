"""
Live Trading Script with Kelly Position Sizing
-----------------------------------------------
This script loads pre-trained models for AAVE, BTC, DOGE, XRP (stored in test/TestData/<symbol>/)
and runs a simulated trading loop using 1‑hour klines from Binance.

Key features:
- Independent capital for each symbol (2500 USD each, total 10000 USD).
- Trading decision based on model probability and fixed threshold (0.5).
- Position size determined by half‑Kelly formula with fixed odds ratio b = 2.
- Every hour, fetches the latest kline, updates internal buffers, computes 17 features,
  obtains a probability, and acts accordingly (buy if probability > threshold and no position,
  sell all if probability ≤ threshold and position held).
- Logs all actions and probabilities – suitable for background execution.

Run with nohup:
    nohup python TestModel.py > trading.log 2>&1 &

Dependencies:
    pip install pandas numpy lightgbm joblib python-binance
"""

import os
import time
import logging
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from binance.client import Client as BinanceClient

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Paths – assumes script is inside a folder "test" that also contains "TestData"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "TestData")

# Symbols to trade (must match folder names in MODEL_ROOT)
SYMBOLS = ['AAVE', 'BTC', 'DOGE', 'XRP']

# Binance symbol mapping (Binance uses USDT pairs)
BINANCE_MAP = {
    'AAVE': 'AAVEUSDT',
    'BTC': 'BTCUSDT',
    'DOGE': 'DOGEUSDT',
    'XRP': 'XRPUSDT'
}

# Trading parameters
INITIAL_CAPITAL_PER_SYMBOL = 2500.0   # each symbol gets its own cash pool
THRESHOLD = 0.5                       # probability threshold to enter long
KELLY_B = 2.0                          # fixed odds ratio for Kelly formula
HALF_KELY = True                       # use half‑Kelly (recommended for crypto)
BUFFER_SIZE = 300                      # number of past candles kept for feature calculation

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]   # logs go to stdout/stderr
)
logger = logging.getLogger('LiveKelly')

# ------------------------------------------------------------------------------
# Feature Engineering (exact copy from training)
# ------------------------------------------------------------------------------
def add_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 17 features used in training.
    Input df must contain: date, open, high, low, close, volume, quote_volume.
    Returns a DataFrame with the 17 features plus original columns.
    """
    df = df.sort_values('date').reset_index(drop=True)

    # Basic returns
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_24'] = df['close'].pct_change(24)

    # Volatility
    for period in [12, 24]:
        df[f'volatility_{period}'] = df['returns_1'].rolling(period).std()
        df[f'volatility_ratio_{period}'] = (
            df[f'volatility_{period}'] /
            df[f'volatility_{period}'].rolling(period * 2).mean()
        )

    # OBV and its MA
    obv = [0]
    for i in range(1, len(df)):
        if df.loc[i, 'close'] > df.loc[i-1, 'close']:
            obv.append(obv[-1] + df.loc[i, 'volume'])
        elif df.loc[i, 'close'] < df.loc[i-1, 'close']:
            obv.append(obv[-1] - df.loc[i, 'volume'])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ma_6'] = df['obv'].rolling(6).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, high_close, low_close)
    df['atr_14'] = tr.rolling(14).mean()

    # MACD signal
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['macd_signal'] = macd.ewm(span=9, adjust=False).mean()

    # Realized price approx (24h VWAP)
    df['realized_price_approx'] = (
        df['quote_volume'].rolling(24).sum() /
        df['volume'].rolling(24).sum()
    )

    # Hour features
    df['hour'] = df['date'].dt.hour
    df['is_last_2h'] = df['hour'].isin([22, 23]).astype(int)
    df['volume_last_2h'] = df['volume'] * df['is_last_2h']
    df['volume_last2h_ratio_24h'] = (
        df['volume_last_2h'].rolling(24).sum() /
        df['volume'].rolling(24).sum()
    )
    df['is_first_hour'] = (df['hour'] == 0).astype(int)
    df['volume_first_hour'] = df['volume'] * df['is_first_hour']
    df['volume_first_hour_ratio_24h'] = (
        df['volume_first_hour'].rolling(24).sum() /
        df['volume'].rolling(24).sum()
    )

    # Date features
    df['dayofmonth'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Interactions
    df['month_day'] = df['month'] * df['dayofmonth'] / 100
    df['daymonth_obv_ma'] = df['dayofmonth'] * df['obv_ma_6']
    df['month_last2h'] = df['month'] * df['volume_last2h_ratio_24h']
    df['last2h_ratio_dayofweek'] = df['volume_last2h_ratio_24h'] * df['dayofweek']
    df['ret24_volratio'] = df['returns_24'] * df['volatility_ratio_24']
    df['vol_12_24_product'] = df['volatility_12'] * df['volatility_24']

    # Keep only the 17 features plus original columns
    keep_cols = [
        'month_day', 'daymonth_obv_ma', 'atr_14', 'obv_ma_6', 'obv',
        'month_last2h', 'last2h_ratio_dayofweek', 'volatility_ratio_12',
        'volatility_ratio_24', 'volume_first_hour_ratio_24h', 'volume_last2h_ratio_24h',
        'realized_price_approx', 'ret24_volratio', 'volatility_24', 'dayofmonth',
        'macd_signal', 'vol_12_24_product'
    ]
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
    df = df[keep_cols + base_cols]
    df.dropna(inplace=True)
    return df

# ------------------------------------------------------------------------------
# Model Manager (loads models and feature lists)
# ------------------------------------------------------------------------------
class ModelManager:
    def __init__(self, root: str, symbols: list):
        self.models = {}
        self.features = {}
        for sym in symbols:
            model_path = os.path.join(root, sym, "model.pkl")
            feat_path = os.path.join(root, sym, "features.txt")
            if not os.path.exists(model_path) or not os.path.exists(feat_path):
                raise FileNotFoundError(f"Missing model/features for {sym}")
            self.models[sym] = joblib.load(model_path)
            with open(feat_path) as f:
                self.features[sym] = [line.strip() for line in f]
            logger.info(f"Loaded model for {sym}")

    def predict(self, symbol: str, features_df: pd.DataFrame) -> float:
        X = features_df[self.features[symbol]]
        return self.models[symbol].predict_proba(X)[0, 1]

# ------------------------------------------------------------------------------
# Binance Data Helpers
# ------------------------------------------------------------------------------
def fetch_historical_klines(symbol: str, limit: int = 300) -> pd.DataFrame:
    """Fetch recent klines from Binance public API."""
    client = BinanceClient()
    try:
        klines = client.get_klines(symbol=symbol, interval='1h', limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
    except Exception as e:
        logger.error(f"Binance fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_latest_kline(symbol: str) -> Optional[Dict]:
    """Get the most recent complete hourly kline."""
    df = fetch_historical_klines(symbol, limit=1)
    if df.empty:
        return None
    row = df.iloc[-1]
    return {
        'date': row['date'],
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close'],
        'volume': row['volume'],
        'quote_volume': row['quote_volume']
    }

# ------------------------------------------------------------------------------
# Simulated Trader per symbol (with Kelly sizing)
# ------------------------------------------------------------------------------
class SymbolTrader:
    def __init__(self, symbol: str, model_manager: ModelManager,
                 initial_cash: float, threshold: float,
                 kelly_b: float, half_kelly: bool, buffer_size: int = 300):
        self.symbol = symbol
        self.model_manager = model_manager
        self.cash = initial_cash
        self.position = 0.0
        self.threshold = threshold
        self.kelly_b = kelly_b
        self.half_kelly = half_kelly
        self.buffer_size = buffer_size
        self.buffer = pd.DataFrame()   # raw OHLCV data

    def kelly_fraction(self, prob: float) -> float:
        """Compute (half-)Kelly fraction based on fixed b."""
        q = 1 - prob
        f = (self.kelly_b * prob - q) / self.kelly_b
        f = max(0.0, f)                 # never negative
        if self.half_kelly:
            f = f / 2.0
        return f

    def update_buffer(self, row: pd.Series):
        """Add a new candle, keep only last buffer_size rows."""
        new_df = pd.DataFrame([row])
        if self.buffer.empty:
            self.buffer = new_df
        else:
            self.buffer = pd.concat([self.buffer, new_df], ignore_index=True)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer.iloc[-self.buffer_size:].reset_index(drop=True)

    def get_signal(self) -> Optional[float]:
        """Compute probability using current buffer. Return None if insufficient data."""
        if len(self.buffer) < 100:      # need enough data for stable features
            return None
        feat_df = add_features_single(self.buffer)
        if feat_df.empty:
            return None
        last_row = feat_df.iloc[-1:]
        prob = self.model_manager.predict(self.symbol, last_row)
        return prob

    def process_candle(self, candle: Dict):
        """Called every hour with a new candle."""
        # Convert candle dict to Series for buffer
        row_series = pd.Series(candle)
        self.update_buffer(row_series)

        prob = self.get_signal()
        if prob is None:
            logger.debug(f"{self.symbol} – insufficient data, skipping")
            return

        current_price = candle['close']
        logger.info(f"{self.symbol} probability: {prob:.4f}")

        # Trading logic
        if prob > self.threshold:
            if self.position == 0:
                # Buy using Kelly fraction of available cash
                fraction = self.kelly_fraction(prob)
                invest = self.cash * fraction
                if invest > 0:
                    qty = invest / current_price
                    self.position = qty
                    self.cash -= invest
                    logger.info(
                        f"SIM BUY {self.symbol} at {current_price:.4f}, qty {qty:.6f}, "
                        f"invest {invest:.2f} (prob {prob:.4f})"
                    )
            # else already in position – hold
        else:
            if self.position > 0:
                # Sell everything
                proceeds = self.position * current_price
                self.cash += proceeds
                logger.info(
                    f"SIM SELL {self.symbol} at {current_price:.4f}, proceeds {proceeds:.2f}"
                )
                self.position = 0.0

        # Log current equity for this symbol (optional, for debugging)
        equity = self.cash + self.position * current_price
        logger.debug(f"{self.symbol} equity after this hour: {equity:.2f}")

    def current_equity(self, current_price: float) -> float:
        """Return total value (cash + position) at given price."""
        return self.cash + self.position * current_price

# ------------------------------------------------------------------------------
# Main trading loop
# ------------------------------------------------------------------------------
def main():
    # Load models
    manager = ModelManager(MODEL_ROOT, SYMBOLS)

    # Create traders for each symbol
    traders = {}
    for sym in SYMBOLS:
        traders[sym] = SymbolTrader(
            symbol=sym,
            model_manager=manager,
            initial_cash=INITIAL_CAPITAL_PER_SYMBOL,
            threshold=THRESHOLD,
            kelly_b=KELLY_B,
            half_kelly=HALF_KELY,
            buffer_size=BUFFER_SIZE
        )

    # Pre‑fill buffers with historical data
    for sym in SYMBOLS:
        bin_sym = BINANCE_MAP.get(sym)
        if not bin_sym:
            logger.error(f"No Binance mapping for {sym}, skipping.")
            continue
        hist_df = fetch_historical_klines(bin_sym, limit=BUFFER_SIZE)
        if hist_df.empty:
            logger.warning(f"Could not fetch historical data for {sym}, will start empty.")
        else:
            # Convert each row to dict and add to buffer (simplified: assign entire DataFrame)
            # We'll just store the DataFrame as initial buffer
            traders[sym].buffer = hist_df.copy()
            logger.info(f"Initialised buffer for {sym} with {len(hist_df)} candles.")

    logger.info("Starting main loop. Press Ctrl+C to stop.")
    while True:
        try:
            # 1. Fetch latest candle for each symbol
            for sym in SYMBOLS:
                bin_sym = BINANCE_MAP.get(sym)
                if not bin_sym:
                    continue
                latest = fetch_latest_kline(bin_sym)
                if latest:
                    traders[sym].process_candle(latest)
                else:
                    logger.warning(f"Could not fetch latest kline for {sym}")

            # 2. Sleep until the next whole hour (plus a small offset)
            now = datetime.now(timezone.utc)
            next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            sleep_seconds = (next_hour - now).total_seconds() + 10   # add 10s buffer
            logger.info(f"Sleeping {sleep_seconds:.0f} seconds until next hour.")
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, shutting down.")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()