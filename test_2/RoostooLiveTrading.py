"""
Live Trading Script for Roostoo
---------------------------------------------------------------------
This script loads pre-trained models for AAVE, BTC, DOGE, XRP (stored in test/TestData/<symbol>/)
and runs a live trading loop using 1‑hour klines from Binance.

Key features:
- Independent capital for each symbol (2500 USD each, total 10000 USD).
- Trading decision based on model probability and fixed threshold (0.5).
- Position size determined by half‑Kelly formula with fixed odds ratio b = 2.
- Every hour, fetches the latest kline, updates internal buffers, computes 17 features,
  obtains a probability, and places a real MARKET order on Roostoo when a signal occurs.
- **Exchange info integration**: retrieves trading rules (AmountPrecision, MiniOrder) for each symbol
  to ensure order quantity complies with exchange requirements.
- **Time synchronisation** with Roostoo server is performed before each order to ensure timestamp validity.
- All orders are logged with full details: timestamp, symbol, side, price, quantity,
  order ID, API response, and strategy state (cash/position).
- Robust error handling and logging; script runs continuously until stopped.

How to run (on Linux/macOS with environment variables):
    export ROOSTOO_API_KEY="your_api_key"
    export ROOSTOO_SECRET_KEY="your_secret_key"
    # Optional: export ROOSTOO_BASE_URL="https://api.roostoo.com"   # for production, default is mock
    cd /path/to/test
    nohup python3 RoostooLiveTrading.py > trading.log 2>&1 &

On Windows (cmd):
    set ROOSTOO_API_KEY=your_api_key
    set ROOSTOO_SECRET_KEY=your_secret_key
    start /B python RoostooLiveTrading.py > trading.log 2>&1 &

Dependencies:
    pip install pandas numpy lightgbm joblib python-binance requests
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
# Roostoo API configuration (read from environment)
# ------------------------------------------------------------------------------
ROOSTOO_BASE_URL = os.environ.get('ROOSTOO_BASE_URL', 'https://mock-api.roostoo.com')
ROOSTOO_API_KEY = os.environ.get('ROOSTOO_API_KEY')
ROOSTOO_SECRET_KEY = os.environ.get('ROOSTOO_SECRET_KEY')

if not ROOSTOO_API_KEY or not ROOSTOO_SECRET_KEY:
    raise ValueError("Please set ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY environment variables.")

# ------------------------------------------------------------------------------
# General configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "TestData")

SYMBOLS = ['AAVE', 'BTC', 'DOGE', 'XRP']
BINANCE_MAP = {
    'AAVE': 'AAVEUSDT',
    'BTC': 'BTCUSDT',
    'DOGE': 'DOGEUSDT',
    'XRP': 'XRPUSDT'
}

INITIAL_CAPITAL_PER_SYMBOL = 250000.0
THRESHOLD = 0.5
KELLY_B = 2.0
HALF_KELY = True                # use half‑Kelly to reduce risk
BUFFER_SIZE = 300                # number of past 1‑hour candles kept for feature calculation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('LiveKellyReal')

# ------------------------------------------------------------------------------
# Fetch exchange info (trading rules)
# ------------------------------------------------------------------------------
def fetch_exchange_info() -> dict:
    """
    Retrieve trading rules from Roostoo.
    Returns a dictionary of TradePairs, or empty dict on failure.
    """
    url = f"{ROOSTOO_BASE_URL}/v3/exchangeInfo"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get('TradePairs', {})
    except Exception as e:
        logger.error(f"Failed to fetch exchange info: {e}")
        return {}

# ------------------------------------------------------------------------------
# Roostoo time synchronisation
# ------------------------------------------------------------------------------
def get_server_time() -> Optional[int]:
    """
    Retrieve current server time from Roostoo (milliseconds).
    Returns None if the request fails.
    """
    url = f"{ROOSTOO_BASE_URL}/v3/serverTime"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get('ServerTime')
    except Exception as e:
        logger.error(f"Failed to get server time: {e}")
        return None

# ------------------------------------------------------------------------------
# Roostoo API helpers (signature, order placement)
# ------------------------------------------------------------------------------
def _get_timestamp() -> str:
    """Return current millisecond timestamp as string (fallback if server time unavailable)."""
    return str(int(time.time() * 1000))

def _generate_signature(payload: dict, secret: str) -> str:
    """
    Generate HMAC SHA256 signature for the given payload.
    Payload must be a dictionary of key-value strings.
    The signature is computed over the sorted query string (key=value&...).
    """
    sorted_keys = sorted(payload.keys())
    query_string = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
    signature = hmac.new(
        secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def place_order_roostoo(pair: str, side: str, quantity: float,
                        order_type: str = 'MARKET', price: float = None) -> dict:
    """
    Place an order on Roostoo.
    Returns the JSON response from the API.
    """
    # First, obtain server time to ensure timestamp is within 60 seconds
    server_ts = get_server_time()
    if server_ts is None:
        logger.warning("Could not fetch server time, using local timestamp.")
        ts = _get_timestamp()
    else:
        ts = str(server_ts)

    url = f"{ROOSTOO_BASE_URL}/v3/place_order"
    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(quantity),
        'timestamp': ts
    }
    if order_type.upper() == 'LIMIT' and price is not None:
        payload['price'] = str(price)

    signature = _generate_signature(payload, ROOSTOO_SECRET_KEY)
    headers = {
        'RST-API-KEY': ROOSTOO_API_KEY,
        'MSG-SIGNATURE': signature,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    # build the body string
    sorted_keys = sorted(payload.keys())
    body = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

    try:
        resp = requests.post(url, headers=headers, data=body, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        logger.error(f"Roostoo order failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
    return None

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
# SymbolTrader with real Roostoo orders
# ------------------------------------------------------------------------------
class SymbolTrader:
    def __init__(self, symbol: str, model_manager: ModelManager,
                 initial_cash: float, threshold: float,
                 kelly_b: float, half_kelly: bool, buffer_size: int = 300,
                 exchange_info: dict = None):
        self.symbol = symbol
        self.model_manager = model_manager
        self.cash = initial_cash
        self.position = 0.0
        self.threshold = threshold
        self.kelly_b = kelly_b
        self.half_kelly = half_kelly
        self.buffer_size = buffer_size
        self.buffer = pd.DataFrame()   # raw OHLCV data

        # Trading rules from exchangeInfo
        self.exchange_info = exchange_info if exchange_info else {}
        self.amount_precision = self.exchange_info.get('AmountPrecision', 6)  # default to 6 decimal places
        self.mini_order = self.exchange_info.get('MiniOrder', 1.0)

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
        """Called every hour with a new candle. Places real Roostoo orders."""
        row_series = pd.Series(candle)
        self.update_buffer(row_series)

        prob = self.get_signal()
        if prob is None:
            logger.debug(f"{self.symbol} – insufficient data, skipping")
            return

        current_price = candle['close']
        logger.info(f"{self.symbol} probability: {prob:.4f}")

        # Trading logic – now using real Roostoo orders
        if prob > self.threshold:
            if self.position == 0:
                # Buy using Kelly fraction of available cash
                fraction = self.kelly_fraction(prob)
                invest = self.cash * fraction
                if invest > 0:
                    qty = invest / current_price
                    # Adjust quantity according to exchange precision
                    if self.amount_precision == 0:
                        qty = int(qty)
                    else:
                        qty = round(qty, self.amount_precision)
                    # Check minimum order value
                    order_value = qty * current_price
                    if order_value < self.mini_order:
                        logger.warning(
                            f"{self.symbol} order value {order_value:.2f} < min {self.mini_order}, skip"
                        )
                        return
                    pair = f"{self.symbol}/USD"
                    order_result = place_order_roostoo(
                        pair=pair,
                        side='BUY',
                        quantity=qty,
                        order_type='MARKET'
                    )
                    if order_result and order_result.get('Success'):
                        # Extract order ID from response
                        order_id = order_result.get('OrderDetail', {}).get('OrderID', 'N/A')
                        self.position = qty
                        self.cash -= invest
                        logger.info(
                            f"REAL BUY {self.symbol} at ~{current_price:.4f}, qty {qty:.6f}, "
                            f"invest {invest:.2f} (prob {prob:.4f}) – order confirmed, OrderID: {order_id}"
                        )
                    else:
                        logger.error(f"BUY order failed for {self.symbol}")
                # else invest <= 0 – nothing to do
            # else already in position – hold
        else:
            if self.position > 0:
                # Sell everything
                qty = self.position
                pair = f"{self.symbol}/USD"
                order_result = place_order_roostoo(
                    pair=pair,
                    side='SELL',
                    quantity=qty,
                    order_type='MARKET'
                )
                if order_result and order_result.get('Success'):
                    order_id = order_result.get('OrderDetail', {}).get('OrderID', 'N/A')
                    proceeds = qty * current_price
                    self.cash += proceeds
                    logger.info(
                        f"REAL SELL {self.symbol} at ~{current_price:.4f}, proceeds {proceeds:.2f} – order confirmed, OrderID: {order_id}"
                    )
                    self.position = 0.0
                else:
                    logger.error(f"SELL order failed for {self.symbol}")

        # Record equity (optional, for debugging)
        equity = self.cash + self.position * current_price
        logger.debug(f"{self.symbol} equity after this hour: {equity:.2f}")

# ------------------------------------------------------------------------------
# Main trading loop
# ------------------------------------------------------------------------------
def main():
    # Fetch exchange trading rules
    exchange_info = fetch_exchange_info()
    if not exchange_info:
        logger.warning("Could not retrieve exchange info. Order validation may be incomplete.")

    # Load models
    manager = ModelManager(MODEL_ROOT, SYMBOLS)

    # Create traders for each symbol
    traders = {}
    for sym in SYMBOLS:
        pair = f"{sym}/USD"
        traders[sym] = SymbolTrader(
            symbol=sym,
            model_manager=manager,
            initial_cash=INITIAL_CAPITAL_PER_SYMBOL,
            threshold=THRESHOLD,
            kelly_b=KELLY_B,
            half_kelly=HALF_KELY,
            buffer_size=BUFFER_SIZE,
            exchange_info=exchange_info.get(pair, {})
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