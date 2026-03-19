"""
Real-time Simulated Trading
----------------------------------------------
This script fetches live 1-hour klines from Binance, computes 17 features,
predicts upward probability using pre-trained models, and simulates trading
with a virtual 10000 USD capital. No real orders are placed on Roostoo.

Outputs:
- For each symbol, prints the latest probability and any trade signals.
- At the end, plots equity curve and prints final return, Sharpe, max drawdown.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client as BinanceClient

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Symbols to trade
SYMBOLS = ['AAVE', 'BTC', 'DOGE', 'XRP']
BINANCE_SYMBOL_MAP = {
    'AAVE': 'AAVEUSDT',
    'BTC': 'BTCUSDT',
    'DOGE': 'DOGEUSDT',
    'XRP': 'XRPUSDT'
}

# Trading parameters
THRESHOLD = 0.5          # probability threshold to enter long
INITIAL_CAPITAL = 10000.0  # virtual starting capital

# Data buffer size (must be >= 300 to cover all rolling windows)
BUFFER_SIZE = 300

# Model root directory (relative to this script)
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "TestData")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimTrader")

# ------------------------------------------------------------------------------
# Feature Engineering (same as training)
# ------------------------------------------------------------------------------
def add_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 17 features used in training.
    Input df must contain: date, open, high, low, close, volume, quote_volume.
    Returns a DataFrame with the 17 features and original columns.
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

    # Keep only the 17 features (plus original columns)
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
# Model Manager
# ------------------------------------------------------------------------------
class ModelManager:
    def __init__(self, root: str, symbols: List[str]):
        self.models: Dict[str, lgb.LGBMClassifier] = {}
        self.features: Dict[str, List[str]] = {}
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
# Data Fetcher (Binance)
# ------------------------------------------------------------------------------
def fetch_historical_klines(symbol: str, limit: int = 300) -> pd.DataFrame:
    client = BinanceClient()
    try:
        klines = client.get_klines(symbol=symbol, interval='1h', limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_volume','trades','taker_buy_base',
            'taker_buy_quote','ignore'
        ])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open','high','low','close','volume','quote_volume']:
            df[col] = df[col].astype(float)
        return df[['date','open','high','low','close','volume','quote_volume']]
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_latest_kline(symbol: str) -> Optional[Dict]:
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
# Simulated Trader
# ------------------------------------------------------------------------------
class SimulatedTrader:
    def __init__(self, model_manager: ModelManager, symbols: List[str],
                 threshold: float = 0.5, initial_capital: float = 10000.0):
        self.model_manager = model_manager
        self.symbols = symbols
        self.threshold = threshold
        self.initial_capital = initial_capital

        # Buffers for each symbol
        self.buffers: Dict[str, pd.DataFrame] = {sym: pd.DataFrame() for sym in symbols}

        # Virtual portfolio: for each symbol, position quantity
        self.positions: Dict[str, float] = {sym: 0.0 for sym in symbols}
        # Cash balance (USD)
        self.cash = initial_capital
        # Track total equity over time
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []

        # Initialize buffers with historical data
        self._init_buffers()

    def _init_buffers(self):
        for sym in self.symbols:
            bin_sym = BINANCE_SYMBOL_MAP.get(sym)
            if not bin_sym:
                continue
            df = fetch_historical_klines(bin_sym, limit=BUFFER_SIZE)
            if df.empty:
                logger.warning(f"Could not init buffer for {sym}")
            else:
                self.buffers[sym] = df.sort_values('date').reset_index(drop=True)
                logger.info(f"Buffer for {sym} initialized with {len(df)} candles")

    def update_buffer(self, symbol: str, new_row: Dict):
        if symbol not in self.buffers:
            return
        new_df = pd.DataFrame([new_row])
        if self.buffers[symbol].empty:
            self.buffers[symbol] = new_df
        else:
            self.buffers[symbol] = pd.concat([self.buffers[symbol], new_df], ignore_index=True)
        if len(self.buffers[symbol]) > BUFFER_SIZE:
            self.buffers[symbol] = self.buffers[symbol].iloc[-BUFFER_SIZE:].reset_index(drop=True)

    def get_signal(self, symbol: str) -> Optional[float]:
        if symbol not in self.buffers or self.buffers[symbol].empty:
            return None
        feat_df = add_features_single(self.buffers[symbol])
        if feat_df.empty:
            logger.warning(f"Feature computation returned empty for {symbol}.")
            return None
        # 检查特征列是否存在
        expected = self.model_manager.features[symbol]
        missing = [col for col in expected if col not in feat_df.columns]
        if missing:
            logger.error(f"Missing features for {symbol}: {missing}")
            logger.error(f"Available columns: {feat_df.columns.tolist()}")
            return None
        last = feat_df.iloc[-1:]
        prob = self.model_manager.predict(symbol, last)
        return prob

    def update_equity(self):
        """Calculate total portfolio value (cash + positions) at current prices."""
        total = self.cash
        for sym in self.symbols:
            if self.positions[sym] > 0:
                # Need current price. For simplicity, use last close from buffer.
                if sym in self.buffers and not self.buffers[sym].empty:
                    price = self.buffers[sym].iloc[-1]['close']
                    total += self.positions[sym] * price
        self.equity_curve.append(total)
        self.timestamps.append(datetime.now(timezone.utc))

    def execute_simulated_trade(self, symbol: str, prob: float, price: float):
        """Simulate a trade: buy if prob>threshold and no position; sell if prob<=threshold and have position."""
        if prob > self.threshold:
            if self.positions[symbol] == 0:
                # Buy with all available cash
                qty = self.cash / price
                self.positions[symbol] = qty
                self.cash = 0.0
                logger.info(f"SIM BUY {symbol} at {price:.2f}, qty {qty:.6f}")
        else:
            if self.positions[symbol] > 0:
                # Sell entire position
                self.cash += self.positions[symbol] * price
                logger.info(f"SIM SELL {symbol} at {price:.2f}, proceeds {self.positions[symbol]*price:.2f}")
                self.positions[symbol] = 0.0

    def run_once(self):
        """One iteration: fetch latest data, compute signals, simulate trades, update equity."""
        # 1. Fetch latest kline for each symbol
        for sym in self.symbols:
            bin_sym = BINANCE_SYMBOL_MAP.get(sym)
            if not bin_sym:
                continue
            latest = fetch_latest_kline(bin_sym)
            if latest:
                self.update_buffer(sym, latest)
            else:
                logger.warning(f"Failed to fetch latest kline for {sym}")

        # 2. For each symbol, get signal and simulate trade
        for sym in self.symbols:
            prob = self.get_signal(sym)
            if prob is None:
                continue
            logger.info(f"{sym} probability: {prob:.4f}")
            # Use the latest close price from buffer (the one we just added)
            if sym in self.buffers and not self.buffers[sym].empty:
                price = self.buffers[sym].iloc[-1]['close']
                self.execute_simulated_trade(sym, prob, price)

        # 3. Update equity curve
        self.update_equity()

    def run_loop(self, interval_seconds: int = 3600):
        """Main loop. Runs every interval_seconds (should be 3600 for hourly)."""
        logger.info("Starting simulated trading loop. Press Ctrl+C to stop.")
        while True:
            try:
                self.run_once()
                # Sleep until next whole hour + 10s
                now = datetime.now(timezone.utc)
                next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
                sleep_sec = (next_hour - now).total_seconds() + 10
                logger.info(f"Sleeping {sleep_sec:.0f}s until next hour.")
                time.sleep(sleep_sec)
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as e:
                logger.exception(f"Error: {e}")
                time.sleep(60)

    def print_results(self):
        """Print performance metrics without plotting."""
        if len(self.equity_curve) < 2:
            logger.warning("Not enough data to compute metrics.")
            return

        final = self.equity_curve[-1]
        total_return = (final / self.initial_capital) - 1

        # Calculate returns series from equity curve
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        # Annualization factor (hourly data)
        periods_per_year = 365 * 24
        annual_factor = np.sqrt(periods_per_year)

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe = returns.mean() / returns.std() * annual_factor if returns.std() != 0 else 0

        # Max drawdown
        cumulative = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())

        # Annualized return
        years = len(self.equity_curve) / periods_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * annual_factor
            sortino = (returns.mean() * annual_factor) / downside_std if downside_std != 0 else 0
        else:
            sortino = 0

        # Calmar ratio
        calmar = annual_return / max_dd if max_dd != 0 else 0

        # SSC (0.4 Sortino + 0.4 Sharpe + 0.3 Calmar)
        ssc = 0.4 * sortino + 0.4 * sharpe + 0.3 * calmar

        # Print results
        logger.info("========== Final Performance ==========")
        logger.info(f"Final Equity: {final:.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Return: {annual_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Sortino Ratio: {sortino:.2f}")
        logger.info(f"Calmar Ratio: {calmar:.2f}")
        logger.info(f"SSC: {ssc:.4f}")
        logger.info(f"Max Drawdown: {max_dd:.2%}")
        logger.info("======================================")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load models
    manager = ModelManager(MODEL_ROOT, SYMBOLS)

    # Create trader
    trader = SimulatedTrader(manager, SYMBOLS, threshold=THRESHOLD, initial_capital=INITIAL_CAPITAL)

    # Run simulation
    trader.run_loop()

    # After stopping, print results
    trader.print_results()