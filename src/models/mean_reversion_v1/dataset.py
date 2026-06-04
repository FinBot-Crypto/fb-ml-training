"""
Mean Reversion V1 - RSI + Funding Rate + OI.
Target: direcao do RSI (RSI[t+12] > RSI[t]).
"""
import logging
import numpy as np
import pandas as pd
from src.shared.base_dataset import BaseDataset
from src.shared.indicators import calculate_rsi, calculate_sma
from . import config

logger = logging.getLogger(__name__)


class MeanReversionV1Dataset(BaseDataset):

    def __init__(self, symbol: str, direction: str = "long"):
        super().__init__(symbol=symbol, tier=config.TIER)
        self.funding_df = None
        self.oi_df = None
        self.btc_df = None
        self.direction = direction.lower()

    def set_futures_data(self, funding_df=None, oi_df=None):
        self.funding_df = funding_df
        self.oi_df = oi_df
        return self

    def set_btc_data(self, btc_df=None):
        self.btc_df = btc_df
        return self

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Features para {self.symbol} ({len(df)} candles)...")

        close = df['close']

        # RSI
        tf = config.TIMEFRAME
        mult = 56 if tf == '15m' else (28 if tf == '30m' else (168 if tf == '5m' else 14))
        df['rsi_14'] = calculate_rsi(close, mult)
        df['rsi_smooth'] = df['rsi_14'].ewm(span=2, adjust=False).mean()
        r4h = 16 if tf == '15m' else (8 if tf == '30m' else (48 if tf == '5m' else 4))
        df['rsi_14_4h'] = df['rsi_14'].rolling(r4h).mean()

        df['rsi_slope'] = df['rsi_14'].diff(6)

        # BTC features (estacionárias)
        if self.symbol == 'BTC/USDT':
            df['btc_rsi_14'] = calculate_rsi(df['close'], mult)
        else:
            if self.btc_df is not None and len(self.btc_df) > 0:
                btc = self.btc_df.copy()
                btc['btc_rsi_14'] = calculate_rsi(btc['close'], mult)
                df_ts = df.set_index('timestamp')
                df_ts = df_ts.join(btc.set_index('timestamp')[['btc_rsi_14']], how='left')
                df = df_ts.reset_index()
            else:
                df['btc_rsi_14'] = np.nan

        # Manter apenas features configuradas + essenciais
        keep = config.FEATURES + ['timestamp', 'close']
        return df[[c for c in keep if c in df.columns]].ffill()

    def add_target_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: 1 se RSI estiver MAIOR (LONG) ou MENOR (SHORT) em LOOKAHEAD_CANDLES.
        """
        df = df.copy()
        la = config.LOOKAHEAD_CANDLES

        future_rsi = df['rsi_smooth'].shift(-la)
        if self.direction == "short":
            df['target'] = (future_rsi < df['rsi_smooth']).astype(float)
            target_str = "cai"
        else:
            df['target'] = (future_rsi > df['rsi_smooth']).astype(float)
            target_str = "sobe"
        df.loc[df.index[-la:], 'target'] = np.nan

        pos = df['target'].sum()
        n = len(df.dropna(subset=['target']))
        logger.info(f"  Target RSI: {pos:.0f} {target_str} ({pos/max(n,1):.1%}) de {n} | "
                    f"autocorr={df['rsi_smooth'].autocorr():.3f}")

        return df

    def _features_1h(self, df):
        close, high, low, vol = df['close'], df['high'], df['low'], df['volume']
        df['rsi_14'] = calculate_rsi(close, 14)
        df['rsi_smooth'] = df['rsi_14'].ewm(span=2, adjust=False).mean()
        return df
