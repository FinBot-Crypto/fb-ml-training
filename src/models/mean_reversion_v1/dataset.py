"""
Mean Reversion V1 - 19 features, 2 timeframes, target balanceado.
"""
import logging
import numpy as np
import pandas as pd
from src.shared.base_dataset import BaseDataset
from src.shared.indicators import (
    calculate_rsi, calculate_sma, calculate_bollinger_bands,
    calculate_atr
)
from . import config

logger = logging.getLogger(__name__)


class MeanReversionV1Dataset(BaseDataset):

    def __init__(self, symbol: str):
        super().__init__(symbol=symbol, tier=config.TIER)

    def _features_1h(self, df):
        close, high, low, vol = df['close'], df['high'], df['low'], df['volume']

        df['rsi_14'] = calculate_rsi(close, 14)
        df['rsi_24'] = calculate_rsi(close, 24)
        df['rsi_smooth'] = df['rsi_14'].ewm(span=2, adjust=False).mean()
        df['ret_12h'] = close.pct_change(12)
        df['ret_24h'] = close.pct_change(24)

        sma20 = calculate_sma(close, 20)
        sma60 = calculate_sma(close, 60)
        df['dev_sma_20'] = (close - sma20) / sma20
        df['dev_sma_60'] = (close - sma60) / sma60

        bb_mid, bb_up, bb_lo = calculate_bollinger_bands(close, 20, 2)
        bb_rng = bb_up - bb_lo
        df['bb_pos_20'] = (close - bb_lo) / bb_rng
        df['bb_width_20'] = bb_rng / bb_mid

        atr = calculate_atr(df, 14)
        df['atr_ratio'] = atr / close

        vol_sma20 = calculate_sma(vol, 20)
        df['vol_ratio'] = vol / vol_sma20

        low_24h = low.rolling(24).min()
        df['dist_24h_low'] = (close - low_24h) / close
        df['low_wick'] = (close - low) / (high - low)

        return df

    def _features_4h(self, df):
        df_ts = df.set_index('timestamp')
        close_4 = df_ts['close'].resample('4h').last().shift(1).dropna()
        high_4 = df_ts['high'].resample('4h').max().shift(1).dropna()
        low_4 = df_ts['low'].resample('4h').min().shift(1).dropna()
        vol_4 = df_ts['volume'].resample('4h').sum().shift(1).dropna()

        rsi = calculate_rsi(close_4, 14)
        bb_mid, bb_up, bb_lo = calculate_bollinger_bands(close_4, 20, 2)
        bb_rng = bb_up - bb_lo
        ret = close_4.pct_change(6)
        vsma = calculate_sma(vol_4, 20)
        lo48 = low_4.rolling(12).min()

        features = pd.DataFrame(index=close_4.index)
        features['rsi_14_4h'] = rsi
        features['bb_pos_20_4h'] = (close_4 - bb_lo) / bb_rng
        features['bb_width_20_4h'] = bb_rng / bb_mid
        features['ret_24h_4h'] = ret
        features['vol_ratio_4h'] = vol_4 / vsma
        features['dist_48h_low_4h'] = (close_4 - lo48) / close_4

        return features

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Features para {self.symbol} ({len(df)} candles)...")
        df = self._features_1h(df)
        feat_4h = self._features_4h(df)
        df_ts = df.set_index('timestamp')
        df_res = df_ts.join(feat_4h, how='left').ffill()
        return df_res.reset_index()

    def add_target_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target balanceado: 1 se retorno > mediana do simbolo.
        Isso da ~50/50 e evita o vies de 85% do mercado.
        """
        df = df.copy()
        la = config.LOOKAHEAD_CANDLES

        max_future = df['close'].shift(-1)
        for i in range(2, la + 1):
            max_future = np.maximum(max_future, df['close'].shift(-i))

        future_return = max_future / df['close'] - 1
        median_ret = future_return.median()
        df['target'] = (future_return > median_ret).astype(float)
        df.loc[df.index[-la:], 'target'] = np.nan

        pos = df['target'].sum()
        total = len(df.dropna(subset=['target']))
        logger.info(f"  Target (mediana={median_ret:.4%}): {pos:.0f} positivos "
                    f"({pos/max(total,1):.1%}) de {total}")

        return df
