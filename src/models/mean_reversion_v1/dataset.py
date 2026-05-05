"""
Mean Reversion V1 - 19 features, 2 timeframes, target balanceado.

Features:
  1h: RSI14/24, retornos 12h/24h, desvios SMA, BB, ATR, volume, candle
  4h: RSI14, BB, retorno 24h, volume, distância de mínima

Target: 1 se retorno > mediana (balanceado 50/50)
Score: 2 * predict_proba - 1
"""
import logging
import numpy as np
import pandas as pd
from src.shared.base_dataset import BaseDataset
from src.shared.indicators import (
    calculate_rsi, calculate_sma, calculate_bollinger_bands,
    calculate_atr, calculate_momentum, calculate_volatility
)
from . import config

logger = logging.getLogger(__name__)


class MeanReversionV1Dataset(BaseDataset):

    def __init__(self, symbol: str):
        super().__init__(symbol=symbol, tier=config.TIER)

    def _features_1h(self, df):
        """19 features no timeframe 1h."""
        close, high, low, vol = df['close'], df['high'], df['low'], df['volume']

        # RSI multiplos periodos
        df['rsi_14'] = calculate_rsi(close, 14)
        df['rsi_24'] = calculate_rsi(close, 24)
        df['rsi_smooth'] = df['rsi_14'].ewm(span=2, adjust=False).mean()

        # Retornos em janelas
        df['ret_12h'] = close.pct_change(12)
        df['ret_24h'] = close.pct_change(24)

        # Desvios da media movel
        sma20 = calculate_sma(close, 20)
        sma60 = calculate_sma(close, 60)
        df['dev_sma_20'] = (close - sma20) / sma20
        df['dev_sma_60'] = (close - sma60) / sma60

        # Bollinger Bands (SMA 20)
        bb_mid, bb_up, bb_lo = calculate_bollinger_bands(close, 20, 2)
        bb_rng = bb_up - bb_lo
        df['bb_pos_20'] = (close - bb_lo) / bb_rng
        df['bb_width_20'] = bb_rng / bb_mid

        # ATR normalizado
        atr = calculate_atr(df, 14)
        df['atr_ratio'] = atr / close

        # Volume ratio
        vol_sma20 = calculate_sma(vol, 20)
        df['vol_ratio'] = vol / vol_sma20

        # Distancia da minima de 24h
        low_24h = low.rolling(24).min()
        df['dist_24h_low'] = (close - low_24h) / close

        # Estrutura do candle (pavio inferior)
        df['low_wick'] = (close - low) / (high - low)

        return df

    def _features_4h(self, df):
        """Features no timeframe 4h, forward-filled para 1h."""
        df_ts = df.set_index('timestamp')
        close_4 = df_ts['close'].resample('4h').last()
        high_4 = df_ts['high'].resample('4h').max()
        low_4 = df_ts['low'].resample('4h').min()
        vol_4 = df_ts['volume'].resample('4h').sum()

        # Shift para evitar lookahead
        close_4 = close_4.shift(1).dropna()
        high_4 = high_4.shift(1).dropna()
        low_4 = low_4.shift(1).dropna()
        vol_4 = vol_4.shift(1).dropna()

        # RSI 14 em 4h
        rsi = calculate_rsi(close_4, 14)
        # BB em 4h (SMA 20 em 4h = 80h)
        bb_mid, bb_up, bb_lo = calculate_bollinger_bands(close_4, 20, 2)
        bb_rng = bb_up - bb_lo
        # Retorno 24h em 4h = 6 candles
        ret = close_4.pct_change(6)
        # Volume ratio
        vsma = calculate_sma(vol_4, 20)
        # Minima 48h em 4h = 12 candles
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
        logger.info(f"Features para {self.symbol} ({len(df)} candles 1h)...")

        df = self._features_1h(df)
        feat_4h = self._features_4h(df)

        # Merge 4h de volta para 1h
        df_ts = df.set_index('timestamp')
        df_res = df_ts.join(feat_4h, how='left')
        df_res = df_res.ffill()

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
