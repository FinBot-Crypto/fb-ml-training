"""
Mean Reversion V1 - Major (LSTM, score contínuo -1 a +1).

Target: clip(retorno_futuro / SCALE_PCT, -1, +1)
  retorno +3% → target +1.0  (compra forte)
  retorno  0% → target  0.0  (neutro)
  retorno -2% → target -1.0  (venda forte)

Saída do modelo: tanh → score contínuo em [-1, +1].
Quem define o threshold de operação é o fb-decision-engine.
"""
import logging
import numpy as np
import pandas as pd
from src.shared.base_dataset import BaseDataset
from src.shared.indicators import (
    calculate_rsi, calculate_sma, calculate_bollinger_bands
)
from . import config

logger = logging.getLogger(__name__)

SCALE_PCT = 0.02  # retorno / 2% → clip → [-1, +1]


class MeanReversionV1Dataset(BaseDataset):

    def __init__(self, symbol: str):
        super().__init__(symbol=symbol, tier=config.TIER)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Criando features para {self.symbol} ({len(df)} candles)...")

        close = df['close']
        volume = df['volume']

        df['sma'] = calculate_sma(close, config.SMA_PERIOD)
        df['rsi'] = calculate_rsi(close, config.RSI_PERIOD)
        df['rsi_smooth'] = df['rsi'].ewm(span=2, adjust=False).mean()
        df['deviation_from_sma'] = (close - df['sma']) / df['sma']

        bb_mid, bb_up, bb_lo = calculate_bollinger_bands(
            close, config.SMA_PERIOD, config.BB_STD
        )
        bb_rng = bb_up - bb_lo
        df['bb_position'] = (close - bb_lo) / bb_rng
        df['bb_width'] = bb_rng / bb_mid

        vol_sma = calculate_sma(volume, config.VOLUME_SMA_PERIOD)
        df['volume_ratio'] = volume / vol_sma

        return df

    def add_target_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target contínuo em [-1, +1].
        Score = clip(retorno_max_futuro / 2%, -1, +1)

        +1.0 = movimento forte pra cima (ex: +3%)
        +0.5 = movimento moderado pra cima (ex: +1%)
         0.0 = estável / neutro
        -0.5 = movimento moderado pra baixo (ex: -1%)
        -1.0 = movimento forte pra baixo (ex: -3%)
        """
        df = df.copy()
        la = config.LOOKAHEAD_CANDLES

        max_future = df['close'].shift(-1)
        for i in range(2, la + 1):
            max_future = np.maximum(max_future, df['close'].shift(-i))

        future_return = max_future / df['close'] - 1
        df['target'] = np.clip(future_return / SCALE_PCT, -1, 1)
        df.loc[df.index[-la:], 'target'] = np.nan

        stats = df['target'].dropna()
        logger.info(f"  Target contínuo [-1,+1]: média={stats.mean():.3f} "
                    f"std={stats.std():.3f} | {len(stats)} amostras")

        return df
