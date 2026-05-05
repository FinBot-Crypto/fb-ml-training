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

        period = 56 if config.TIMEFRAME == '15m' else 14
        df['rsi_14'] = calculate_rsi(close, period)
        df['rsi_smooth'] = df['rsi_14'].ewm(span=2, adjust=False).mean()

        # Divergencia RSI: preco faz fundo mas RSI nao
        rsi_min_24 = df['rsi_14'].rolling(24).min()
        low_24 = close.rolling(24).min()
        df['rsi_divergence'] = (close - low_24) / close - (df['rsi_14'] - rsi_min_24) / 100

        # BB Squeeze: largura das bandas relativa a media historica
        bb_mid, bb_up, bb_lo = calculate_bollinger_bands(close, 20, 2)
        bb_width = (bb_up - bb_lo) / bb_mid
        bb_sma = bb_width.rolling(100).mean()
        df['bb_squeeze'] = bb_width / bb_sma

        # Velas consecutivas na mesma direcao
        direction = np.sign(close.diff())
        cons = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
        df['cons_candle'] = cons * direction

        return df

    def _features_4h(self, df):
        df_ts = df.set_index('timestamp')
        close_4 = df_ts['close'].resample('4h').last().shift(1).dropna()

        rsi = calculate_rsi(close_4, 14)
        features = pd.DataFrame(index=close_4.index)
        features['rsi_14_4h'] = rsi
        return features

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Features para {self.symbol} ({len(df)} candles)...")
        df = self._features_1h(df)
        feat_4h = self._features_4h(df)
        df_ts = df.set_index('timestamp')
        df_res = df_ts.join(feat_4h, how='left').ffill()
        # Manter apenas features configuradas + colunas essenciais
        keep = config.FEATURES + ['timestamp', 'close', 'open', 'high', 'low', 'volume']
        for c in df_res.columns:
            if c not in keep:
                df_res = df_res.drop(columns=[c])
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
