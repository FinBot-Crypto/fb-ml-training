"""
Mean Reversion V1 - 8 features RSI-focadas comprovadas por permutation test.
"""
import logging
import numpy as np
import pandas as pd
from src.shared.base_dataset import BaseDataset
from src.shared.indicators import calculate_rsi, calculate_sma
from . import config

logger = logging.getLogger(__name__)


class MeanReversionV1Dataset(BaseDataset):

    def __init__(self, symbol: str):
        super().__init__(symbol=symbol, tier=config.TIER)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Features RSI para {self.symbol} ({len(df)} candles)...")

        close = df['close']
        volume = df['volume']

        # --- 1h features ---
        rsi14 = calculate_rsi(close, 14)
        rsi24 = calculate_rsi(close, 24)
        df['rsi_14'] = rsi14
        df['rsi_24'] = rsi24
        df['rsi_smooth'] = rsi14.ewm(span=2, adjust=False).mean()
        df['rsi_slope'] = rsi14.diff(3)  # velocidade da virada
        df['rsi_cross'] = rsi14 - rsi24   # divergencia

        # Stochastic RSI: onde o RSI esta no seu proprio range recente
        rsi_min = rsi14.rolling(14).min()
        rsi_max = rsi14.rolling(14).max()
        df['stoch_rsi'] = (rsi14 - rsi_min) / (rsi_max - rsi_min)

        # MFI (Money Flow Index): RSI ponderado por volume
        typical = (df['high'] + df['low'] + close) / 3
        raw_money = typical * volume
        pos_flow = raw_money.where(typical > typical.shift(1), 0).rolling(14).sum()
        neg_flow = raw_money.where(typical < typical.shift(1), 0).rolling(14).sum()
        mfi_ratio = pos_flow / neg_flow
        df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))

        # --- 4h features ---
        df_ts = df.set_index('timestamp')
        close_4h = df_ts['close'].resample('4h').last().shift(1).dropna()
        rsi_4h = calculate_rsi(close_4h, 14)
        feat_4h = pd.DataFrame(index=close_4h.index)
        feat_4h['rsi_14_4h'] = rsi_4h

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
