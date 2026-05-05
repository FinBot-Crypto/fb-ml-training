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

    def __init__(self, symbol: str):
        super().__init__(symbol=symbol, tier=config.TIER)
        self.funding_df = None
        self.oi_df = None

    def set_futures_data(self, funding_df=None, oi_df=None):
        self.funding_df = funding_df
        self.oi_df = oi_df
        return self

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Features para {self.symbol} ({len(df)} candles)...")

        close = df['close']

        # RSI
        rsi_period = 56 if config.TIMEFRAME in ('15m',) else (28 if config.TIMEFRAME == '30m' else 14)
        df['rsi_14'] = calculate_rsi(close, rsi_period)
        df['rsi_smooth'] = df['rsi_14'].ewm(span=2, adjust=False).mean()
        rsi_4h = 16 if config.TIMEFRAME == '15m' else (8 if config.TIMEFRAME == '30m' else 4)
        df['rsi_14_4h'] = df['rsi_14'].rolling(rsi_4h).mean()

        # Desvio da media movel (distancia da SMA 60)
        sma60 = calculate_sma(close, 60)
        df['deviation_sma'] = (close - sma60) / sma60

        # Funding rate (shift do timestamp para garantir zero lookahead)
        if self.funding_df is not None and len(self.funding_df) > 0:
            fd = self.funding_df.copy()
            fd.index = fd.index + pd.Timedelta(hours=8)  # funding so vale pro futuro
            df_ts = df.set_index('timestamp')
            df_ts = df_ts.join(fd[['fundingRate']], how='left')
            df_ts['funding_rate'] = df_ts['fundingRate'].ffill()
            df_ts['funding_change'] = df_ts['funding_rate'].diff()
            df = df_ts.drop(columns=['fundingRate']).reset_index()
        else:
            df['funding_rate'] = np.nan
            df['funding_change'] = np.nan

        # Open interest (shift do timestamp)
        if self.oi_df is not None and len(self.oi_df) > 0:
            oi = self.oi_df.copy()
            oi.index = oi.index + pd.Timedelta(hours=1)  # OI so vale 1h depois
            df_ts = df.set_index('timestamp')
            df_ts = df_ts.join(oi[['openInterestValue']], how='left')
            df_ts['open_interest'] = df_ts['openInterestValue'].ffill()
            df_ts['oi_change_1h'] = df_ts['open_interest'].pct_change()
            df_ts['oi_change_24h'] = df_ts['open_interest'].pct_change(24)
            df = df_ts.drop(columns=['openInterestValue']).reset_index()
        else:
            df['open_interest'] = np.nan
            df['oi_change_1h'] = np.nan
            df['oi_change_24h'] = np.nan

        # Manter apenas features configuradas + essenciais
        keep = config.FEATURES + ['timestamp', 'close']
        return df[[c for c in keep if c in df.columns]].ffill()

    def add_target_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: reversao a media em 12h.
        1 = preco reverteu (voltou em direcao a SMA) dentro de LOOKAHEAD_CANDLES.
        0 = preco NAO reverteu (foi para longe ou ficou estavel).

        Só sao rotulados candles com |desvio| > 2% da SMA 60.
        """
        df = df.copy()
        la = config.LOOKAHEAD_CANDLES
        sma60 = calculate_sma(df['close'], 60)
        deviation = (df['close'] - sma60) / sma60

        # Preco futuro min/max
        max_future = df['close'].shift(-1)
        min_future = df['close'].shift(-1)
        for i in range(2, la + 1):
            max_future = np.maximum(max_future, df['close'].shift(-i))
            min_future = np.minimum(min_future, df['close'].shift(-i))

        # Reversao: desvio diminuiu em modulo
        # Se desvio positivo (preco acima da SMA): reverteu se preco caiu
        # Se desvio negativo (preco abaixo da SMA): reverteu se preco subiu
        acima = deviation > 0.02
        abaixo = deviation < -0.02
        reverteu_alta = abaixo & (max_future > sma60)  # subiu ate passar a SMA
        reverteu_baixa = acima & (min_future < sma60)  # caiu ate passar a SMA

        df['target'] = np.nan
        df.loc[reverteu_alta | reverteu_baixa, 'target'] = 1.0
        df.loc[acima | abaixo, 'target'] = df['target'].fillna(0.0)
        df.loc[df.index[-la:], 'target'] = np.nan

        has_target = df['target'].dropna()
        pos = (has_target == 1).sum()
        logger.info(f"  Reversao a media: {pos:.0f} reversoes ({pos/max(len(has_target),1):.1%}) "
                    f"de {len(has_target)} candles desviados")

        return df

    def _features_1h(self, df):
        close, high, low, vol = df['close'], df['high'], df['low'], df['volume']
        df['rsi_14'] = calculate_rsi(close, 14)
        df['rsi_smooth'] = df['rsi_14'].ewm(span=2, adjust=False).mean()
        return df
