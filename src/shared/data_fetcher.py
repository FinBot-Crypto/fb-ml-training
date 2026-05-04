"""
Fetcher de dados da Binance com tratamento robusto de erros e paginação.
"""
import logging
import asyncio
import pandas as pd
import ccxt
from typing import Optional, List

logger = logging.getLogger(__name__)

TIMEFRAME_MS = {
    '1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
    '1h': 3600000, '4h': 14400000, '1d': 86400000,
}

MAX_PER_REQUEST = 1000


class BinanceDataFetcher:
    """
    Fetcher robusto para dados da Binance via CCXT com paginação automática.
    """

    def __init__(self, testnet: bool = False):
        exchange_config = {
            'enableRateLimit': True,
            'rateLimit': 200,
        }

        if testnet:
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                },
            }

        self.exchange = ccxt.binance(exchange_config)
        logger.info(f"BinanceDataFetcher inicializado (testnet={testnet})")

    async def _fetch_chunk(
        self, symbol: str, timeframe: str,
        since: Optional[int] = None, end_time: Optional[int] = None
    ) -> Optional[list]:
        """Busca um chunk de até MAX_PER_REQUEST candles."""
        params = {}
        if end_time is not None:
            params['endTime'] = end_time

        for attempt in range(3):
            try:
                ohlcv = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv,
                    symbol, timeframe, since, MAX_PER_REQUEST, params
                )
                return ohlcv if ohlcv else []
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(f"Erro ao buscar chunk de {symbol}: {e}")
                return None

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
    ) -> Optional[pd.DataFrame]:
        """
        Busca dados OHLCV com paginação automática.

        Args:
            symbol: Par de trading (ex: BTC/USDT)
            timeframe: Timeframe (ex: 1h, 4h, 1d)
            limit: Número de candles (máx ~3000 prático)

        Returns:
            DataFrame com OHLCV ou None se falhar
        """
        tf_ms = TIMEFRAME_MS.get(timeframe, 3600000)
        all_candles = []

        while len(all_candles) < limit:
            if not all_candles:
                chunk = await self._fetch_chunk(symbol, timeframe)
            else:
                chunk = await self._fetch_chunk(
                    symbol, timeframe, end_time=all_candles[0][0] - 1
                )

            if chunk is None:
                logger.error(f"Falha ao buscar dados para {symbol}")
                return None
            if not chunk:
                break

            if not all_candles:
                all_candles = chunk[:]
            else:
                all_candles = chunk + all_candles

            if len(chunk) < MAX_PER_REQUEST:
                break

        if not all_candles:
            return None

        all_candles = all_candles[-limit:]

        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        if len(df) < 10:
            logger.warning(f"Poucos dados para {symbol}: {len(df)} candles")
            return None

        if df['close'].isna().any():
            logger.warning(f"NaN encontrados em {symbol}")
            return None

        logger.info(f"OK {symbol} carregado: {len(df)} candles ({timeframe})")
        return df
    
    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 1000
    ) -> dict:
        """
        Busca dados para múltiplos símbolos em paralelo.
        
        Args:
            symbols: Lista de pares de trading
            timeframe: Timeframe
            limit: Número de candles
        
        Returns:
            Dict {symbol: DataFrame} com dados bem-sucedidos
        """
        tasks = [
            self.fetch_ohlcv(symbol, timeframe, limit)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks)
        
        data = {}
        for symbol, df in zip(symbols, results):
            if df is not None:
                data[symbol] = df
        
        logger.info(f"Carregados {len(data)} de {len(symbols)} símbolos")
        return data
