import asyncio
import logging
import os
import json
import ccxt
import pandas as pd
import numpy as np
import nats
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fb-ml-training")

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class MLTrainingService:
    def __init__(self):
        self.nc = None
        self.js = None
        self.exchange = ccxt.binance({'enableRateLimit': True})

    async def connect_nats(self):
        while True:
            try:
                self.nc = await nats.connect(NATS_URL)
                self.js = self.nc.jetstream()
                logger.info(f"Conectado ao NATS em {NATS_URL}")
                return
            except Exception as e:
                logger.error(f"Erro ao conectar NATS: {e}")
                await asyncio.sleep(5)

    async def fetch_training_data(self, symbol, timeframe='1h', limit=1000):
        try:
            ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            logger.error(f"Erro ao buscar dados para treino de {symbol}: {e}")
            return None

    def prepare_features(self, df):
        # Feature Engineering simples
        df['rsi'] = self.calculate_rsi(df['close'])
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Target: 1 se o preço subir 1% nas próximas 4 horas
        df['target'] = (df['close'].shift(-4) > df['close'] * 1.01).astype(int)
        
        df.dropna(inplace=True)
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    async def train_model(self, symbol):
        logger.info(f"Iniciando treinamento para {symbol}...")
        df = await self.fetch_training_data(symbol)
        if df is None or len(df) < 300:
            logger.warning(f"Dados insuficientes para {symbol}")
            return False

        df = self.prepare_features(df)
        X = df[['rsi', 'sma_50', 'sma_200']]
        y = df['target']

        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(X, y)

        model_path = os.path.join(MODELS_DIR, f"model_{symbol.replace('/', '_')}.joblib")
        dump(model, model_path)
        logger.info(f"Modelo salvo em {model_path}")
        return True

    async def handle_training_request(self, msg):
        try:
            data = json.loads(msg.data.decode())
            symbol = data.get('symbol')
            if symbol:
                success = await self.train_model(symbol)
                if success:
                    await self.js.publish("ml.training.finished", json.dumps({"symbol": symbol, "status": "success"}).encode())
            await msg.ack()
        except Exception as e:
            logger.error(f"Erro processando treino: {e}")

    async def run(self):
        await self.connect_nats()
        
        # Subscribe para requisições de treino
        await self.js.subscribe(
            "ml.training.request",
            durable="ML_TRAINER",
            cb=self.handle_training_request,
            manual_ack=True
        )
        logger.info("ML Training Service aguardando requisições em 'ml.training.request'...")

        while True:
            if self.nc.is_closed:
                await self.connect_nats()
            await asyncio.sleep(10)

if __name__ == "__main__":
    service = MLTrainingService()
    asyncio.run(service.run())
