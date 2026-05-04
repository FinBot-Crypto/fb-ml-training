"""
Teste LSTM para Mean Reversion V1.
Sequencias de 48h, target >1% em 12h, timeframe 1h.
"""
import asyncio, sys, logging, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Users\Renan\PythonProjects\financas_crypto_bot\services\fb-ml-training')

from src.shared.logging_config import setup_logging
from src.shared.data_fetcher import BinanceDataFetcher
from src.shared.config import MAJOR_TIER_SYMBOLS, MODELS_DIR
from src.models.mean_reversion_v1.dataset import MeanReversionV1Dataset
from src.models.mean_reversion_v1.lstm_train import MeanReversionV1LSTMTrainer, make_sequences
from src.models.mean_reversion_v1 import config

logger = setup_logging(level=logging.INFO)


async def main():
    fetcher = BinanceDataFetcher(testnet=False)
    all_X_tr, all_X_va = [], []
    all_y_tr, all_y_va = [], []

    for symbol in MAJOR_TIER_SYMBOLS:
        logger.info(f"Buscando {symbol}...")
        df = await fetcher.fetch_ohlcv(symbol, config.TIMEFRAME, config.CANDLES_TO_FETCH)
        if df is None: continue
        ds = MeanReversionV1Dataset(symbol=symbol)
        X_tr, X_va, y_tr, y_va = ds.prepare(df)
        all_X_tr.append(X_tr); all_y_tr.append(y_tr)
        all_X_va.append(X_va); all_y_va.append(y_va)

    X_train = pd.concat(all_X_tr).reset_index(drop=True)
    X_val = pd.concat(all_X_va).reset_index(drop=True)
    y_train = pd.concat(all_y_tr).reset_index(drop=True)
    y_val = pd.concat(all_y_va).reset_index(drop=True)

    logger.info(f"\nDataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_train.shape[1]} features")

    trainer = MeanReversionV1LSTMTrainer(models_dir=MODELS_DIR)
    metrics = trainer.train(X_train, y_train, X_val, y_val)

    eval_metrics = trainer.evaluate(X_val, y_val)
    model_path = trainer.save_model("Major")

    logger.info(f"\n{'='*60}")
    logger.info(f"RESUMO LSTM - Mean Reversion V1 (Major)")
    logger.info(f"  Train MSE: {metrics['train_mse']:.6f} | MAE: {metrics['train_mae']:.6f}")
    logger.info(f"  Val   MSE: {metrics['val_mse']:.6f} | MAE: {metrics['val_mae']:.6f}")
    logger.info(f"  Melhor epoca: {metrics['best_epoch']}")
    logger.info(f"  Modelo: {model_path}")
    logger.info(f"{'='*60}")

asyncio.run(main())

