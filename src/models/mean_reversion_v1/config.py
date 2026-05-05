"""
Mean Reversion V1 - 3 features RSI, 15m, 40000 candles.
"""
TIER = "Major"
TIMEFRAME = "15m"
CANDLES_TO_FETCH = 40000
LOOKAHEAD_CANDLES = 48  # 12h em 15m

SEQ_LEN = 96  # 24h de contexto
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    'rsi_14', 'rsi_smooth', 'rsi_14_4h',
]
