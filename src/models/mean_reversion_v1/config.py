"""
Mean Reversion V1 - 6 features, 1h, 20000 candles.
"""
TIER = "Major"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 20000
LOOKAHEAD_CANDLES = 12

SEQ_LEN = 24
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    # RSI basico
    'rsi_14', 'rsi_smooth', 'rsi_14_4h',
    # Novas features
    'rsi_divergence', 'bb_squeeze', 'cons_candle',
]
