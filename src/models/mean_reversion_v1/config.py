"""
Mean Reversion V1 - 3 features RSI comprovadas, 20000 candles.
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
    'rsi_14', 'rsi_smooth', 'rsi_14_4h',
]
