"""
Mean Reversion V1 - 3 features RSI, lookahead 12h.
"""
TIER = "Major"
TIMEFRAME = "15m"
CANDLES_TO_FETCH = 6400
LOOKAHEAD_CANDLES = 48

SEQ_LEN = 96
LSTM_HIDDEN = 96
LSTM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    'rsi_14', 'rsi_smooth', 'rsi_14_4h',
    'funding_rate', 'funding_change',
]
