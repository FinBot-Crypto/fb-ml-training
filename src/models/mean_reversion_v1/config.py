"""
Mean Reversion V1 - RSI + Funding Rate + Open Interest.
"""
TIER = "Major"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 1000
LOOKAHEAD_CANDLES = 12

SEQ_LEN = 24
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    # RSI
    'rsi_14', 'rsi_smooth', 'rsi_14_4h',
    # Funding rate
    'funding_rate', 'funding_change',
    # Open interest
    'open_interest', 'oi_change_1h', 'oi_change_24h',
]
