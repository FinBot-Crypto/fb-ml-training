"""
Mean Reversion V1 - RSI + Funding Rate + Open Interest.
"""
TIER = "Major"
TIMEFRAME = "15m"
CANDLES_TO_FETCH = 4000
LOOKAHEAD_CANDLES = 48

SEQ_LEN = 96
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    'rsi_14', 'rsi_smooth', 'rsi_14_4h',
    'deviation_sma',  # distancia da media
    'funding_rate', 'funding_change',
    'open_interest', 'oi_change_1h', 'oi_change_24h',
]
