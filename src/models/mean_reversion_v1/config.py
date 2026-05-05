"""
Mean Reversion V1 - RSI + Funding Rate + Open Interest.
"""
TIER = "Major"
TIMEFRAME = "5m"
CANDLES_TO_FETCH = 8640
LOOKAHEAD_CANDLES = 144  # 12h

SEQ_LEN = 288  # 24h
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
