"""
Mean Reversion V1 - Major Tier (LSTM).
Sequencias de 48h, target >1% em 12h, timeframe 1h.
"""
TIER = "Major"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 10000

RSI_PERIOD = 14
SMA_PERIOD = 60
BB_STD = 2
VOLUME_SMA_PERIOD = 20

TARGET_RETURN_PCT = 1.0
LOOKAHEAD_CANDLES = 12

SEQ_LEN = 48
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001

FEATURES = [
    'rsi', 'rsi_smooth',
    'sma', 'deviation_from_sma',
    'bb_position', 'bb_width',
    'volume_ratio',
]
