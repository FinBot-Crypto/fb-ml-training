"""
Mean Reversion V1 - 18 features (rsi_24 removida), 20000 candles 1h.
"""
TIER = "Major"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 20000
LOOKAHEAD_CANDLES = 12

SEQ_LEN = 24
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    'rsi_14', 'rsi_smooth',
    'ret_12h', 'ret_24h',
    'dev_sma_20', 'dev_sma_60',
    'bb_pos_20', 'bb_width_20', 'atr_ratio',
    'vol_ratio', 'dist_24h_low', 'low_wick',
    'rsi_14_4h', 'bb_pos_20_4h', 'bb_width_20_4h',
    'ret_24h_4h', 'vol_ratio_4h', 'dist_48h_low_4h',
]
