"""
Mean Reversion V1 - 8 features de RSI comprovadas.
"""
TIER = "Major"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 10000
LOOKAHEAD_CANDLES = 12

SEQ_LEN = 24
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    # RSI puro
    'rsi_14', 'rsi_24', 'rsi_smooth',
    # RSI 4h
    'rsi_14_4h',
    # Derivados de RSI
    'stoch_rsi', 'rsi_slope', 'rsi_cross',
    # Volume-RSI
    'mfi_14',
]
