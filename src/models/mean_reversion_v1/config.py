"""
Mean Reversion V1 - Major Tier (LSTM, classificacao binaria).

Target: 1 se retorno > 0 nas proximas 12h (direcao positiva).
Score: 2 * predict_proba - 1 (sigmoide → [-1, +1]).
"""
TIER = "Major"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 10000

RSI_PERIOD = 14
SMA_PERIOD = 60
BB_STD = 2
VOLUME_SMA_PERIOD = 20

LOOKAHEAD_CANDLES = 12

SEQ_LEN = 96
LSTM_HIDDEN = 256
LSTM_LAYERS = 3
DROPOUT = 0.4
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0005

FEATURES = [
    'rsi', 'rsi_smooth',
    'sma', 'deviation_from_sma',
    'bb_position', 'bb_width',
    'volume_ratio',
]
