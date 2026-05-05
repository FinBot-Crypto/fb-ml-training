"""
Mean Reversion V1 - 20 features, 2 timeframes, target balanceado.
"""
TIER = "Major"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 10000
LOOKAHEAD_CANDLES = 12

SEQ_LEN = 24
LSTM_HIDDEN = 96
LSTM_LAYERS = 1
DROPOUT = 0.4
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003

FEATURES = [
    # 1h - momentum
    'rsi_14', 'rsi_24', 'rsi_smooth',
    'ret_12h', 'ret_24h',
    # 1h - tendencia
    'dev_sma_20', 'dev_sma_60',
    # 1h - volatilidade
    'bb_pos_20', 'bb_width_20', 'atr_ratio',
    # 1h - volume e estrutura
    'vol_ratio', 'dist_24h_low', 'low_wick',
    # 4h - contexto
    'rsi_14_4h', 'bb_pos_20_4h', 'bb_width_20_4h',
    'ret_24h_4h', 'vol_ratio_4h', 'dist_48h_low_4h',
]
