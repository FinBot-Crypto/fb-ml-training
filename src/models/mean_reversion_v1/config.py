"""
Mean Reversion V1 - Major Tier (BTC, ETH)

Modelo: LSTM que preve direcao do RSI em 12h.
Metricas: AUC 0.83 | WR 74% (TP 2.5% / SL 2%)
Producao: RSI < 38 + score >= 0.65 → LONG
"""
TIER = "Major"
TIMEFRAME = "15m"
CANDLES_TO_FETCH = 6400
LOOKAHEAD_CANDLES = 48

SEQ_LEN = 144
LSTM_HIDDEN = 48
LSTM_LAYERS = 1
DROPOUT = 0.5
# BATCH_SIZE e EPOCHS serao ajustados pelo trainer se necessario
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001

FEATURES = [
    'rsi_14', 'rsi_smooth', 'rsi_14_4h',
    'btc_rsi_14',
    'btc_sma_ratio_12', 'btc_sma_ratio_24', 'btc_sma_ratio_36', 'btc_sma_ratio_48',
    'funding_rate', 'funding_change',
    'oi_change_1h', 'oi_change_24h'
]

