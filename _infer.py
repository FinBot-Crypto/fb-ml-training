"""
Inferencia: carrega modelo, busca dados de hoje, mostra resultados.
Uso: python _infer.py <caminho_do_modelo.pt>
"""
import sys, asyncio, logging, numpy as np, pandas as pd, torch
sys.path.insert(0, '.')

from src.shared.data_fetcher import DataFetcher
from src.shared.config import MAJOR_TIER_SYMBOLS
from src.models.mean_reversion_v1.dataset import MeanReversionV1Dataset
from src.models.mean_reversion_v1.lstm_train import LSTMMeanReversion, make_sequences
from src.models.mean_reversion_v1 import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/model_mean_reversion_v1_lstm_Major.pt'

checkpoint = torch.load(model_path, map_location=device, weights_only=False)

model = LSTMMeanReversion(
    input_size=len(config.FEATURES),
    hidden_size=config.LSTM_HIDDEN,
    num_layers=config.LSTM_LAYERS,
    dropout=0
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

async def main():
    f = DataFetcher()
    all_scores = []

    for symbol in MAJOR_TIER_SYMBOLS:
        print(f"\nBuscando {symbol}...")
        df = await f.fetch_ohlcv(symbol, config.TIMEFRAME, config.CANDLES_TO_FETCH)
        if df is None:
            continue
        ds = MeanReversionV1Dataset(symbol=symbol)
        X, y = ds.prepare(df)[1], ds.prepare(df)[3]  # X_val, y_val

        seq_len = config.SEQ_LEN
        if len(X) < seq_len:
            continue
        Xs, ys = make_sequences(X, y, seq_len)

        with torch.no_grad():
            proba = model(torch.from_numpy(Xs).to(device)).cpu().numpy().flatten()
        score = 2 * proba - 1

        # Pegar timestamps do ultimo dia
        last_n = min(96, len(score))
        recent = pd.DataFrame({
            'symbol': symbol,
            'score': score[-last_n:],
            'target': ys[-last_n:],
        })
        recent['acertou'] = ((recent['score'] > 0) & (recent['target'] == 1)) | \
                            ((recent['score'] < 0) & (recent['target'] == 0))
        all_scores.append(recent)

    if not all_scores:
        print("Sem dados")
        return

    df_result = pd.concat(all_scores).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"RELATORIO DE INFERENCIA - ULTIMOS DADOS")
    print(f"{'='*70}")

    for thresh in [0.3, 0.5, 0.7]:
        long_sigs = (df_result['score'] >= thresh).sum()
        short_sigs = (df_result['score'] <= -thresh).sum()

        long_ok = ((df_result['score'] >= thresh) & (df_result['acertou'] == True)).sum() if long_sigs > 0 else 0
        short_ok = ((df_result['score'] <= -thresh) & (df_result['acertou'] == True)).sum() if short_sigs > 0 else 0

        print(f"\nThreshold {thresh}:")
        print(f"  LONG:  {long_sigs} sinais, {long_ok} acertos ({long_ok/max(long_sigs,1):.0%})")
        print(f"  SHORT: {short_sigs} sinais, {short_ok} acertos ({short_ok/max(short_sigs,1):.0%})")
        print(f"  Total: {long_sigs+short_sigs} sinais")

    print(f"\nScore stats: media={df_result['score'].mean():.4f} "
          f"std={df_result['score'].std():.4f} "
          f"min={df_result['score'].min():.4f} "
          f"max={df_result['score'].max():.4f}")

    print(f"\nPrimeiros scores:")
    print(df_result.tail(12).to_string(index=False))

asyncio.run(main())
