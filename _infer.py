"""
Inferencia nos dados MAIS RECENTES (ultimas semanas).
Uso: python _infer.py <caminho_do_modelo.pt>
"""
import sys, asyncio, logging, numpy as np, pandas as pd, torch
sys.path.insert(0, '.')

from src.shared.data_fetcher import DataFetcher
from src.shared.config import MAJOR_TIER_SYMBOLS
from src.models.mean_reversion_v1.dataset import MeanReversionV1Dataset
from src.models.mean_reversion_v1.lstm_train import LSTMMeanReversion, make_sequences
from src.models.mean_reversion_v1 import config

logging.basicConfig(level=logging.WARNING)
device = torch.device('cpu')

model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/model_mean_reversion_v1_lstm_Major.pt'

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
seq_len_trained = checkpoint.get('config', {}).get('seq_len', config.SEQ_LEN)

model = LSTMMeanReversion(
    input_size=len(config.FEATURES),
    hidden_size=checkpoint.get('config', {}).get('hidden', config.LSTM_HIDDEN),
    num_layers=checkpoint.get('config', {}).get('layers', config.LSTM_LAYERS),
    dropout=0
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

async def main():
    f = DataFetcher()
    all_rows = []

    for symbol in MAJOR_TIER_SYMBOLS:
        df = await f.fetch_ohlcv(symbol, config.TIMEFRAME, config.CANDLES_TO_FETCH)
        if df is None: continue
        ds = MeanReversionV1Dataset(symbol=symbol)
        X, y = ds.prepare(df)[1], ds.prepare(df)[3]
        seq_len = seq_len_trained

        if len(X) < seq_len + 48: continue
        # Pegar apenas os ultimos 4 dias (96 candles de 1h)
        X_recent = X.tail(96)
        y_recent = y.tail(96)
        Xs, ys = make_sequences(X_recent, y_recent, seq_len)
        if len(Xs) == 0: continue

        with torch.no_grad():
            proba = model(torch.from_numpy(Xs).to(device)).numpy().flatten()
        score = 2 * proba - 1

        for i in range(len(score)):
            all_rows.append({
                'symbol': symbol,
                'score': round(float(score[i]), 4),
                'target': int(ys[i]),
                'acertou': bool((score[i] > 0 and ys[i] == 1) or (score[i] < 0 and ys[i] == 0))
            })

    if not all_rows:
        print("Sem dados para inferencia")
        return

    df = pd.DataFrame(all_rows)
    n = len(df)

    print(f"\n{'='*70}")
    print(f"INFERENCIA - {n} candles ({n//2} por ativo)")
    print(f"{'='*70}")

    long_sigs = (df['score'] >= 0.3).sum()
    short_sigs = (df['score'] <= -0.3).sum()
    print(f"\nSinais no periodo ({n} candles):")
    print(f"  LONG >= 0.3:  {long_sigs}  ({long_sigs/max(n,1):.1%})")
    print(f"  SHORT <= -0.3: {short_sigs}  ({short_sigs/max(n,1):.1%})")
    print(f"  Scores > 0:    {(df['score']>0).sum()}  ({(df['score']>0).sum()/max(n,1):.1%})")
    print(f"  Scores < 0:    {(df['score']<0).sum()}  ({(df['score']<0).sum()/max(n,1):.1%})")

    print(f"\nScore stats: media={df['score'].mean():.4f} "
          f"std={df['score'].std():.4f} "
          f"[{df['score'].min():.4f}, {df['score'].max():.4f}]")

    # Distribuicao
    print(f"\nDistribuicao dos scores:")
    print(f"  (-inf, -0.3]: {(df['score']<=-0.3).sum()}")
    print(f"  (-0.3, -0.1]: {((df['score']>-0.3)&(df['score']<=-0.1)).sum()}")
    print(f"  (-0.1,  0.1]: {((df['score']>-0.1)&(df['score']<=0.1)).sum()}  <- maioria aqui")
    print(f"  ( 0.1,  0.3]: {((df['score']>0.1)&(df['score']<=0.3)).sum()}")
    print(f"  ( 0.3,  inf): {(df['score']>0.3).sum()}")

    print(f"\nUltimos 20 resultados:")
    for _, r in df.tail(20).iterrows():
        s = f"{'LONG' if r['score']>0 else 'SHORT' if r['score']<0 else '--':>6}"
        ok = '+' if r['acertou'] else '-'
        print(f"  {r['symbol']:<10} score={r['score']:<+8.4f} {s:<6} target={int(r['target'])} {ok}")

asyncio.run(main())
