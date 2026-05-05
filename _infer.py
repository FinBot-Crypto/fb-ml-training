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

    for thresh in sorted([0.75, 0.7, 0.6, 0.5, 0.4, 0.3], reverse=True):
        long = df[df['score'] >= thresh]
        short = df[df['score'] <= -thresh]
        total = len(long) + len(short)
        if total == 0: continue
        long_ok = long['acertou'].sum()
        short_ok = short['acertou'].sum()
        pct = (long_ok + short_ok) / total
        print(f"  >= {thresh:.2f}: {total:3d} sinais ({len(long)}L+{len(short)}S) "
              f"acertos={long_ok+short_ok:.0f} erros={total-(long_ok+short_ok):.0f} "
              f"acc={pct:.0%}")

    print(f"\nScore stats: media={df['score'].mean():.4f} "
          f"std={df['score'].std():.4f} "
          f"[{df['score'].min():.4f}, {df['score'].max():.4f}]")

    print(f"\nUltimos 16 scores ({len(df[df['symbol']=='BTC/USDT'])} BTC + {len(df[df['symbol']=='ETH/USDT'])} ETH):")
    for _, r in df.tail(16).iterrows():
        s = 'LONG' if r['score'] > 0.3 else ('SHORT' if r['score'] < -0.3 else '--')
        ok = '+' if r['acertou'] else '-'
        print(f"  {r['symbol']:<10} score={r['score']:<8.4f} {s:<6} target={int(r['target'])} {ok}")

asyncio.run(main())
