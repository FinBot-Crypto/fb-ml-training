import sys, asyncio, torch, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Users\Renan\PythonProjects\financas_crypto_bot\services\fb-ml-training')
from src.shared.data_fetcher import DataFetcher
from src.shared.config import MAJOR_TIER_SYMBOLS
from src.shared.indicators import calculate_rsi
from src.models.mean_reversion_v1.dataset import MeanReversionV1Dataset
from src.models.mean_reversion_v1.lstm_train import LSTMMeanReversion, make_sequences
from src.models.mean_reversion_v1 import config

device = 'cpu'
ckpt = torch.load(r'C:\Users\Renan\Downloads\model_mean_reversion_v1_lstm_Major.pt', map_location=device, weights_only=False)
cfg = ckpt.get('config', {})
model = LSTMMeanReversion(len(config.FEATURES), cfg.get('hidden', 96), cfg.get('layers', 1), 0).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
seq = cfg.get('seq_len', 144)
TP, SL = 0.025, 0.02

async def main():
    f = DataFetcher()
    all_trades = {s: [] for s in MAJOR_TIER_SYMBOLS}
    
    for s in MAJOR_TIER_SYMBOLS:
        df = await f.fetch_ohlcv(s, config.TIMEFRAME, config.CANDLES_TO_FETCH)
        if df is None: continue
        fr = await f.fetch_funding_rate_history(s, 1000)
        ds = MeanReversionV1Dataset(symbol=s).set_futures_data(funding_df=fr)
        X, y = ds.prepare(df)[1], ds.prepare(df)[3]
        rsi_raw = calculate_rsi(df['close'], 56).tail(len(X)).values
        close_all = df['close'].values
        Xs, ys = make_sequences(X, y, seq)
        with torch.no_grad():
            p = model(torch.from_numpy(Xs).to(device)).numpy().flatten()
        
        for i in range(len(p)):
            idx = i + seq + seq
            if idx >= len(close_all): continue
            entry = close_all[idx]
            max_f = -np.inf; min_f = np.inf
            for t in range(1, 49):
                if idx + t < len(close_all):
                    max_f = max(max_f, close_all[idx + t])
                    min_f = min(min_f, close_all[idx + t])
            rsi = rsi_raw[i + seq] if i + seq < len(rsi_raw) else 50
            all_trades[s].append({
                'dia': i // 96, 'score': p[i], 'rsi': rsi,
                'max_ret': (max_f/entry-1)*100, 'min_ret': (min_f/entry-1)*100
            })

    # Configs to test
    configs = [
        ('Config A: RSI<38 + score>=0.65 (arrojado)', 38, 0.65),
        ('Config B: RSI<35 + score>=0.75 (conservador)', 35, 0.75),
    ]
    
    for nome, rsi_t, sc_t in configs:
        print(f"\n{'='*70}")
        print(f"{nome}")
        print(f"TP={TP*100:.1f}% | SL={SL*100:.1f}%")
        print(f"{'='*70}")
        
        # Juntar trades de BTC + ETH por dia
        todos = []
        for s in MAJOR_TIER_SYMBOLS:
            for t in all_trades[s]:
                if t['rsi'] < rsi_t and t['score'] >= sc_t:
                    todos.append(t)
        
        df = pd.DataFrame(todos)
        if len(df) == 0:
            print("  Nenhum trade")
            continue
        
        n_dias = df['dia'].nunique()
        print(f"Total trades: {len(df)} em {n_dias} dias ({len(df)/max(n_dias,1):.1f}/dia)")
        
        # 5 dias aleatorios
        dias = sorted(df['dia'].unique())
        np.random.seed(42)
        escolhidos = sorted(np.random.choice(dias, min(5, len(dias)), replace=False))
        
        total_w, total_l, total_n = 0, 0, 0
        for d in escolhidos:
            day = df[df['dia'] == d]
            wins = ((day['max_ret'] >= TP*100) & (day['min_ret'] > -SL*100)).sum()
            losses = (day['min_ret'] <= -SL*100).sum()
            neutros = len(day) - wins - losses
            lucro_dia = wins * TP - losses * SL
            total_w += wins; total_l += losses; total_n += neutros
            print(f"  Dia {d+1:2d}: {len(day):2d} trades | {wins}W/{losses}L/{neutros}N | lucro={lucro_dia*100:+.1f}%")
        
        total = total_w + total_l
        if total > 0:
            print(f"  ---> Total 5 dias: {total_w}W/{total_l}L/{total_n}N | WR={total_w/max(total,1):.0%} | lucro={(total_w*TP-total_l*SL)*100:+.1f}%")

import asyncio; asyncio.run(main())
