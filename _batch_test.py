"""
Teste 15 configuracoes diferentes, roda todas e mostra ranking.
"""
import asyncio, sys, logging, numpy as np, pandas as pd, torch, itertools, copy
sys.path.insert(0, '.')

logging.basicConfig(level=logging.WARNING)
torch.set_num_threads(4)

from src.shared.data_fetcher import DataFetcher
from src.shared.config import MAJOR_TIER_SYMBOLS, MODELS_DIR
from src.models.mean_reversion_v1.dataset import MeanReversionV1Dataset
from src.models.mean_reversion_v1.lstm_train import MeanReversionV1LSTMTrainer, make_sequences
from src.models.mean_reversion_v1 import config as base_config
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIGS = [
    # (nome, timeframe, candles, lookahead, seq_len, hidden, lr, features)
    
    # === Baseline (7 features, 15m, 6h) ===
    ('15m_6h_7feat_lr3', '15m', 4000, 24, 96, 64, 0.0003, 7),
    
    # === Timeframes ===
    ('5m_6h_7feat', '5m', 8640, 72, 288, 64, 0.0003, 7),
    ('1h_6h_7feat', '1h', 720, 6, 24, 64, 0.0003, 7),
    
    # === Lookaheads ===
    ('15m_3h_7feat', '15m', 4000, 12, 96, 64, 0.0003, 7),
    ('15m_12h_7feat', '15m', 4000, 48, 96, 64, 0.0003, 7),
    
    # === Learning rates ===
    ('15m_6h_7feat_lr1', '15m', 4000, 24, 96, 64, 0.001, 7),
    ('15m_6h_7feat_lr5', '15m', 4000, 24, 96, 64, 0.0001, 7),
    
    # === Model capacity ===
    ('15m_6h_7feat_h32', '15m', 4000, 24, 96, 32, 0.0003, 7),
    ('15m_6h_7feat_h128', '15m', 4000, 24, 96, 128, 0.0003, 7),
    
    # === Feature sets ===
    ('15m_6h_3feat', '15m', 4000, 24, 96, 64, 0.0003, 3),
    ('15m_6h_5feat', '15m', 4000, 24, 96, 64, 0.0003, 5),
    ('15m_6h_7nooi', '15m', 4000, 24, 96, 64, 0.0003, 6),
    
    # === Specials ===
    ('15m_6h_7feat_drop4', '15m', 4000, 24, 96, 64, 0.0003, 7),
    ('15m_6h_7feat_l2', '15m', 4000, 24, 96, 64, 0.0003, 7),
    ('15m_6h_7feat_bs64', '15m', 4000, 24, 96, 64, 0.0003, 7),
]

FEATURE_SETS = {
    3: ['rsi_14', 'rsi_smooth', 'rsi_14_4h'],
    5: ['rsi_14', 'rsi_smooth', 'rsi_14_4h', 'funding_rate', 'oi_change_1h'],
    6: ['rsi_14', 'rsi_smooth', 'rsi_14_4h', 'funding_rate', 'funding_change', 'oi_change_1h'],
    7: ['rsi_14', 'rsi_smooth', 'rsi_14_4h', 'funding_rate', 'funding_change', 'oi_change_1h', 'oi_change_24h'],
}

EPOCHS = 30
BATCH_SIZE = 64
DROPOUT = 0.2
LAYERS = 1
LOOKAHEAD_BY_TF = {'5m': 72, '15m': 24, '1h': 6}
SEQ_BY_TF = {'5m': 288, '15m': 96, '1h': 24}

async def run_one(name, tf, candles, la, seq, hidden, lr, n_feat):
    sys.stdout.write(f"\n[{name}] ")
    sys.stdout.flush()
    
    f = DataFetcher()
    X_tr, X_va, Y_tr, Y_va = [], [], [], []
    
    for symbol in MAJOR_TIER_SYMBOLS:
        df = await f.fetch_ohlcv(symbol, tf, candles)
        if df is None: continue
        fr = await f.fetch_funding_rate_history(symbol, 1000)
        oi = await f.fetch_open_interest_history(symbol, 1000)
        ds = MeanReversionV1Dataset(symbol=symbol).set_futures_data(funding_df=fr, oi_df=oi)
        base_config.TIMEFRAME = tf
        base_config.LOOKAHEAD_CANDLES = la
        base_config.FEATURES = FEATURE_SETS.get(n_feat, FEATURE_SETS[7])
        a,b,c,d = ds.prepare(df)
        X_tr.append(a); X_va.append(b); Y_tr.append(c); Y_va.append(d)
    
    if not X_tr: return None
    Xt = pd.concat(X_tr).reset_index(drop=True)
    Xv = pd.concat(X_va).reset_index(drop=True)
    Yt = pd.concat(Y_tr).reset_index(drop=True)
    Yv = pd.concat(Y_va).reset_index(drop=True)
    
    trainer = MeanReversionV1LSTMTrainer(models_dir='/tmp/models')
    trainer.model = MeanReversionV1LSTMTrainer.LSTMMeanReversion(len(base_config.FEATURES), hidden, LAYERS, DROPOUT).to(device)
    
    Xs_tr, ys_tr = make_sequences(Xt, Yt, seq)
    Xs_va, ys_va = make_sequences(Xv, Yv, seq)
    if len(Xs_tr) < 50 or len(Xs_va) < 20: return None
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)
    
    best_auc = 0
    for ep in range(min(EPOCHS, len(Xs_tr)//BATCH_SIZE + 5)):
        trainer.model.train()
        perm = np.random.permutation(len(Xs_tr))
        for i in range(0, len(Xs_tr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            Xb = torch.from_numpy(Xs_tr[idx]).to(device)
            yb = torch.from_numpy(ys_tr[idx]).to(device).unsqueeze(1)
            optimizer.zero_grad()
            criterion(trainer.model(Xb), yb).backward()
            optimizer.step()
        
        trainer.model.eval()
        with torch.no_grad():
            p = trainer.model(torch.from_numpy(Xs_va).to(device)).cpu().numpy().flatten()
        try:
            auc = roc_auc_score(ys_va, p)
            if auc > best_auc: best_auc = auc
        except:
            pass
    
    return best_auc

async def main():
    results = []
    for cfg in CONFIGS:
        try:
            auc = await run_one(*cfg)
            results.append((cfg[0], auc))
            sys.stdout.write(f"AUC={auc:.4f}\n" if auc else "FAIL\n")
        except Exception as e:
            results.append((cfg[0], None))
            sys.stdout.write(f"ERR:{str(e)[:30]}\n")
        sys.stdout.flush()
    
    print(f"\n{'='*70}")
    print(f"RANKING FINAL")
    print(f"{'='*70}")
    ranked = sorted([r for r in results if r[1] is not None], key=lambda x: -x[1])
    for i, (name, auc) in enumerate(ranked):
        print(f"  #{i+1:2d}  {name:<25} AUC={auc:.4f}")
    print(f"{'='*70}")

asyncio.run(main())
