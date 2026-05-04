"""
LSTM Trainer para Mean Reversion V1.
Score contínuo [-1, +1], loss MSE, saída tanh.
"""
import logging, os, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from . import config

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_sequences(X, y, seq_len):
    X_vals = X.values if isinstance(X, pd.DataFrame) else X
    y_vals = y.values if isinstance(y, pd.Series) else y
    n = len(X_vals)
    X_seq, y_seq = [], []
    for i in range(seq_len, n):
        X_seq.append(X_vals[i - seq_len:i])
        y_seq.append(y_vals[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


class LSTMMeanReversion(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        out = self.dropout(last)
        return torch.tanh(self.fc(out))


class MeanReversionV1LSTMTrainer:
    def __init__(self, models_dir):
        self.model_name = "mean_reversion_v1_lstm"
        self.tier = config.TIER
        self.models_dir = models_dir
        self.model = None
        self.model_path = None
        self.feature_names_ = None
        logger.info(f"LSTM Trainer inicializado: {self.model_name} ({self.tier})")

    def train(self, X_train, y_train, X_val, y_val):
        seq_len = config.SEQ_LEN
        self.feature_names_ = list(X_train.columns)
        n_features = X_train.shape[1]

        Xs_tr, ys_tr = make_sequences(X_train, y_train, seq_len)
        Xs_va, ys_va = make_sequences(X_val, y_val, seq_len)

        logger.info(f"  Sequencias criadas: {len(Xs_tr)} train, {len(Xs_va)} val")
        logger.info(f"  Formato: (batch, {seq_len}, {n_features})")

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(ys_tr)),
            batch_size=config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xs_va), torch.from_numpy(ys_va)),
            batch_size=config.BATCH_SIZE, shuffle=False
        )

        self.model = LSTMMeanReversion(
            input_size=n_features,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            dropout=config.DROPOUT,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        best_val_loss = float('inf')
        best_epoch = 0
        patience = 30
        wait = 0

        for epoch in range(config.EPOCHS):
            self.model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device).unsqueeze(1)
                    loss = criterion(self.model(Xb), yb)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    p_tr = self.model(torch.from_numpy(Xs_tr).to(device)).cpu().numpy().flatten()
                    p_va = self.model(torch.from_numpy(Xs_va).to(device)).cpu().numpy().flatten()
                    logger.info(f"  Epoca {epoch+1:3d}: loss_tr={train_loss:.4f} loss_val={val_loss:.4f} "
                                f"mse_tr={float(mean_squared_error(ys_tr, p_tr)):.4f} "
                                f"mse_val={float(mean_squared_error(ys_va, p_va)):.4f}")
            else:
                logger.info(f"  Epoca {epoch+1:3d}: loss_tr={train_loss:.4f} loss_val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(self.model.state_dict(), os.path.join(self.models_dir, f"{self.model_name}_best.pt"))
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"  Early stop na epoca {epoch+1} (melhor: {best_epoch+1})")
                    self.model.load_state_dict(
                        torch.load(os.path.join(self.models_dir, f"{self.model_name}_best.pt"))
                    )
                    break

        # Metrics (regressao)
        with torch.no_grad():
            train_pred = self.model(torch.from_numpy(Xs_tr).to(device)).cpu().numpy().flatten()
            val_pred = self.model(torch.from_numpy(Xs_va).to(device)).cpu().numpy().flatten()

        metrics = {
            'train_mse': float(mean_squared_error(ys_tr, train_pred)),
            'train_mae': float(mean_absolute_error(ys_tr, train_pred)),
            'val_mse': float(mean_squared_error(ys_va, val_pred)),
            'val_mae': float(mean_absolute_error(ys_va, val_pred)),
            'best_epoch': best_epoch + 1,
        }

        logger.info(f"  Train MSE: {metrics['train_mse']:.6f} | MAE: {metrics['train_mae']:.6f}")
        logger.info(f"  Val   MSE: {metrics['val_mse']:.6f} | MAE: {metrics['val_mae']:.6f}")

        return metrics

    def evaluate(self, X_val, y_val):
        if self.model is None:
            raise ValueError("Modelo nao foi treinado")
        seq_len = config.SEQ_LEN
        Xs_va, ys_va = make_sequences(X_val, y_val, seq_len)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(Xs_va).to(device)).cpu().numpy().flatten()
        metrics = {
            'val_mse': float(mean_squared_error(ys_va, pred)),
            'val_mae': float(mean_absolute_error(ys_va, pred)),
        }
        logger.info(f"  Avaliacao: MSE={metrics['val_mse']:.6f} MAE={metrics['val_mae']:.6f}")
        return metrics

    def save_model(self, tier_name):
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, f"model_{self.model_name}_{tier_name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_names': self.feature_names_,
            'config': {
                'seq_len': config.SEQ_LEN,
                'hidden': config.LSTM_HIDDEN,
                'layers': config.LSTM_LAYERS,
                'dropout': config.DROPOUT,
                'n_features': len(self.feature_names_),
            }
        }, self.model_path)
        logger.info(f"  Modelo salvo: {self.model_path}")
        return self.model_path

    def predict_score(self, X):
        """Retorna score [-1, +1] para cada candle."""
        seq_len = config.SEQ_LEN
        if len(X) < seq_len:
            return np.zeros(len(X))
        Xs, _ = make_sequences(X, np.zeros(len(X)), seq_len)
        with torch.no_grad():
            score = self.model(torch.from_numpy(Xs).to(device)).cpu().numpy().flatten()
        full = np.zeros(len(X))
        full[seq_len:] = score
        return full
