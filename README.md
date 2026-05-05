# 🏋️ fb-ml-training

Serviço robusto de treinamento de **6 modelos independentes** de Machine Learning para predição de movimentos de preço em criptomoedas.

## 🎯 Objetivo

Treinar e manter 6 modelos de trading independentes, cada um otimizado para:
- **2 Estratégias**: Breakout (rompimento) e Mean Reversion (reversão à média)
- **3 Tiers de Ativos**: Major, Strong Alt, High Volatility

Cada modelo é treinado com dados reais de mercado, features engineered especificamente, e é pronto para inferências em produção.

## 📊 Modelos Treinados

| Modelo | Estratégia | Tier | Donchian/SMA | Ativos de Treino | Status |
|--------|-----------|------|--------------|------------------|--------|
| **breakout_v1** | Breakout | Major | 15 | BTC, ETH | ✅ Em desenvolvimento |
| **breakout_v2** | Breakout | Strong Alt | 20 | SOL, MATIC, AVAX, LINK, DOGE, ADA, XRP | ⏳ Próximo |
| **breakout_v3** | Breakout | High Vol | 30 | ARB, OP, LDO, ATOM, NEAR, INJ, PEPE, SHIB, MEME, GALA | ⏳ Próximo |
| **mean_reversion_v1** | Mean Reversion | Major | SMA 20 | BTC, ETH | ⏳ Próximo |
| **mean_reversion_v2** | Mean Reversion | Strong Alt | SMA 30 | SOL, MATIC, AVAX, LINK, DOGE, ADA, XRP | ⏳ Próximo |
| **mean_reversion_v3** | Mean Reversion | High Vol | SMA 40 | ARB, OP, LDO, ATOM, NEAR, INJ, PEPE, SHIB, MEME, GALA | ⏳ Próximo |

## 🏗️ Arquitetura

```
src/
├── models/                    # 6 MODELOS INDEPENDENTES
│   ├── breakout_v1/
│   │   ├── config.py         # Parâmetros v1 (Donchian 15, n_estimators 150, etc)
│   │   ├── dataset.py        # Dataset: cria features Breakout
│   │   ├── train.py          # Trainer: RandomForest v1
│   │   └── evaluate.py       # Script: treina + avalia + salva
│   ├── breakout_v2/
│   ├── breakout_v3/
│   ├── mean_reversion_v1/
│   ├── mean_reversion_v2/
│   └── mean_reversion_v3/
│
└── shared/                    # CÓDIGO REUTILIZÁVEL
    ├── config.py             # Tiers, símbolos, constantes globais
    ├── indicators.py         # RSI, ATR, Donchian, Bollinger, etc
    ├── data_fetcher.py       # Binance API com retry automático
    ├── base_dataset.py       # Classe base para datasets
    └── base_trainer.py       # Classe base para treinamento

data/
├── raw/                       # Dados brutos da Binance
└── processed/                 # Dados processados + features

models/                        # Modelos salvos (.joblib)
```

## 🚀 Como Treinar

### Treinar Breakout V1 (Primeiro Modelo de Teste)
```bash
cd fb-ml-training

# Executar avaliação completa (busca dados + treina + avalia)
python -m src.models.breakout_v1.evaluate
```

Este comando irá:
1. ✅ Buscar 1000 candles de 1h de BTC/USDT e ETH/USDT via Binance
2. ✅ Criar features de Breakout (Donchian 15, RSI, ATR, Volatilidade, Momentum)
3. ✅ Adicionar target label (+1% em 4h)
4. ✅ Separar train (70%) / validation (30%)
5. ✅ Treinar RandomForestClassifier v1
6. ✅ Avaliar em dados de validação
7. ✅ Exibir métricas (Accuracy, Precision, Recall, F1, AUC)
8. ✅ Salvar modelo em `/models/model_breakout_v1_BTC_USDT.joblib`
9. ✅ Salvar resultados em `data/processed/breakout_v1_results.json`

## 📈 Datasets Utilizados

### Fonte de Dados
- **Binance API** (via CCXT)
- **Timeframe**: 1h
- **Candles**: 1000 por símbolo (~41 dias)
- **Múltiplos ciclos**: Bull, bear, sideways markets

### Features por Estratégia

**Breakout V1/V2/V3:**
```
- Donchian High (período: 15/20/30)
- Donchian Low (período: 15/20/30)
- Donchian Mid
- Price to High (posição relativa)
- RSI (14)
- ATR (14)
- Volatilidade
- Momentum (5)
```

**Mean Reversion V1/V2/V3:**
```
- SMA (período: 20/30/40)
- RSI (14)
- RSI Smooth (EMA de 2)
- Desvio da SMA (Z-score)
- Bandas de Bollinger (±2σ)
- Posição em Bollinger (0-1)
```

### Target Label
```
target = 1 se close[t+4] > close[t] * 1.01  (subir > 1% em 4h)
target = 0 caso contrário
```

## 🔑 Configurações por Versão

### Breakout
| Parâmetro | V1 (Major) | V2 (Strong Alt) | V3 (High Vol) |
|-----------|-----------|-----------------|---------------|
| Donchian Period | 15 | 20 | 30 |
| n_estimators | 150 | 100 | 80 |
| max_depth | 6 | 5 | 4 |
| min_samples_split | 20 | 15 | 10 |

### Mean Reversion
| Parâmetro | V1 (Major) | V2 (Strong Alt) | V3 (High Vol) |
|-----------|-----------|-----------------|---------------|
| SMA Period | 20 | 30 | 40 |
| n_estimators | 150 | 100 | 80 |
| max_depth | 6 | 5 | 4 |
| min_samples_split | 20 | 15 | 10 |

## 📊 Esperado nas Avaliações

**Métricas de Validação Esperadas** (após treinamento com dados reais):
- `Accuracy`: 50-60% (market prediction é difícil)
- `Precision`: 45-65% (poucos falsos positivos)
- `Recall`: 40-70% (captura uptrends válidos)
- `F1`: 45-60% (balanço entre precision/recall)
- `AUC`: 55-70% (discrimina bem uptrend vs downtrend)

**Obs**: Modelos bem calibrados terão alta precision + recall balanceados, evitando overfitting.

## 🔄 Fluxo de Treino (Walk-Forward Validation)

```
1000 Candles (41 dias)
├─ Primeiros 700 (29 dias) → TRAIN
└─ Últimos 300 (12 dias) → VALIDATION (período futuro)
```

Isso garante que o modelo vê padrões reais de mercado sem data leakage.

## 📦 Arquivos Chave

| Arquivo | Função |
|---------|--------|
| `src/shared/config.py` | Tiers, símbolos, constantes por versão |
| `src/shared/data_fetcher.py` | Fetch robusto com retry da Binance |
| `src/shared/indicators.py` | Cálculos de RSI, ATR, Donchian, Bollinger |
| `src/shared/base_dataset.py` | Classe base: prepare(), add_target_label() |
| `src/shared/base_trainer.py` | Classe base: train(), evaluate(), save_model() |
| `src/models/breakout_v1/evaluate.py` | Script principal de treino/avaliação |

## 🐳 Docker

```bash
# Build
docker build -t fb-ml-training:latest .

# Run (treina breakout_v1)
docker run \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  fb-ml-training:latest \
  python -m src.models.breakout_v1.evaluate
```

## 📝 Próximos Passos

1. ✅ Treinar e validar **breakout_v1**
2. ⏳ Treinar e validar **breakout_v2**, **breakout_v3**
3. ⏳ Treinar e validar **mean_reversion_v1**, **v2**, **v3**
4. ⏳ Integrar com `fb-strategy-ml` (inferências em tempo real)
5. ⏳ Implementar retraining automático via NATS

---

*FinBot-Crypto - ML Training Layer (Enterprise-Grade)*
