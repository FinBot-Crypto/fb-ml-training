# 🏋️ fb-ml-training

Microserviço responsável pelo treinamento assíncrono de modelos de Machine Learning para predição de movimentos de preço.

## 🎯 Objetivo
O `fb-ml-training` automatiza o ciclo de vida dos modelos. Ele é disparado sob demanda ou agendamento para re-treinar a inteligência do bot com os dados mais recentes do mercado, garantindo que as estratégias se adaptem às novas condições de volatilidade.

## 🚀 Funcionalidades
- **Fetching de Big Data**: Baixa até 1000 períodos históricos (OHLCV) via Binance API.
- **Feature Engineering**: Transforma dados brutos em indicadores preditivos (RSI, SMA_50, SMA_200).
- **Treinamento Supervisionado**: Utiliza `RandomForestClassifier` para predição binária de subida de preço (>1% em 4h).
- **Serialização de Modelos**: Salva modelos otimizados em formato `.joblib` em volume persistente.
- **Pipeline de Eventos**: Escuta `ml.training.request` e notifica a conclusão em `ml.training.finished`.

## 🔄 Fluxo CI/CD
1. **Push para `main`**: Dispara o workflow de deploy centralizado.
2. **Build Docker**: Instala pacotes pesados como `scikit-learn` e `pandas`.
3. **Deploy via SSH**: Atualiza o serviço na VPS.
4. **Volume Persistente**: O serviço mapeia `/app/models` para o host para garantir que os modelos sobrevivam a reinícios de containers.

## 🔑 Variáveis e Secrets Necessárias
| Nome | Descrição | Local |
|------|-----------|-------|
| `NATS_URL` | Endereço do NATS | `.env` |
| `MODELS_DIR` | Caminho para salvar os modelos | Docker |
| `VPS_SSH_*` | Credenciais de acesso à VPS | GitHub Secrets |

## 🏗️ Infraestrutura Utilizada
- **NATS JetStream**: Comunicação confiável para disparar tarefas de treino longas.
- **Scikit-Learn**: Biblioteca de Machine Learning.
- **Docker Volumes**: Persistência física dos modelos treinados.

## 📡 Simulação de Trigger (NATS)
Para disparar um treino manualmente via NATS:
```bash
nats pub ml.training.request '{"symbol": "BTC/USDT"}'
```

---
*FinBot-Crypto - ML Training Layer*
