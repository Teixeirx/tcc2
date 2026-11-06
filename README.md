# CriptoIA — TCC II

Sistema de análise técnica de criptomoedas com IA multimodal, backtesting histórico e comparação de estratégias.

> Trabalho de Conclusão de Curso II — Ciência da Computação (UFN)
>
> **Status:** implementação concluída para os entregáveis de TCC II; resultados experimentais reais do proxy 4×4, do Gemini oficial em `1d` (amostral) e do complemento exploratório em `1h`/`4h` gerados e persistidos em `CriptoIa/resultados/`. Este repositório é acadêmico e não constitui recomendação de investimento.

---

## 1. Visão geral

O CriptoIA integra:

1. **Coleta de dados** OHLCV da Binance via CCXT (com paginação para períodos longos).
2. **Indicadores técnicos** em Pandas: MM20, MM50, RSI-14, MACD/Sinal e Bandas de Bollinger.
3. **Análise multimodal** com Google Gemini (gráfico PNG + indicadores numéricos). O modelo é configurável por `MODELO_GEMINI`; a execução registrada neste repositório usou `gemini-3.5-flash-lite`.
4. **Dashboard HTML** interativo com Plotly.
5. **Backtesting determinístico** dos sinais técnicos com prevenção de look-ahead bias (Experimento 1, grade 4×4).
6. **Backtesting multimodal real do Gemini** em amostragem semanal do timeframe diário (Experimento 2) e complemento exploratório em `1h`/`4h` numa janela curta (Experimento 3).
7. **Comparação com Buy-and-Hold e LSTM real em PyTorch**, no mesmo período out-of-sample.
8. **Análise paralela** de múltiplos pares/timeframes com isolamento de falhas.
9. **Interface web Flask** com login, CSRF, sessões protegidas, histórico e visualização dos resultados.

### Três experimentos de backtesting — não confundir

**Experimento 1 — proxy determinístico (4 pares × 4 timeframes).** Usa uma **estratégia determinística de indicadores como proxy** da lógica técnica do CriptoIA. Ele **não reproduz as respostas do Gemini**, pois isso exigiria uma chamada multimodal por candle (inviável em custo e tempo para ~13 mil candles por par no timeframe de 1h). A acurácia é medida no **candle seguinte** (`close[t+1]/close[t] − 1`), e não no prazo de `tempo_estimado`. Esses números **não devem ser apresentados como acurácia das recomendações do Gemini**.

**Experimento 2 — Gemini multimodal real (4 pares, 1d, amostral).** Faz uma chamada real ao Gemini para 63 pontos semanais por par (252 inferências no total), sempre com uma janela de 90 candles estritamente anteriores ao ponto de decisão. Avalia a direção no **prazo de `tempo_estimado`**, o atingimento do `preco_alvo`, o retorno simulado e o benchmark Buy-and-Hold na mesma janela. É o experimento que responde diretamente à métrica (a) do TCC1, mas cobre apenas o timeframe diário e uma amostra, não a grade 4×4 completa.

**Experimento 3 — complemento exploratório (4 pares, 1h/4h, 4 meses).** Faz 304 inferências reais do Gemini (160 em `1h` e 144 em `4h`) entre julho e outubro de 2026, com amostragem a cada 3 dias e a mesma janela de 90 candles por análise. O objetivo é complementar o TCC com evidência intraday, **sem substituir** o Experimento 2: a janela curta e o período diferente não são diretamente comparáveis aos 18 meses do `1d` oficial. Os resultados ficam em arquivos separados (`backtesting_gemini_intraday_tcc.*`).

---

## 2. Estrutura do projeto

```text
tcc2/
├── README.md
├── .gitignore
└── CriptoIa/
    ├── principal.py                 # CLI: análise, backtesting, comparação, múltiplos
    ├── RELATORIO_TECNICO.md         # relatório técnico detalhado do TCC II
    ├── requisitos.txt               # dependências fixadas
    ├── .env.example                 # modelo de configuração (sem segredos)
    ├── docs/                        # fonte LaTeX do TCC (Tcc1.tex)
    ├── coverage/                    # artefatos de cobertura (gerados)
    ├── backend/
    │   ├── coleta_dados.py          # CCXT/Binance
    │   ├── analise_tecnica.py       # indicadores e mínimo de 50 candles
    │   ├── gerador_grafico.py       # candlestick PNG
    │   ├── analisador_ia.py         # Gemini + validação + fallback MANTER
    │   ├── backtesting.py           # backtesting paginado e métricas (Experimento 1)
    │   ├── backtesting_gemini.py    # backtesting multimodal real Gemini (Experimentos 2 e 3)
    │   ├── comparacao.py            # Buy-and-Hold + LSTM PyTorch walk-forward
    │   └── multiplos_ativos.py      # análise paralela e relatório consolidado
    ├── frontend/
    │   └── gerador_dashboard.py     # dashboard HTML/Plotly
    ├── webapp/
    │   ├── app.py                   # rotas Flask, CSRF, segurança
    │   ├── auth.py                  # credenciais e proteção de sessão
    │   ├── history_store.py         # persistência atômica de históricos
    │   ├── templates/
    │   └── static/
    ├── tests/
    │   ├── unit/                    # CT01–CT06 e testes adicionais
    │   └── integration/             # pipeline com I/O mockado
    └── resultados/
        ├── dashboard.html                # dashboard interativo gerado (frontend)
        ├── grafico_bitcoin.png          # gráfico candlestick gerado (Gemini)
        ├── backtesting_tcc.json/csv        # Experimento 1 — proxy determinístico
        ├── backtesting_relatorio.txt
        ├── backtesting_resumo.txt
        ├── comparacao_tcc.json/csv         # Experimento 1 — comparação com B&H e LSTM
        ├── comparacao_relatorio.txt
        ├── comparacao_resumo.txt
        ├── backtesting_gemini_tcc.json/csv # Experimento 2 — Gemini multimodal real (1d)
        ├── backtesting_gemini_resumo.txt
        ├── backtesting_gemini_intraday_tcc.json/csv # Experimento 3 — exploratório (1h/4h)
        ├── backtesting_gemini_intraday_tcc_resumo.txt
        └── gemini_backtest_cache.json      # Cache de inferências (retomada automática)
```

---

## 3. Instalação

Requisitos: Python 3.12+ e conexão com internet apenas para executar análises/backtesting reais. Os testes automatizados **não** dependem de internet.

```powershell
cd CriptoIa
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch CPU (recomendado para não baixar CUDA)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r requisitos.txt
```

Em Linux/macOS, use `source .venv/bin/activate` e o mesmo comando de instalação do PyTorch.

### Configuração

```powershell
Copy-Item .env.example .env
```

Preencha no `.env`:

- `BINANCE_API_KEY` e `BINANCE_SECRET_KEY` (opcionais para leitura pública; usadas se fornecidas);
- `GEMINI_API_KEY` (obrigatória para análise multimodal real);
- `MODELO_GEMINI` (opcional; padrão `gemini-3.5-flash-lite`);
- `WEB_ADMIN_USER` e `WEB_ADMIN_PASS` (senha forte);
- `FLASK_SECRET_KEY`, gerada com:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

O arquivo `.env` **não é versionado**. O `.env.example` contém apenas placeholders.

---

## 4. Uso

### 4.1 Análise individual

```powershell
python principal.py
python principal.py --par ETH/USDT --intervalo 4h --quantidade 90
```

O fluxo coleta dados, calcula indicadores, gera o PNG, chama o Gemini, gera o dashboard e abre o arquivo no navegador. Falhas retornam fallback estruturado `acao="MANTER"`, são persistidas em `resultados/fallback_analise.json` e não geram traceback não tratado.

### 4.2 Backtesting

```powershell
# Um par/timeframe
python principal.py --backtesting --par BTC/USDT --intervalo 1d

# Grade completa do TCC: 4 pares × 4 timeframes
python principal.py --backtesting --grade --max-workers 2
```

Saídas: `resultados/backtesting_tcc.json`, `.csv` e relatório textual gerado pelo próprio comando.

### 4.2.1 Backtesting multimodal real do Gemini (Experimento 2)

```powershell
# 4 ativos, timeframe 1d, amostragem semanal (63 pontos por par)
python principal.py --backtesting-gemini --grade --passo 7 --delay 2.0
```

Saídas: `resultados/backtesting_gemini_tcc.json`, `.csv`, `backtesting_gemini_resumo.txt` e o cache `gemini_backtest_cache.json`. O cache permite retomar a execução sem repetir chamadas já concluídas. Sem `--grade`, use `--par` e `--intervalo` para um único par.

**Por que o Experimento 2 ficou no `1d`?** O `--passo` é expresso em **dias** e convertido para candles conforme o timeframe. Para o período do TCC, uma amostragem semanal exigiria aproximadamente:

| Timeframe | Candles no período | Passo semanal | Pontos por par | Chamadas (4 pares) |
|---|---:|---:|---:|---:|
| `1h` | 13.104 | 168 candles | ~78 | ~312 |
| `4h` | 3.276 | 42 candles | ~76 | ~304 |
| `1d` | 546 | 7 candles | 63 | 252 |
| `1w` | 78 | 1 candle | — | inviável* |

\* O `1w` tem apenas 78 candles no período, abaixo do mínimo de 110 exigido pela janela de 90 candles passados + 15 futuros. O `1h` e o `4h` funcionariam, mas somariam ~616 chamadas adicionais ao Gemini no recorte de 18 meses. Como o trabalho usa o **modelo gratuito**, com limite de requisições/tokens, a opção metodológica do Experimento 2 foi concentrar a inferência real no `1d` e usar o proxy determinístico para a grade 4×4. O Experimento 3 contorna parcialmente essa limitação com uma janela curta de 4 meses (Seção 4.2.2). Essa é uma limitação declarada, não um resultado omitido.

**Importante:** a coluna “Pontos por par” é a quantidade de **decisões avaliadas**, não o tamanho do contexto enviado ao modelo. Em todas as linhas, cada inferência continua recebendo a mesma janela de **90 candles**; o que muda é o espaçamento entre os pontos de decisão. Menos pontos reduzem a confiança estatística do resultado, mas não a quantidade de candles que o Gemini analisa em cada chamada.

### 4.2.2 Experimento 3 — complemento exploratório intraday (1h/4h)

O Experimento 2 oficial continua restrito ao `1d`. Para complementar o TCC com uma amostra **real** do Gemini em `1h` e `4h`, o módulo aceita período e timeframes próprios, com saída separada:

```powershell
python principal.py --backtesting-gemini --grade `
  --intervalos 1h,4h `
  --inicio 2026-07-01 --fim 2026-10-31 `
  --passo 3 --delay 2.0 `
  --prefixo-saida backtesting_gemini_intraday_tcc
```

- **Status:** executado em 06/11/2026; 304 inferências reais (160 em `1h`, 144 em `4h`) e **zero fallbacks**.
- **Período:** 4 meses (01/07/2026 a 31/10/2026), fixo para reprodutibilidade.
- **Amostragem:** a cada 3 dias; cada inferência continua usando **90 candles**.
- **Aquecimento:** a primeira decisão ocorre após 90 candles (≈04/07 em `1h` e ≈16/07 em `4h`); o período efetivo de decisões é menor que a janela de coleta.
- **Volume:** 40 pontos/par em `1h`, 36 em `4h`, 304 chamadas nos 4 pares.
- **Saídas separadas:** `backtesting_gemini_intraday_tcc.json`, `.csv` e `_resumo.txt`; o cache compartilhado permite retomar sem repetir chamadas.
- **Comparação:** Buy-and-Hold e proxy são recalculados na **mesma janela curta** de cada par/timeframe.
- **Escopo:** exploratório; não substitui o Experimento 2 de 18 meses e não deve ser apresentado como resultado principal.

**Resultado em uma linha:** acurácia de prazo > 50% em **7 de 8** combinações; retorno não negativo em **3 de 8**; o Gemini superou o Buy-and-Hold em **1 de 8**. A tabela completa e a leitura estão na Seção 6.4. O resultado oficial em `1d` permanece inalterado.

O cache persiste cada resposta e permite retomar a execução em etapas. Também é possível usar `--passo 2` (mais pontos, ~452 chamadas) ou `--max-amostras` para um teste piloto.

### 4.3 Comparação CriptoIA × Buy-and-Hold × LSTM

```powershell
# Um par/timeframe
python principal.py --comparacao --par BTC/USDT --intervalo 1d

# Grade completa (LSTM com 20 épocas, seq_len 24)
python principal.py --comparacao --grade
```

Saídas: `resultados/comparacao_tcc.json`, `.csv`, além de `comparacao_relatorio.txt` e `comparacao_resumo.txt`.

Parâmetros opcionais do LSTM:

```powershell
python principal.py --comparacao --par BTC/USDT --intervalo 1d `
  --epochs 20 --seq-len 24 --hidden-size 32 --passo-retreino 1000
```

### 4.4 Múltiplos ativos em paralelo

```powershell
python principal.py --multiplos --pares BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT `
  --timeframes 1h,4h,1d,1w --max-workers 3
```

Sem `--usar-ia`, o modo múltiplos não gera gráficos nem chama o Gemini; com a flag, gera e analisa. O relatório consolidado é impresso; use `--salvar-csv` para gravá-lo.

### 4.5 Interface web

```powershell
python -m webapp.app
```

Acesse `http://127.0.0.1:5000`. Rotas:

- `/login` — autenticação;
- `/` — dashboard;
- `/analise` — execução de análise;
- `/historico` — histórico persistido;
- `/backtesting` — simulação sob demanda e grade TCC persistida.

Para produção, não use o servidor de desenvolvimento do Flask. Coloque atrás de um servidor WSGI (Waitress/Gunicorn) e HTTPS, com `CRIPTOIA_ENV=production`, `SESSION_COOKIE_SECURE=1` e `FLASK_DEBUG=0`.

---

## 5. Metodologia

### 5.1 Backtesting anti look-ahead

- O sinal é calculado no fechamento do candle `t`, usando **somente** indicadores até `t`.
- O resultado é medido com o retorno futuro `close[t+1]/close[t] − 1` (`shift(-1)`).
- O retorno futuro nunca é usado para gerar o sinal.
- Estratégia long-only: sinal `+1` fica comprado; `0`/`-1` fica em caixa.
- Não há custos de transação, alavancagem, spread ou slippage.
- O período solicitado é jan/2025 a jun/2026; a coleta é paginada (limite de 1.000 candles/requisição) e marcada como incompleta se falhar no meio.

### 5.2 LSTM

- Rede LSTM real em PyTorch, treinada do zero em CPU.
- Features: RSI normalizado, diferença MACD/Sinal, distância relativa MM20/MM50, posição nas Bandas de Bollinger e retorno do candle anterior.
- Alvo: direção do retorno do próximo candle.
- Normalização ajustada apenas no treino (`StandardScaler`).
- Walk-forward: o modelo é retreinado periodicamente usando apenas dados até o candle de decisão.
- Janela comum de comparação: últimos 40% dos dados (período out-of-sample do LSTM). CriptoIA e Buy-and-Hold são recalculados exatamente nessa janela.
- Em timeframes semanais, `seq_len` é reduzido automaticamente e registrado em `seq_len_solicitado`/`aviso_seq_len`.

### 5.3 Critérios de sucesso do TCC

O TCC1 define os critérios por ativo, mas não especifica qual timeframe deve representar o resumo. Para não criar um parâmetro inexistente, este trabalho reporta os 16 resultados brutos do proxy e usa o timeframe diário como **recorte operacional** do resumo (quando disponível). Os critérios são:

1. acurácia direcional > 50% em pelo menos 3 dos 4 ativos;
2. retorno acumulado não negativo no período de avaliação.

**Experimento 1 — proxy determinístico (recorte 1d):**

- **Acurácia > 50%:** 2 de 4 ativos (BTC 56.5%, DOGE 55.2%; ETH 47.4%, SOL 49.6%).
- **Retorno não negativo:** 1 de 4 ativos (BTC +16.20%; ETH −15.83%, SOL −39.06%, DOGE −5.08%).
- **Conclusão:** os dois critérios **não** foram atendidos pela estratégia proxy.

**Experimento 2 — Gemini multimodal real (1d, 63 pontos semanais por par):**

- **Acurácia de prazo > 50%:** 3 de 4 ativos (BTC 62.2%, ETH 54.2%, SOL 60.4%; DOGE 49.0%).
- **Retorno não negativo:** 0 de 4 ativos (BTC −30.97%, ETH −54.95%, SOL −64.07%, DOGE −57.46%).
- **Conclusão:** o critério de acurácia foi atingido, mas o de retorno **não**; portanto, o conjunto dos dois critérios continua **não atendido**.
- **Ressalva:** o experimento cobre apenas 1d e uma amostra semanal; não substitui a grade 4×4 completa.

**Experimento 3 — Gemini multimodal real (1h/4h, 304 pontos, janela exploratória de 4 meses):**

- **Acurácia de prazo > 50%:** 7 de 8 combinações (1h: 3/4; 4h: 4/4).
- **Retorno não negativo:** 3 de 8 combinações (1h: 1/4; 4h: 2/4).
- **Buy-and-Hold:** o Gemini superou o benchmark em apenas 1 de 8 combinações (DOGE 1h).
- **Conclusão:** nos dois timeframes, o critério de acurácia foi atingido isoladamente, mas o de retorno **não**; o conjunto dos critérios continua **não atendido**.
- **Ressalva:** janela curta e exploratória, com período diferente do `1d` oficial; não é uma comparação direta com os 18 meses do Experimento 2.

Os números do Experimento 1 são da estratégia proxy, **não** das respostas do Gemini. Os do Experimento 2 e 3 são de inferência real do Gemini, mas amostrais; o Experimento 2 é o resultado oficial em `1d`.

---

## 6. Resultados reais (atualizados em 06/11/2026)

### 6.1 Experimento 1 — proxy determinístico no timeframe diário

| Par | Acurácia | Retorno | Max. Drawdown | Operações |
|---|---:|---:|---:|---:|
| BTC/USDT | 56.5% | +16.20% | −14.08% | 71 |
| ETH/USDT | 47.4% | −15.83% | −28.70% | 60 |
| SOL/USDT | 49.6% | −39.06% | −39.85% | 65 |
| DOGE/USDT | 55.2% | −5.08% | −41.04% | 80 |

A tabela completa (16 combinações, incluindo 1h/4h/1w) está em `resultados/backtesting_resumo.txt` e `resultados/backtesting_tcc.json`. Todas as 16 coletas tiveram cobertura de 100% do período estimado.

### 6.2 Experimento 1 — comparação na janela comum (timeframe diário)

Janela comum do diário: 2025-12-13 a 2026-06-29.

| Par | CriptoIA retorno / acurácia | Buy-and-Hold | LSTM retorno / acurácia | Baseline LSTM |
|---|---:|---:|---:|---:|
| BTC/USDT | +1.28% / 54.0% | −35.03% | −25.37% / 52.3% | 53.8% |
| ETH/USDT | +2.25% / 46.5% | −49.53% | −48.59% / 46.7% | 53.8% |
| SOL/USDT | −14.05% / 43.3% | −44.67% | −34.11% / 47.2% | 54.8% |
| DOGE/USDT | +15.26% / 56.4% | −48.16% | −5.27% / 55.8% | 59.3% |

Na grade completa de 16 combinações:

- CriptoIA teve retorno maior que Buy-and-Hold em **15 de 16** combinações;
- LSTM teve acurácia > 50% em **10 de 16**, mas superou o baseline de classe majoritária em apenas **6 de 16**;
- CriptoIA teve acurácia > 50% em **3 de 16** combinações;
- os resultados semanais têm amostra muito pequena (78 candles, 28 válidos após warm-up) e não devem ser generalizados.

Esses resultados não demonstram superioridade estatística; são uma avaliação exploratória com uma única configuração de LSTM e sem custos de transação.

### 6.3 Experimento 2 — Gemini multimodal real (1d, amostral)

Modelo registrado: `gemini-3.5-flash-lite`. Foram 63 pontos de decisão por par (252 no total), amostrados a cada 7 dias, com janela de 90 candles anteriores e zero fallbacks na versão final. A janela efetiva começa em 2025-04-01 (após o warm-up de 90 candles) e termina no último horizonte de cada par (2026-06-24 ou 2026-06-30).

| Par | Amostras | Comp./Vend./Mant. | Acur. t+1 | Acur. prazo | Alvo atingido | Retorno Gemini | Buy-and-Hold (mesma janela) | Concord. proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC/USDT | 63 | 38 / 7 / 18 | 48.9% | 62.2% | 39.5% | −30.97% | −31.16% | 42.9% |
| ETH/USDT | 63 | 38 / 10 / 15 | 54.2% | 54.2% | 47.4% | −54.95% | −17.48% | 33.3% |
| SOL/USDT | 63 | 39 / 9 / 15 | 54.2% | 60.4% | 48.7% | −64.07% | −46.23% | 31.7% |
| DOGE/USDT | 63 | 43 / 6 / 14 | 51.0% | 49.0% | 32.6% | −57.46% | −56.25% | 25.4% |

Leitura:

- A acurácia no prazo estimado superou 50% em **3 de 4** ativos (BTC, ETH e SOL), atendendo isoladamente ao primeiro critério do TCC1.
- O retorno acumulado foi negativo em **4 de 4** ativos, reprovando o segundo critério; logo, o conjunto dos critérios continua **não atendido**.
- Contra o Buy-and-Hold na mesma janela, o Gemini teve retorno maior apenas em BTC (+0.19 p.p.); ficou abaixo em ETH (−37.47 p.p.), SOL (−17.84 p.p.) e DOGE (−1.21 p.p.).
- O retorno é composto por decisão amostral; como o passo é de 7 dias e o horizonte pode chegar a 14 dias (ou mais), há sobreposição de horizontes. Não há custos de transação, alavancagem ou posição vendida.
- A concordância com o proxy de indicadores ficou entre 25.4% e 42.9%, indicando que as decisões do Gemini não são reproduzíveis por regras fixas de MM/RSI/MACD.

### 6.4 Experimento 3 — Gemini multimodal real (1h/4h exploratório)

Modelo `gemini-3.5-flash-lite`; 304 inferências (160 em `1h`, 144 em `4h`), zero fallbacks. Período de coleta: 01/07/2026 a 31/10/2026, com amostragem a cada 3 dias e 90 candles de contexto por decisão. A primeira decisão ocorre após o warm-up de 90 candles; em `1h`, isso corresponde a ≈04/07/2026, e em `4h`, a ≈16/07/2026.

| TF | Par | Amostras | Comp./Vend./Mant. | Acur. t+1 | Acur. prazo | Alvo | Retorno Gemini | Buy-and-Hold (mesma janela) | Concord. proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1h` | BTC/USDT | 40 | 19 / 6 / 15 | 64.0% | 64.0% | 5.3% | −0.71% | +23.00% | 37.5% |
| `1h` | ETH/USDT | 40 | 19 / 6 / 15 | 56.0% | 48.0% | 10.5% | −1.89% | +24.31% | 65.0% |
| `1h` | SOL/USDT | 40 | 19 / 11 / 10 | 56.7% | 70.0% | 15.8% | +4.55% | +17.77% | 60.0% |
| `1h` | DOGE/USDT | 40 | 15 / 3 / 22 | 66.7% | 61.1% | 0.0% | −0.81% | −4.62% | 45.0% |
| `4h` | BTC/USDT | 36 | 17 / 5 / 14 | 45.5% | 59.1% | 29.4% | +5.35% | +20.12% | 41.7% |
| `4h` | ETH/USDT | 36 | 17 / 4 / 15 | 61.9% | 57.1% | 41.2% | +11.22% | +17.31% | 55.6% |
| `4h` | SOL/USDT | 36 | 19 / 8 / 9 | 37.0% | 55.6% | 10.5% | −2.08% | +24.73% | 50.0% |
| `4h` | DOGE/USDT | 36 | 12 / 3 / 21 | 33.3% | 53.3% | 8.3% | −9.49% | +2.30% | 50.0% |

**Não confundir contagem de combinações com taxa de acerto.** O “7 de 8” significa 7 das 8 combinações par × timeframe; **não** significa 87,5% de acertos. Agregando todos os pontos direcionais, a acurácia de prazo é 108/183 = 59,0% (`1h`: 60/98 = 61,2%; `4h`: 48/85 = 56,5%) e a de `t+1` é 97/183 = 53,0%.

Leitura:

- A acurácia de prazo superou 50% em **7 de 8** combinações: `1h` 3/4 (BTC, SOL e DOGE) e `4h` 4/4. O ponto fora foi ETH `1h` (48.0%).
- O retorno não foi negativo em **3 de 8** combinações: SOL `1h` (+4.55%), BTC `4h` (+5.35%) e ETH `4h` (+11.22%). Nos demais, ficou entre −0.71% e −9.49%.
- O Buy-and-Hold foi superado em apenas **1 de 8** combinações: DOGE `1h` (−0.81% vs. −4.62%). Em média simples, o Gemini rendeu +0.77% contra +15.62% do Buy-and-Hold nas 8 combinações. O período foi de alta para BTC, ETH e SOL, o que favorece a estratégia passiva long-only.
- A taxa de atingimento do preço-alvo ficou entre 0.0% e 41.2%; em `1h`, ficou entre 0.0% e 15.8%, indicando projeções de alvo pouco realistas nesse timeframe.
- A concordância com o proxy ficou entre 37.5% e 65.0%, sem relação simples com a acurácia.
- Critérios do TCC1: em `1h`, a acurácia atende (3/4), mas o retorno não (1/4); em `4h`, a acurácia atende (4/4), mas o retorno não (2/4). O conjunto dos dois critérios **não é atendido** em nenhum dos timeframes.
- O retorno é composto por decisão amostral e há sobreposição de horizontes; não há custos de transação, alavancagem ou posição vendida. A janela de 4 meses é curta e não deve ser generalizada nem comparada diretamente ao Experimento 2 de 18 meses.

---

## 7. Testes e cobertura

```powershell
python -m pytest -q

python -m pytest --cov=backend --cov=frontend --cov=webapp `
  --cov-report=term-missing --cov-report=html:coverage/html
```

Situação atual:

- **200 testes passed**;
- **cobertura total: 92%** (backend + frontend + webapp; 1.660 statements, 126 não cobertos);
- testes unitários sem internet e sem APIs reais;
- testes de integração usam mocks de rede e geram artefatos em diretórios temporários.

Mapeamento dos casos de teste do TCC:

| Caso | Cobertura |
|---|---|
| CT01 | Coleta OHLCV, colunas, quantidade e falha de conexão |
| CT02 | Indicadores obrigatórios, RSI/Bollinger e mínimo de 50 candles |
| CT03 | Geração de PNG, tamanho e conteúdo visual |
| CT04 | Validação do JSON do Gemini, campos, ação e fallback MANTER |
| CT05 | Dashboard HTML, abertura no navegador (mock) e conteúdo |
| CT06 | Falhas de API, JSON, timeout e fallback estruturado sem traceback |

Relatórios: `coverage/html/index.html` e `coverage/coverage_report.txt` (gerados localmente; não versionados).

---

## 8. Segurança

- `.env` foi removido do índice do Git e passou a ser ignorado.
- **Rotacione as chaves** da Binance e do Gemini que já tenham sido versionadas em commits anteriores; removê-las do índice não apaga o histórico.
- CSRF habilitado em todos os POSTs; tokens de sessão e cookies `HttpOnly`/`SameSite=Lax`.
- Senhas armazenadas com hash do Werkzeug (`pbkdf2:sha256`).
- Limite de 5 tentativas de login por IP em janela de 5 minutos (em memória; para produção distribuída, use Redis ou similar).
- Headers `X-Content-Type-Options`, `X-Frame-Options` e `Referrer-Policy`.
- Em produção, defina `WEB_ADMIN_PASS`, `FLASK_SECRET_KEY`, `CRIPTOIA_ENV=production` e `SESSION_COOKIE_SECURE=1`.
- Se `WEB_ADMIN_PASS`/`FLASK_SECRET_KEY` não estiverem definidas, a aplicação avisa e usa a senha padrão de desenvolvimento; nunca execute assim fora da máquina local.
- O servidor Flask de desenvolvimento escuta apenas `127.0.0.1`.

---

## 9. Limitações e trabalhos futuros

1. **Três níveis de evidência:** o Experimento 1 é proxy determinístico na grade 4×4; o Experimento 2 é Gemini real, mas amostral (63 pontos por par) e restrito ao timeframe diário; o Experimento 3 complementa com Gemini real em `1h`/`4h`, mas apenas numa janela exploratória de 4 meses. Não há replay completo dos ~68 mil candles (≈17 mil por par) nem cobertura Gemini de `1w`. A restrição do Experimento 2 ao `1d` decorre da quota do modelo gratuito (ver Seção 4.2.1); o Experimento 3 contorna parcialmente essa limitação com período curto.
2. **Horizonte e sobreposição:** o proxy usa o candle seguinte; o Gemini usa o prazo de `tempo_estimado`, mas as amostragens semanal (`1d`) e de 3 dias (`1h`/`4h`) podem sobrepor horizontes de 7–14 dias. O retorno dos Experimentos 2 e 3 é composto por decisão, não uma curva de capital sem sobreposição.
3. **Modelo:** o TCC1 previa Gemini 2.5 Flash; a execução registrada usou `gemini-3.5-flash-lite` por disponibilidade/custo. O modelo é configurável, mas os resultados não são automaticamente transferíveis entre versões.
4. **Custos de transação:** taxas, spread e slippage não estão incluídos; resultados positivos podem desaparecer com custos realistas.
5. **Long-only:** sinais de venda não geram posição vendida; isso limita a estratégia em mercados de baixa.
6. **LSTM sem tuning:** uma única configuração lightweight, treinada em CPU. Não houve busca de hiperparâmetros, validação cruzada temporal extensa ou ensemble.
7. **Amostra semanal pequena:** 78 candles no total e 28 válidos após o warm-up; resultados semanais são apenas indicativos.
8. **Dados da Binance:** o período depende da disponibilidade e da qualidade da API; a coleta marca resultados incompletos, mas não corrige falhas de dados da exchange.
9. **Interface:** rate limit em memória, adequado a uma instância acadêmica; produção exige armazenamento distribuído, HTTPS e revisão de segurança.

---

## 10. Reprodutibilidade e integridade acadêmica

- Seed do LSTM: `42`.
- Período oficial (Experimentos 1 e 2): `2025-01-01` a `2026-06-30` (UTC); Experimento 3: `2026-07-01` a `2026-10-31`.
- Experimento 2: `python principal.py --backtesting-gemini --grade --passo 7 --delay 2.0`; modelo registrado `gemini-3.5-flash-lite`; 63 pontos por par; zero fallbacks na versão final.
- Experimento 3: `python principal.py --backtesting-gemini --grade --intervalos 1h,4h --inicio 2026-07-01 --fim 2026-10-31 --passo 3 --delay 2.0 --prefixo-saida backtesting_gemini_intraday_tcc`; 304 inferências; zero fallbacks.
- Resultados completos e relatórios gerados a partir de dados reais da Binance estão em `CriptoIa/resultados/`.
- Nenhum número acadêmico fixo é exibido pela interface; a página de backtesting carrega apenas históricos executados e os JSONs consolidados reais (proxy, Gemini `1d` e complemento `1h`/`4h`).
- Consulte `CriptoIa/RELATORIO_TECNICO.md` para o registro detalhado de mudanças, comandos, resultados e pendências.

---

## 11. Conclusão

O projeto entrega o pipeline completo e três níveis de evidência: proxy determinístico na grade 4×4, Gemini real no `1d` (oficial) e Gemini real no `1h`/`4h` (exploratório). Os resultados são reportados sem ajuste: no `1d`, 3/4 de acurácia > 50% e 0/4 de retorno não negativo; no complemento `1h`/`4h`, 7/8 de acurácia > 50%, 3/8 de retorno não negativo e 1/8 de superação do Buy-and-Hold. O conjunto dos critérios do TCC1 **não foi atendido** em nenhum recorte, e a principal contribuição acadêmica é a avaliação transparente, reproduzível e com limitações explícitas.
