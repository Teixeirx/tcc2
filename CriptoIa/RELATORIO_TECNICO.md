# Relatório Técnico — CriptoIA (TCC II)

**Data do registro:** 06/11/2026  
**Escopo:** correção dos gaps identificados na auditoria, implementação dos módulos de backtesting/comparação, backtesting multimodal real do Gemini, complemento exploratório intraday, testes CT01–CT06, interface web segura e geração dos resultados experimentais reais.

---

## 1. Resumo executivo

Foram implementados e executados:

- backtesting real dos sinais técnicos em 4 pares × 4 timeframes (jan/2025–jun/2026) — Experimento 1, proxy determinístico;
- backtesting multimodal real do Gemini em 4 pares, timeframe diário, com 63 pontos semanais por par (252 inferências) — Experimento 2;
- complemento exploratório intraday do Gemini em `1h` e `4h`, com 304 inferências em 4 meses — Experimento 3;
- comparação com Buy-and-Hold e LSTM real em PyTorch, na mesma janela out-of-sample;
- mínimos de dados e validações para evitar resultados enganosos;
- fallback estruturado `MANTER` em falhas de coleta/API/JSON;
- interface Flask com login, CSRF, sessões protegidas, histórico e resultados de backtesting (proxy, Gemini `1d` e complemento `1h`/`4h`);
- persistência atômica de históricos;
- suíte de testes unitários e de integração com mocks, sem dependência de internet;
- relatórios e artefatos persistidos em `resultados/`.

**Resultado dos testes:** 200 passed, cobertura total de 92% (backend + frontend + webapp), incluindo o novo módulo de backtesting Gemini e a renderização do complemento intraday.

**Resultado experimental:** os critérios de sucesso do TCC **não foram atendidos em conjunto**. No Experimento 1 (proxy, 1d), 2 de 4 ativos tiveram acurácia > 50% e apenas 1 de 4 teve retorno não negativo. No Experimento 2 (Gemini real, 1d amostral), a acurácia de prazo superou 50% em 3 de 4 ativos, mas o retorno foi negativo em 4 de 4. No Experimento 3 (exploratório, 1h/4h, 4 meses), a acurácia superou 50% em 7 de 8 combinações, mas o retorno não negativo ocorreu em apenas 3 de 8 e o Buy-and-Hold foi superado em apenas 1 de 8. Sem ajuste ou fabricação.

---

## 2. Arquivos alterados/criados

### 2.1 Alterados

| Arquivo | Alteração |
|---|---|
| `CriptoIa/backend/analise_tecnica.py` | Adicionado `MINIMO_CANDLES_INDICADORES = 50`; `enriquecer_dataframe()` rejeita amostras menores com `ValueError`. |
| `CriptoIa/backend/coleta_dados.py` | Detecção de chaves placeholder/curtas para usar modo público da Binance. |
| `CriptoIa/backend/analisador_ia.py` | Validação de todos os campos obrigatórios, normalização de ação inválida, prompt com o par correto e fallback `MANTER` para chave ausente, JSON inválido, timeout e falhas de API. Modelo configurável por `MODELO_GEMINI`; correção do `import time` usado no retry de 503. |
| `CriptoIa/frontend/gerador_dashboard.py` | `gerar_dashboard()` aceita `caminho_saida`; Plotly configurável (`include_plotlyjs`). |
| `CriptoIa/principal.py` | Novos modos CLI `--backtesting`, `--backtesting-gemini`, `--comparacao`, `--multiplos`, `--grade`; opções de data, workers, amostragem e LSTM; fallback estruturado; envio do par ao Gemini. |
| `CriptoIa/requisitos.txt` | Versões fixadas; inclusão de `torch`, `numpy` e `Flask`. |
| `README.md` | Documentação reescrita para TCC II, com instalação, comandos, metodologia, resultados, testes, segurança e limitações. |

### 2.2 Criados

| Arquivo | Finalidade |
|---|---|
| `CriptoIa/backend/backtesting.py` | Coleta paginada, sinais, métricas anti look-ahead, execução paralela, critérios, JSON/CSV e relatório (Experimento 1). |
| `CriptoIa/backend/backtesting_gemini.py` | Backtesting multimodal real do Gemini: amostragem configurável, janela de 90 candles, parser de `tempo_estimado`/`preco_alvo`, cache atômico, Buy-and-Hold na mesma janela e relatórios dos Experimentos 2 e 3. |
| `CriptoIa/backend/comparacao.py` | Buy-and-Hold, features/target, LSTM PyTorch, walk-forward, janela comum, exportação e relatório. |
| `CriptoIa/backend/multiplos_ativos.py` | Análise paralela de pares/timeframes, isolamento de erros, relatório consolidado e CSV. |
| `CriptoIa/webapp/app.py` | Rotas Flask, CSRF, headers, rate limit de login, validações, histórico e grade TCC. |
| `CriptoIa/webapp/auth.py` | Credenciais por ambiente, hash de senha e `login_required`. |
| `CriptoIa/webapp/history_store.py` | Persistência JSON atômica, lock e limites de 100 análises/50 backtests. |
| `CriptoIa/webapp/templates/*` | Login, dashboard, análise, histórico e backtesting com resultados reais dos Experimentos 1, 2 e 3. |
| `CriptoIa/webapp/static/*` | Estilos da interface. |
| `CriptoIa/tests/**` | CT01–CT06, testes de módulos, interface, histórico, backtesting, comparação, backtesting Gemini e integração. |
| `CriptoIa/.env.example` | Placeholders de configuração e parâmetros do LSTM web. |
| `.gitignore` | Ignora `.env`, `.venv`, caches, cobertura, artefatos locais e históricos pessoais. |
| `CriptoIa/resultados/*` | Resultados reais, relatórios e resumos dos Experimentos 1, 2 e 3. |

### 2.3 Segurança de segredo

- `CriptoIa/.env` foi removido do índice com `git rm --cached` e passou a ser ignorado.
- O arquivo local permanece para execução, mas **as chaves já versionadas devem ser rotacionadas**, pois remover do índice não apaga o histórico Git.
- Nenhum segredo real foi incluído em `.env.example`, README ou relatórios.

---

## 3. Metodologia implementada

### 3.1 Coleta e período

- Pares: `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `DOGE/USDT`.
- Timeframes: `1h`, `4h`, `1d`, `1w`.
- Período: `2025-01-01 00:00:00` a `2026-06-30 23:59:59` UTC.
- Paginação de 1.000 candles por requisição, deduplicação, filtro de bordas e atributos de auditoria (`coleta_incompleta`, `erro_coleta`).

Volumes coletados por par:

| Timeframe | Candles |
|---|---:|
| 1h | 13.104 |
| 4h | 3.276 |
| 1d | 546 |
| 1w | 78 |

Cobertura reportada: 100% do período estimado nas 16 combinações.

### 3.2 Estratégia proxy CriptoIA

- COMPRAR: `MM20 > MM50` e `RSI14 < 70` e `MACD > Sinal_MACD`.
- VENDER: `MM20 < MM50` e `RSI14 > 30` e `MACD < Sinal_MACD`.
- MANTER: demais casos.
- Long-only; sem alavancagem, taxas, spread ou slippage.
- Sinal em `t` aplicado ao retorno `close[t+1]/close[t]-1`.
- Retorno futuro usado apenas para medir acerto; nunca para gerar o sinal.
- Acurácia direcional calculada somente sobre sinais ±1.

### 3.3 LSTM

- PyTorch 2.14.1 CPU, rede LSTM + dropout + camada linear.
- 5 features: RSI normalizado, diferença MACD/Sinal, distância relativa MM20/MM50, posição nas Bandas de Bollinger e retorno anterior.
- Alvo: direção do próximo candle.
- `StandardScaler` ajustado somente no treino.
- Walk-forward com re-treino periódico usando apenas dados até `t`.
- Seed 42; 20 épocas; `seq_len=24`; `hidden_size=32`; `passo_retreino=1000`.
- No timeframe semanal, `seq_len` foi reduzido automaticamente para 7 e registrado no JSON.

### 3.4 Janela comum

A janela comum é o período out-of-sample do LSTM (últimos 40% após o treino inicial). As três estratégias são recalculadas nessa mesma janela, eliminando a comparação injusta entre períodos diferentes.

### 3.5 Experimento 2 — backtesting multimodal real do Gemini

- Pares: os mesmos 4 do TCC; timeframe `1d`.
- Amostragem sistemática a cada 7 dias, do 90º candle até 15 candles antes do fim da série; 63 pontos por par (252 inferências).
- Em cada ponto `t`, o gráfico PNG contém **somente** os 90 candles até `t`; não há dados futuros na imagem nem nos indicadores.
- Modelo registrado: `gemini-3.5-flash-lite` (configurável por `MODELO_GEMINI`); o TCC1 previa `gemini-2.5-flash`.
- A resposta é validada e, em falha persistente, vira `MANTER` com `fallback=True`; o código faz até 3 tentativas.
- `tempo_estimado` é convertido em janela de dias (ex.: “7 a 14 dias” → 7–14; “2 a 4 semanas” → 14–28).
- `preco_alvo` é normalizado (formatos BR/US e percentual) e comparado com máximas/mínimas futuras.
- Métricas: acurácia em `t+1`, acurácia no prazo estimado, taxa de alvo, retorno composto por decisão, Buy-and-Hold na mesma janela e concordância ponto a ponto com o proxy.
- Cache atômico em `resultados/gemini_backtest_cache.json`, com retomada sem chamadas duplicadas.
- Limitações do Experimento 2: amostra semanal, apenas `1d`, retorno composto com possível sobreposição de horizontes, long-only e sem custos. O Experimento 3 complementa com `1h`/`4h` em janela curta.

**Justificativa da restrição ao `1d`:** o `--passo` é expresso em dias e convertido para candles (`1d` = 7 candles; `4h` = 42; `1h` = 168). Uma amostragem semanal nos quatro pares exigiria aproximadamente 312 chamadas em `1h`, 304 em `4h` e 252 em `1d` (~868 no total). O `1w` tem apenas 78 candles no período, abaixo do mínimo de 110 necessário para a janela de 90 candles passados + 15 futuros. Como o trabalho utiliza o **modelo gratuito do Gemini**, com limite de requisições/tokens, a inferência real do Experimento 2 foi deliberadamente restrita ao `1d`; a grade 4×4 completa é coberta pelo proxy determinístico, que não consome API. O Experimento 3 complementa essa lacuna com `1h` e `4h` numa janela curta de 4 meses, sem alterar o recorte oficial.

**Importante:** a amostragem menor reduz o número de **pontos de decisão avaliados**, não a janela de contexto de cada inferência. Em todas as chamadas, o Gemini continua recebendo exatamente os mesmos 90 candles anteriores ao ponto `t`. O trade-off é estatístico (menos observações, maior incerteza), não de quantidade de candles por análise.

### 3.6 Experimento 3 — complemento exploratório intraday (1h/4h)

- **Objetivo:** complementar o Experimento 2 com inferência real do Gemini em `1h` e `4h`, sem alterar o resultado oficial em `1d`.
- **Período:** `2026-07-01` a `2026-10-31` (4 meses), fixo para reprodutibilidade.
- **Pares:** os mesmos 4 do TCC.
- **Timeframes:** `1h` e `4h`.
- **Amostragem:** a cada 3 dias; cada chamada usa 90 candles.
- **Aquecimento:** a primeira decisão ocorre após 90 candles (≈04/07 em `1h` e ≈16/07 em `4h`); o período efetivo de decisões é menor que a janela de coleta.
- **Volume:** 40 pontos/par em `1h`, 36 em `4h`, 304 chamadas no total.
- **Saídas separadas:** `resultados/backtesting_gemini_intraday_tcc.json`, `.csv` e `_resumo.txt`; cache compartilhado para retomada.
- **Comparação:** Buy-and-Hold e proxy na mesma janela curta.
- **Status:** executado em 06/11/2026, com 304 inferências e zero fallbacks. O resultado oficial continua sendo o Experimento 2 em `1d` por 18 meses.

Comando:

```powershell
python principal.py --backtesting-gemini --grade `
  --intervalos 1h,4h `
  --inicio 2026-07-01 --fim 2026-10-31 `
  --passo 3 --delay 2.0 `
  --prefixo-saida backtesting_gemini_intraday_tcc
```

---

## 4. Resultados — backtesting

### 4.1 Grade completa (proxy CriptoIA)

| Par | TF | Acurácia | Retorno | Max. DD | Operações |
|---|---|---:|---:|---:|---:|
| BTC/USDT | 1h | 48.5% | −1.67% | −15.30% | 1.890 |
| BTC/USDT | 4h | 49.0% | −4.72% | −16.54% | 464 |
| BTC/USDT | 1d | 56.5% | +16.20% | −14.08% | 71 |
| BTC/USDT | 1w | 33.3% | 0.00% | 0.00% | 5 |
| ETH/USDT | 1h | 47.6% | −23.26% | −26.67% | 1.689 |
| ETH/USDT | 4h | 47.9% | +5.96% | −17.69% | 422 |
| ETH/USDT | 1d | 47.4% | −15.83% | −28.70% | 60 |
| ETH/USDT | 1w | 50.0% | 0.00% | 0.00% | 3 |
| SOL/USDT | 1h | 48.9% | −11.79% | −38.02% | 1.878 |
| SOL/USDT | 4h | 47.6% | −1.10% | −28.14% | 456 |
| SOL/USDT | 1d | 49.6% | −39.06% | −39.85% | 65 |
| SOL/USDT | 1w | N/A | 0.00% | 0.00% | 0 |
| DOGE/USDT | 1h | 48.8% | +1.78% | −34.10% | 1.828 |
| DOGE/USDT | 4h | 46.7% | +11.55% | −30.48% | 466 |
| DOGE/USDT | 1d | 55.2% | −5.08% | −41.04% | 80 |
| DOGE/USDT | 1w | 33.3% | 0.00% | 0.00% | 2 |

Observações:

- N/A indica ausência de sinais direcionais válidos.
- Em 1w, a estratégia long-only não encontrou sinais de compra para BTC/ETH/SOL/DOGE; os sinais de venda não abrem posição vendida, resultando em retorno 0%.
- A amostra semanal tem apenas 28 candles válidos após o warm-up.

### 4.2 Critérios de sucesso

O TCC1 define os critérios por ativo, mas não especifica o timeframe do resumo. Para não inventar esse parâmetro, o código preserva os 16 resultados brutos e adota o diário apenas como recorte operacional do resumo (`recorte_principal: "1d"`), explicitando essa escolha em `nota_timeframe`.

| Critério | Resultado |
|---|---|
| Acurácia > 50% em ≥ 3 dos 4 ativos (1d) | **Não atendido** — 2/4 (BTC 56.5%, DOGE 55.2%) |
| Retorno não negativo | **Não atendido** — 1/4 (BTC +16.20%) |

A avaliação é da estratégia proxy de indicadores, não das respostas do Gemini.

### 4.3 Experimento 2 — Gemini multimodal real (1d, amostral)

Modelo `gemini-3.5-flash-lite`; 63 pontos por par; zero fallbacks na versão final. A janela efetiva começa em 2025-04-01 (após 90 candles de warm-up) e termina no último horizonte de cada par (2026-06-24 ou 2026-06-30).

| Par | Amostras | Comp./Vend./Mant. | Acur. t+1 | Acur. prazo | Alvo | Retorno Gemini | Buy-and-Hold (mesma janela) | Concord. proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC/USDT | 63 | 38 / 7 / 18 | 48.9% | 62.2% | 39.5% | −30.97% | −31.16% | 42.9% |
| ETH/USDT | 63 | 38 / 10 / 15 | 54.2% | 54.2% | 47.4% | −54.95% | −17.48% | 33.3% |
| SOL/USDT | 63 | 39 / 9 / 15 | 54.2% | 60.4% | 48.7% | −64.07% | −46.23% | 31.7% |
| DOGE/USDT | 63 | 43 / 6 / 14 | 51.0% | 49.0% | 32.6% | −57.46% | −56.25% | 25.4% |

Critérios do TCC1 no Experimento 2:

| Critério | Resultado |
|---|---|
| Acurácia de prazo > 50% em ≥ 3 dos 4 ativos | **Atendido** — 3/4 (BTC 62.2%, ETH 54.2%, SOL 60.4%; DOGE 49.0%) |
| Retorno acumulado não negativo | **Não atendido** — 0/4 (todos negativos) |
| Conjunto dos dois critérios | **Não atendido** |

Leitura:

- O Gemini real teve acurácia direcional de prazo acima de 50% em 3 dos 4 ativos, mas essa acurácia não se converteu em retorno positivo no período de forte queda.
- Contra o Buy-and-Hold na mesma janela, o Gemini superou apenas BTC (+0.19 p.p.); ficou abaixo em ETH (−37.47 p.p.), SOL (−17.84 p.p.) e DOGE (−1.21 p.p.).
- A concordância com o proxy ficou entre 25.4% e 42.9%, mostrando que regras fixas de MM/RSI/MACD não reproduzem as decisões multimodais.
- O retorno é composto por decisão amostral; com passo de 7 dias e horizonte de 7–14 dias (ou mais), há sobreposição. Não é uma curva de capital sem sobreposição e não inclui custos.
- O Experimento 2 cobre apenas `1d`; a grade 4×4 completa continua sendo avaliada pelo proxy (Experimento 1), e o Experimento 3 complementa com `1h`/`4h` em janela curta.

### 4.4 Experimento 3 — Gemini multimodal real (1h/4h exploratório)

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

Critérios do TCC1 no Experimento 3:

| Timeframe | Acurácia de prazo > 50% | Retorno não negativo | Conjunto |
|---|---|---|---|
| `1h` | **Atendido** — 3/4 (BTC, SOL, DOGE) | **Não atendido** — 1/4 (SOL +4.55%) | **Não atendido** |
| `4h` | **Atendido** — 4/4 | **Não atendido** — 2/4 (BTC +5.35%, ETH +11.22%) | **Não atendido** |

Leitura:

- A acurácia de prazo superou 50% em **7 de 8** combinações: `1h` 3/4 e `4h` 4/4. O ponto fora foi ETH `1h` (48.0%).
- O retorno não foi negativo em **3 de 8** combinações: SOL `1h` (+4.55%), BTC `4h` (+5.35%) e ETH `4h` (+11.22%).
- O Buy-and-Hold foi superado em apenas **1 de 8** combinações: DOGE `1h` (−0.81% vs. −4.62%). Em média simples, o Gemini rendeu +0.77% contra +15.62% do Buy-and-Hold. O período foi de alta para BTC, ETH e SOL, o que favorece a estratégia passiva long-only.
- A taxa de atingimento do preço-alvo ficou entre 0.0% e 41.2%; em `1h`, entre 0.0% e 15.8%, indicando projeções de alvo pouco realistas nesse timeframe.
- A concordância com o proxy ficou entre 37.5% e 65.0%, sem relação simples com a acurácia.
- O retorno é composto por decisão amostral e há sobreposição de horizontes; não há custos, alavancagem ou posição vendida. A janela de 4 meses é curta e não deve ser generalizada nem comparada diretamente ao Experimento 2 de 18 meses.

---

## 5. Resultados — comparação de estratégias

### 5.1 Janela comum — timeframe diário

| Par | CriptoIA retorno / acurácia | Buy-and-Hold | LSTM retorno / acurácia | Baseline LSTM |
|---|---:|---:|---:|---:|
| BTC/USDT | +1.28% / 54.0% | −35.03% | −25.37% / 52.3% | 53.8% |
| ETH/USDT | +2.25% / 46.5% | −49.53% | −48.59% / 46.7% | 53.8% |
| SOL/USDT | −14.05% / 43.3% | −44.67% | −34.11% / 47.2% | 54.8% |
| DOGE/USDT | +15.26% / 56.4% | −48.16% | −5.27% / 55.8% | 59.3% |

### 5.2 Grade completa — janela comum

| Par | TF | CriptoIA ret. | CriptoIA acur. | B&H ret. | LSTM ret. | LSTM acur. | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| BTC/USDT | 1h | −3.37% | 48.2% | −32.79% | −21.49% | 51.8% | 50.4% |
| BTC/USDT | 4h | −0.32% | 48.8% | −35.95% | −25.61% | 50.4% | 50.0% |
| BTC/USDT | 1d | +1.28% | 54.0% | −35.03% | −25.37% | 52.3% | 53.8% |
| BTC/USDT | 1w | 0.00% | 0.0% | −10.02% | −13.33% | 33.3% | 58.3% |
| ETH/USDT | 1h | −12.73% | 48.8% | −45.64% | −48.61% | 51.2% | 50.4% |
| ETH/USDT | 4h | −1.96% | 47.7% | −48.15% | −23.99% | 51.3% | 50.3% |
| ETH/USDT | 1d | +2.25% | 46.5% | −49.53% | −48.59% | 46.7% | 53.8% |
| ETH/USDT | 1w | 0.00% | 66.7% | −18.52% | −18.52% | 41.7% | 58.3% |
| SOL/USDT | 1h | +2.18% | 49.1% | −45.95% | −55.80% | 50.9% | 50.8% |
| SOL/USDT | 4h | +1.82% | 47.5% | −48.39% | −24.27% | 48.9% | 51.0% |
| SOL/USDT | 1d | −14.05% | 43.3% | −44.67% | −34.11% | 47.2% | 54.8% |
| SOL/USDT | 1w | 0.00% | N/A | +0.07% | 0.00% | 41.7% | 58.3% |
| DOGE/USDT | 1h | −13.73% | 47.2% | −51.50% | −54.66% | 51.1% | 51.2% |
| DOGE/USDT | 4h | −7.15% | 44.3% | −53.18% | −15.73% | 51.2% | 52.6% |
| DOGE/USDT | 1d | +15.26% | 56.4% | −48.16% | −5.27% | 55.8% | 59.3% |
| DOGE/USDT | 1w | 0.00% | N/A | −14.28% | +16.19% | 75.0% | 50.0% |

### 5.3 Leitura dos resultados

Os resultados desta seção são do **Experimento 1 (proxy determinístico 4×4)**, não do Gemini multimodal.

- CriptoIA superou Buy-and-Hold em retorno em **15 de 16** combinações na janela comum.
- LSTM ficou acima de 50% de acurácia em **10 de 16** combinações.
- LSTM superou o baseline de classe majoritária em apenas **6 de 16** combinações; portanto, parte da acurácia é explicada pelo desbalanceamento das classes.
- CriptoIA teve acurácia > 50% em **3 de 16** combinações.
- Não houve erro de execução do LSTM nas 16 combinações.
- Resultados semanais têm amostra pequena e não devem ser generalizados.
- Não houve teste de significância estatística; os números são descritivos.

### 5.4 Artefatos

- `resultados/backtesting_tcc.json` — resultados completos e critérios;
- `resultados/backtesting_tcc.csv` — tabela para planilha;
- `resultados/backtesting_relatorio.txt` — relatório textual completo;
- `resultados/backtesting_resumo.txt` — tabela/resumo em texto;
- `resultados/comparacao_tcc.json` — resultados completos;
- `resultados/comparacao_tcc.csv` — tabela para planilha;
- `resultados/comparacao_relatorio.txt` — relatório textual completo;
- `resultados/comparacao_resumo.txt` — tabela/resumo em texto;
- `resultados/backtesting_gemini_tcc.json` — resultados completos do Experimento 2 (oficial, `1d`);
- `resultados/backtesting_gemini_tcc.csv` — tabela do Experimento 2;
- `resultados/backtesting_gemini_resumo.txt` — resumo em texto do Experimento 2;
- `resultados/backtesting_gemini_intraday_tcc.json` — resultados completos do Experimento 3 (`1h`/`4h`);
- `resultados/backtesting_gemini_intraday_tcc.csv` — tabela do Experimento 3;
- `resultados/backtesting_gemini_intraday_tcc_resumo.txt` — resumo em texto do Experimento 3;
- `resultados/gemini_backtest_cache.json` — cache compartilhado das 556 inferências (252 do Experimento 2 + 304 do Experimento 3).

---

## 6. Testes e cobertura

### 6.1 Comandos executados

```powershell
cd CriptoIa

python -m pytest -q

python -m pytest --cov=backend --cov=frontend --cov=webapp `
  --cov-report=term-missing --cov-report=html:coverage/html `
  --cov-report=xml:coverage/coverage.xml
```

### 6.2 Resultado

- **200 testes passed**.
- **Cobertura total: 92%** (1.660 statements, 126 não cobertos).

Cobertura por módulo:

| Módulo | Cobertura |
|---|---:|
| `backend/analise_tecnica.py` | 100% |
| `backend/gerador_grafico.py` | 100% |
| `backend/multiplos_ativos.py` | 100% |
| `frontend/gerador_dashboard.py` | 100% |
| `backend/backtesting.py` | 93% |
| `backend/backtesting_gemini.py` | 89% |
| `backend/comparacao.py` | 94% |
| `backend/analisador_ia.py` | 88% |
| `webapp/history_store.py` | 96% |
| `webapp/app.py` | 91% |
| `backend/coleta_dados.py` | 80% |
| `webapp/auth.py` | 75% |
| **Total** | **92%** |

Relatórios gerados localmente: `coverage/coverage_report.txt`, `coverage/coverage.xml` e `coverage/html/index.html` (ignorados pelo Git).

### 6.3 CT01–CT06

| Caso | Testes |
|---|---|
| CT01 — coleta | quantidade/colunas, ausência de nulos, falha de conexão descritiva, modo público. |
| CT02 — indicadores | colunas obrigatórias, RSI 0–100, Bollinger superior > inferior, rejeição de < 50 candles. |
| CT03 — gráfico | PNG gerado, tamanho, conteúdo/título e eixo. |
| CT04 — Gemini | campos obrigatórios, ação válida, JSON inválido, campo ausente, ação inválida, chave ausente e fallback `MANTER`. |
| CT05 — dashboard | HTML gerado, conteúdo, Plotly, indicadores e abertura no navegador com mock. |
| CT06 — falhas | falha de API, timeout, JSON inválido, coleta e fallback estruturado sem traceback; integração CLI e web. |

Além dos CT01–CT06, a suíte cobre o backtesting Gemini: parsers de `tempo_estimado` e `preco_alvo`, cache atômico e corrompido, sinal proxy, dados insuficientes, Buy-and-Hold na mesma janela, execução da grade e renderização dos experimentos oficial (`1d`) e exploratório (`1h`/`4h`) na interface web.

Os testes unitários usam `unittest.mock`/`monkeypatch`; não há chamadas reais à Binance ou ao Gemini. Os testes de integração também mockam a camada de rede e usam diretórios temporários.

---

## 7. Interface web e segurança

### 7.1 Funcionalidades entregues

- login com credenciais por ambiente e senha com hash;
- CSRF em todos os POSTs;
- cookies `HttpOnly`, `SameSite=Lax` e opção `Secure`;
- rate limit de 5 tentativas por IP/5 minutos (em memória);
- sessão renovada no login (`session.clear()` antes de autenticar);
- rotas protegidas por `login_required`;
- validação de par/timeframe e limite de quantidade (50–1000);
- análise com fallback `MANTER` persistido no histórico;
- histórico de análises e backtests;
- página de backtesting carregando a grade real de `resultados/comparacao_tcc.json`, o experimento Gemini oficial de `resultados/backtesting_gemini_tcc.json` e o complemento exploratório de `resultados/backtesting_gemini_intraday_tcc.json`;
- resumo dinâmico dos critérios do TCC para o experimento oficial e contadores descritivos para o complemento `1h`/`4h` (sem números fixos);
- ausência total de tabelas acadêmicas fixas/fabricadas.

### 7.2 Verificações

- rotas `/`, `/analise`, `/historico`, `/backtesting` redirecionam para `/login` sem sessão;
- login válido/inválido, logout, CSRF ausente/válido, headers e rate limit cobertos por testes;
- renderização da grade TCC e do histórico coberta por testes;
- execução de backtesting via web usa configuração enxuta do LSTM e persiste o resultado.

Smoke test autenticado executado com o cliente de teste do Flask e CSRF ativo:

```text
GET /login: 200
POST /login: 302
GET /: 200
GET /analise: 200
GET /historico: 200
GET /backtesting: 200
POST /analise sem CSRF: 400
GET /logout: 302
GET /backtesting após logout: 302
```

O smoke test usou credenciais temporárias via variáveis de ambiente. O `.env` local não define `WEB_ADMIN_PASS` nem `FLASK_SECRET_KEY`; nesse caso a aplicação avisa e usa a senha padrão de desenvolvimento. Essas duas variáveis devem ser definidas antes de qualquer uso fora do ambiente local.

### 7.3 Pendências de produção

- usar servidor WSGI (Waitress/Gunicorn), HTTPS e proxy reverso;
- mover rate limit para armazenamento distribuído;
- definir `CRIPTOIA_ENV=production`, `WEB_ADMIN_PASS`, `FLASK_SECRET_KEY` e `SESSION_COOKIE_SECURE=1`;
- rotacionar chaves já expostas no histórico Git.

---

## 8. Limitações metodológicas

1. **Três níveis de evidência:** o Experimento 1 é proxy determinístico na grade 4×4; o Experimento 2 é Gemini real, mas amostral (63 pontos/par) e restrito ao `1d`; o Experimento 3 complementa com Gemini real em `1h`/`4h`, mas em janela exploratória de 4 meses. Não há replay completo dos ~68 mil candles (≈17 mil por par) nem cobertura Gemini de `1w`.
2. **Horizonte e sobreposição:** o proxy usa o candle seguinte; o Gemini usa o prazo de `tempo_estimado`, mas as amostragens semanal (`1d`) e de 3 dias (`1h`/`4h`) podem sobrepor horizontes de 7–14 dias. O retorno dos Experimentos 2 e 3 é composto por decisão, não uma curva de capital sem sobreposição.
3. **Modelo:** o TCC1 previa Gemini 2.5 Flash; a execução usou `gemini-3.5-flash-lite` por disponibilidade/custo. O modelo é configurável, mas os resultados não são automaticamente transferíveis entre versões.
4. **Custos:** não há taxas, spread, slippage ou custo de funding; o retorno simulado é bruto.
5. **Long-only:** sinais de venda não abrem posição vendida; em bear market a estratégia fica em caixa.
6. **LSTM:** uma configuração lightweight, sem tuning extensivo, validação cruzada temporal ampla ou ensemble; CPU only.
7. **Amostra semanal:** 78 candles totais e 28 válidos; resultados apenas indicativos.
8. **Dados:** dependem da API pública da Binance; falhas parciais são sinalizadas, mas não corrigidas.
9. **Estatística:** não foram aplicados testes de significância, intervalos de confiança ou correção para múltiplas comparações.

---

## 9. Pendências e próximos passos

- [ ] Rotacionar chaves da Binance/Gemini expostas em commits anteriores.
- [x] Executar replay amostral real do Gemini (Experimento 2, 1d, 252 inferências, zero fallbacks).
- [x] Executar o Experimento 3 exploratório (`1h`/`4h`, jul–out/2026, passo de 3 dias, 304 inferências, zero fallbacks); o `1w` permanece fora do replay real por falta de histórico suficiente.
- [ ] Implementar curva de capital sem sobreposição de horizontes para os Experimentos 2 e 3.
- [ ] Adicionar custos de transação e permitir posições vendidas/short.
- [ ] Executar múltiplas seeds/configurações de LSTM e reportar média/desvio.
- [ ] Ampliar a amostra semanal (período maior) ou tratar 1w como análise exploratória.
- [ ] Adicionar testes estatísticos e intervalos de confiança.
- [ ] Configurar CI (GitHub Actions) para rodar pytest/coverage.
- [ ] Empacotar a aplicação web para produção (Waitress/Gunicorn + HTTPS).
- [ ] Revisar a política de retenção dos históricos JSON.

---

## 10. Comandos de reprodução

```powershell
cd CriptoIa

# Dependências
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requisitos.txt

# Testes
python -m pytest -q
python -m pytest --cov=backend --cov=frontend --cov=webapp `
  --cov-report=term-missing --cov-report=html:coverage/html

# Experimentos
python principal.py --backtesting --grade --max-workers 2
python principal.py --comparacao --grade
python principal.py --backtesting-gemini --grade --passo 7 --delay 2.0
python principal.py --backtesting-gemini --grade --intervalos 1h,4h `
  --inicio 2026-07-01 --fim 2026-10-31 --passo 3 --delay 2.0 `
  --prefixo-saida backtesting_gemini_intraday_tcc

# Interface
python -m webapp.app
```

---

## 11. Conclusão

A entrega cobre os requisitos funcionais e não funcionais priorizados: backtesting real, comparação com duas baselines, LSTM verdadeiro, testes CT01–CT06, análise paralela, interface protegida, histórico persistido e documentação. Foram executados três experimentos complementares: (i) proxy determinístico na grade 4×4, com 2/4 de acurácia > 50% e 1/4 de retorno não negativo no `1d`; (ii) Gemini multimodal real no `1d` amostral, com acurácia de prazo > 50% em 3/4, mas retorno negativo em 4/4; e (iii) complemento exploratório em `1h`/`4h`, com acurácia > 50% em 7/8 combinações, retorno não negativo em 3/8 e superação do Buy-and-Hold em apenas 1/8. O conjunto dos critérios do TCC **não foi atingido** em nenhum dos recortes, e o Gemini não superou o Buy-and-Hold na maioria dos ativos na mesma janela. A contribuição acadêmica está em reportar isso de forma transparente, com metodologia reproduzível e limitações explícitas, em vez de apresentar números fabricados.
