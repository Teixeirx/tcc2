"""
comparacao.py — Módulo de comparação de estratégias do CriptoIA.

Compara três abordagens de negociação:
    1. CriptoIA           : Estratégia determinística baseada em indicadores
                            técnicos (proxy da lógica técnica do sistema)
    2. Buy-and-Hold       : Manter o ativo durante todo o período
    3. LSTM Supervisionado: Rede LSTM real (PyTorch) para previsão da
                            direção do próximo candle, com walk-forward

Metodologia:
    - Mesmo período e critérios para as três estratégias na janela comum
      de avaliação (últimos 40% dos dados, após o treino inicial do LSTM).
    - Walk-forward sem look-ahead: o modelo é re-treinado periodicamente
      usando apenas dados disponíveis até o candle de decisão.
    - Features: RSI normalizado, diferença MACD/Sinal, diferença MM20/MM50,
      posição nas Bandas de Bollinger e retorno do candle anterior.
    - Alvo: direção do retorno do PRÓXIMO candle (close[t+1]/close[t]-1).
    - O retorno simulado usa exatamente esse retorno futuro, jamais o
      retorno do candle já conhecido na decisão.
    - Métricas: retorno acumulado, acurácia direcional, máximo drawdown,
      número de operações e baseline de classe majoritária.

Limitações documentadas:
    - Sem custos de transação em nenhuma das estratégias.
    - Long-only para CriptoIA e LSTM.
    - O LSTM é treinado do zero, em CPU, com configuração lightweight;
      não houve tuning extensivo de hiperparâmetros.
    - A estratégia CriptoIA é um proxy determinístico e não reproduz as
      respostas multimodais do Gemini.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Optional

# Garante saída compatível com Windows
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from backend.backtesting import (
    buscar_dados_historicos_paginado,
    calcular_metricas,
    DATA_INICIO_TCC,
    DATA_FIM_TCC,
    PARES_TCC,
    TIMEFRAMES_TCC,
    _timestamp_compativel,
)
from backend.analise_tecnica import enriquecer_dataframe

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── PyTorch (LSTM) ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_DISPONIVEL = True
except Exception:  # pragma: no cover - depende do ambiente
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_DISPONIVEL = False


if TORCH_DISPONIVEL:
    class ModeloLSTM(nn.Module):
        """Rede LSTM simples para classificação binária de direção."""

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 32,
            num_layers: int = 1,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            saida, _ = self.lstm(x)
            ultimo = saida[:, -1, :]
            return self.fc(self.dropout(ultimo)).squeeze(-1)
else:  # pragma: no cover
    class ModeloLSTM:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch não está instalado. Instale com: "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )


# ── Estratégia Buy-and-Hold ────────────────────────────────────────────────────

def calcular_buy_and_hold(
    df: pd.DataFrame,
    data_inicio: Optional[datetime] = None,
) -> dict:
    """
    Calcula o retorno da estratégia Buy-and-Hold.

    Parâmetros:
        df          : DataFrame com coluna 'Fechamento' (dados históricos)
        data_inicio : Início opcional da janela de avaliação

    Retorna:
        Dicionário com métricas da estratégia Buy-and-Hold
    """
    if df is None or df.empty or len(df) < 2:
        return {"erro": "Dados insuficientes para Buy-and-Hold"}

    df_avaliacao = df
    if data_inicio is not None:
        data_inicio = _timestamp_compativel(df.index, data_inicio)
        df_avaliacao = df[df.index >= data_inicio]

    if len(df_avaliacao) < 2:
        return {"erro": "Dados insuficientes na janela de avaliação"}

    preco_inicial = float(df_avaliacao["Fechamento"].iloc[0])
    preco_final = float(df_avaliacao["Fechamento"].iloc[-1])
    retorno_total = (preco_final - preco_inicial) / preco_inicial

    retornos = df_avaliacao["Fechamento"].pct_change().dropna()

    curva = (1 + retornos).cumprod()
    pico = curva.cummax()
    drawdown = (curva - pico) / pico
    max_drawdown = float(drawdown.min())

    return {
        "estrategia": "Buy-and-Hold",
        "preco_inicial": round(preco_inicial, 4),
        "preco_final": round(preco_final, 4),
        "retorno_acumulado": round(retorno_total, 4),
        "retorno_total_pct": round(retorno_total * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "n_candles": int(len(df_avaliacao)),
        "data_inicio": df_avaliacao.index[0].isoformat(),
        "data_fim": df_avaliacao.index[-1].isoformat(),
    }


# ── Features e alvo do modelo supervisionado ──────────────────────────────────

def _preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai as features para o modelo supervisionado.

    Features:
        - RSI_norm     : RSI14 / 100
        - MACD_diff    : MACD - Sinal_MACD
        - MM_diff_pct  : (MM20 - MM50) / Fechamento
        - BB_posicao   : posição do fechamento entre as bandas (0 a 1)
        - Retorno_1    : retorno percentual do candle anterior

    As features usam somente informações disponíveis no fechamento do
    candle corrente. A normalização é feita depois, apenas com dados
    de treino, para evitar vazamento.
    """
    feat = pd.DataFrame(index=df.index)

    feat["RSI_norm"] = df["RSI14"] / 100.0
    feat["MACD_diff"] = df["MACD"] - df["Sinal_MACD"]
    feat["MM_diff_pct"] = (df["MM20"] - df["MM50"]) / df["Fechamento"]

    bb_range = df["Bollinger_Superior"] - df["Bollinger_Inferior"]
    feat["BB_posicao"] = np.where(
        bb_range > 0,
        (df["Fechamento"] - df["Bollinger_Inferior"]) / bb_range,
        0.5,
    )

    feat["Retorno_1"] = df["Fechamento"].pct_change()

    return feat.dropna()


def _calcular_target(df: pd.DataFrame) -> pd.Series:
    """
    Calcula o alvo: 1 se o PRÓXIMO candle sobe, 0 se cai.

    O último valor é NaN (não há próximo candle) e é removido pelo
    alinhamento com as features.
    """
    retorno_futuro = df["Fechamento"].pct_change().shift(-1)
    target = pd.Series(np.nan, index=df.index, dtype=float)
    validos = retorno_futuro.notna()
    target.loc[validos] = (retorno_futuro.loc[validos] > 0).astype(int)
    return target


def _criar_sequencias(
    features: pd.DataFrame,
    target: pd.Series,
    seq_len: int,
):
    """
    Cria sequências [t-seq_len+1 ... t] -> target[t].

    Retorna (X, y, indices) sem incluir alvos NaN.
    """
    X, y, indices = [], [], []
    valores = features.values.astype(np.float32)

    for i in range(seq_len - 1, len(features)):
        alvo = target.iloc[i]
        if pd.isna(alvo):
            continue
        X.append(valores[i - seq_len + 1:i + 1])
        y.append(int(alvo))
        indices.append(features.index[i])

    if not X:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.array([], dtype=np.float32), []

    return np.asarray(X), np.asarray(y, dtype=np.float32), indices


def _definir_seed(seed: int = 42) -> None:
    """Define seeds para reprodutibilidade do LSTM."""
    np.random.seed(seed)
    if TORCH_DISPONIVEL:
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _treinar_lstm(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    n_features: int,
    seq_len: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1e-3,
) -> "ModeloLSTM":
    """Treina uma instância do LSTM e a devolve em modo eval."""
    if not TORCH_DISPONIVEL:
        raise RuntimeError("PyTorch não está instalado.")

    modelo = ModeloLSTM(n_features=n_features, hidden_size=hidden_size)
    criterio = nn.BCEWithLogitsLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.from_numpy(X_treino),
        torch.from_numpy(y_treino),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    modelo.train()
    for _ in range(max(1, epochs)):
        for lote_x, lote_y in loader:
            otimizador.zero_grad()
            logits = modelo(lote_x)
            perda = criterio(logits, lote_y)
            perda.backward()
            otimizador.step()

    modelo.eval()
    return modelo


def executar_modelo_supervisionado(
    df_enriquecido: pd.DataFrame,
    janela_treino_pct: float = 0.6,
    seq_len: int = 24,
    epochs: int = 20,
    hidden_size: int = 32,
    batch_size: int = 128,
    passo_retreino: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Executa o modelo LSTM com walk-forward.

    Protocolo:
        - Treina com os primeiros `janela_treino_pct` dos dados.
        - No período de teste, re-treina a cada `passo_retreino` candles,
          usando somente dados até o candle de decisão.
        - Prevê a direção do próximo candle e aplica o retorno futuro
          correspondente (close[t+1]/close[t]-1).

    Retorna:
        Dicionário com métricas do LSTM e metadados do experimento.
    """
    resultado_base = {
        "estrategia": "LSTM_Supervisionado",
        "arquitetura": "LSTM (PyTorch)",
        "seq_len": seq_len,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "janela_treino_pct": janela_treino_pct,
        "passo_retreino": passo_retreino,
        "nota_metodologica": (
            "LSTM real treinada em CPU, walk-forward com re-treino periódico. "
            "Avaliação no período out-of-sample comum às três estratégias."
        ),
    }

    if not TORCH_DISPONIVEL:
        resultado_base["erro"] = (
            "PyTorch não está instalado; o modelo LSTM não pode ser executado. "
            "Instale com: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
        return resultado_base

    if df_enriquecido is None or df_enriquecido.empty or len(df_enriquecido) < 50:
        resultado_base["erro"] = "Dados insuficientes (mínimo 50 candles)"
        return resultado_base

    features = _preparar_features(df_enriquecido)
    target = _calcular_target(df_enriquecido)
    dados_combinados = pd.concat(
        [features, target.rename("target")], axis=1, sort=False
    ).dropna()

    n = len(dados_combinados)
    # Em timeframes semanais o histórico tem poucas amostras após a MM50.
    # Reduzimos a janela de sequência para permitir uma comparação possível,
    # registrando a decisão no resultado (e documentando a limitação).
    seq_len_original = seq_len
    seq_len = min(seq_len, max(4, n // 4))
    resultado_base["seq_len"] = seq_len
    resultado_base["seq_len_solicitado"] = seq_len_original
    if seq_len != seq_len_original:
        resultado_base["aviso_seq_len"] = (
            f"seq_len reduzido de {seq_len_original} para {seq_len} "
            f"devido ao tamanho da amostra ({n})"
        )

    if n < seq_len + 20:
        resultado_base["erro"] = (
            f"Dados insuficientes após features/sequências: {n} amostras "
            f"(seq_len={seq_len})"
        )
        return resultado_base

    n_treino = int(n * janela_treino_pct)
    if n_treino < seq_len + 5:
        resultado_base["erro"] = "Janela de treino insuficiente para o LSTM"
        return resultado_base

    _definir_seed(seed)

    dados_features = dados_combinados[features.columns].reset_index(drop=True)
    dados_target = dados_combinados["target"].reset_index(drop=True)
    datas = dados_combinados.index
    retornos_futuros = (
        df_enriquecido["Fechamento"]
        .pct_change()
        .shift(-1)
        .reindex(datas)
    )

    previsoes, probabilidades, indices_teste = [], [], []
    n_refits = 0
    scaler = StandardScaler()
    modelo = None

    for t in range(n_treino, n):
        # Re-treino periódico usando apenas dados até t (exclusive)
        if t == n_treino or (t - n_treino) % max(1, passo_retreino) == 0:
            features_treino = dados_features.iloc[:t].values
            target_treino = dados_target.iloc[:t].values

            X_treino, y_treino, _ = _criar_sequencias(
                pd.DataFrame(features_treino, columns=features.columns),
                pd.Series(target_treino),
                seq_len,
            )

            if len(X_treino) < 5 or len(np.unique(y_treino)) < 2:
                continue

            scaler = StandardScaler()
            features_treino_escalado = scaler.fit_transform(features_treino)
            X_treino, y_treino, _ = _criar_sequencias(
                pd.DataFrame(features_treino_escalado, columns=features.columns),
                pd.Series(target_treino),
                seq_len,
            )

            modelo = _treinar_lstm(
                X_treino=X_treino,
                y_treino=y_treino,
                n_features=features.shape[1],
                seq_len=seq_len,
                hidden_size=hidden_size,
                epochs=epochs,
                batch_size=batch_size,
            )
            n_refits += 1

        if modelo is None:
            continue

        # Predição no candle t usando somente o passado até t
        janela = dados_features.iloc[t - seq_len + 1:t + 1].values
        if len(janela) < seq_len:
            continue
        janela_escalada = scaler.transform(janela).astype(np.float32)

        with torch.no_grad():
            tensor = torch.from_numpy(janela_escalada).unsqueeze(0)
            logit = modelo(tensor).item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())

        previsoes.append(int(prob >= 0.5))
        probabilidades.append(prob)
        indices_teste.append(datas[t])

    if not previsoes:
        resultado_base["erro"] = "Nenhuma previsão gerada no walk-forward"
        return resultado_base

    previsoes_arr = np.asarray(previsoes, dtype=int)
    y_teste = np.asarray(
        [int(dados_target.iloc[datas.get_loc(idx)]) for idx in indices_teste],
        dtype=int,
    )
    retornos_teste = retornos_futuros.reindex(indices_teste).fillna(0.0).values

    acuracia = float((previsoes_arr == y_teste).mean())
    baseline = float(max(y_teste.mean(), 1 - y_teste.mean()))

    retorno_estrategia = retornos_teste * previsoes_arr
    retorno_acumulado = float((1 + retorno_estrategia).prod() - 1)

    curva = (1 + retorno_estrategia).cumprod()
    pico = np.maximum.accumulate(curva)
    dd = (curva - pico) / pico
    max_drawdown = float(dd.min())

    n_operacoes = int(np.abs(np.diff(np.concatenate([[0], previsoes_arr]))).sum())

    resultado_base.update({
        "n_treino": int(n_treino),
        "n_teste": int(len(previsoes_arr)),
        "n_refits": int(n_refits),
        "data_inicio_teste": indices_teste[0].isoformat(),
        "data_fim_teste": indices_teste[-1].isoformat(),
        "acuracia_direcional": round(acuracia, 4),
        "acuracia_baseline_maioria": round(baseline, 4),
        "retorno_acumulado": round(retorno_acumulado, 4),
        "retorno_total_pct": round(retorno_acumulado * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "n_operacoes": n_operacoes,
        "probabilidade_media_alta": round(float(np.mean(probabilidades)), 4),
    })
    return resultado_base


# ── Comparação consolidada ─────────────────────────────────────────────────────

def comparar_estrategias(
    par: str,
    intervalo: str = "1d",
    data_inicio: Optional[datetime] = None,
    data_fim: Optional[datetime] = None,
    df_precarregado: Optional[pd.DataFrame] = None,
    **kwargs_lstm,
) -> dict:
    """
    Compara as três estratégias para um par e timeframe específicos.

    A janela comum de avaliação é o período out-of-sample do LSTM
    (últimos 40% dos dados). As métricas de CriptoIA e Buy-and-Hold
    são recalculadas nessa mesma janela, além de serem reportadas no
    período completo como contexto.

    Parâmetros:
        par             : Par de negociação (ex.: 'BTC/USDT')
        intervalo       : Timeframe (ex.: '1d', '4h', '1h', '1w')
        data_inicio     : Início do período (padrão: DATA_INICIO_TCC)
        data_fim        : Fim do período (padrão: DATA_FIM_TCC)
        df_precarregado : DataFrame pré-carregado opcional (evita API)
        **kwargs_lstm   : Parâmetros do LSTM (epochs, seq_len etc.)

    Retorna:
        Dicionário com resultados completos e da janela comum.
    """
    if data_inicio is None:
        data_inicio = DATA_INICIO_TCC
    if data_fim is None:
        data_fim = DATA_FIM_TCC

    print(f"\n[Comparação] {par} | {intervalo}")
    resultado = {
        "par": par,
        "intervalo": intervalo,
        "data_inicio_str": data_inicio.strftime("%Y-%m-%d"),
        "data_fim_str": data_fim.strftime("%Y-%m-%d"),
    }

    try:
        if df_precarregado is not None:
            df_bruto = df_precarregado.copy()
        else:
            df_bruto = buscar_dados_historicos_paginado(
                par, intervalo, data_inicio, data_fim
            )

        if df_bruto.empty:
            resultado["erro"] = "Nenhum dado retornado pela API."
            return resultado

        if "MM20" in df_bruto.columns and "RSI14" in df_bruto.columns:
            df_enriquecido = df_bruto
        else:
            df_enriquecido = enriquecer_dataframe(df_bruto)

        # Período completo (contexto)
        metricas_crip = calcular_metricas(df_enriquecido)
        metricas_crip["estrategia"] = "CriptoIA"
        resultado["criptoia"] = metricas_crip
        resultado["buy_and_hold"] = calcular_buy_and_hold(df_bruto)

        # LSTM com walk-forward
        supervisionado = executar_modelo_supervisionado(df_enriquecido, **kwargs_lstm)
        resultado["supervisionado"] = supervisionado

        # Janela comum: últimos 40% (início do teste do LSTM)
        data_inicio_teste = supervisionado.get("data_inicio_teste")
        if data_inicio_teste:
            inicio_ts = pd.Timestamp(data_inicio_teste)
            resultado["periodo_comum"] = {
                "data_inicio": data_inicio_teste,
                "data_fim": supervisionado.get("data_fim_teste"),
                "criptoia": calcular_metricas(
                    df_enriquecido, data_inicio_avaliacao=inicio_ts
                ),
                "buy_and_hold": calcular_buy_and_hold(df_bruto, data_inicio=inicio_ts),
                "supervisionado": supervisionado,
                "observacao": (
                    "Métricas recalculadas no mesmo período out-of-sample "
                    "do LSTM para comparação justa."
                ),
            }
        else:
            resultado["periodo_comum"] = {
                "erro": "Período comum indisponível porque o LSTM não gerou previsões."
            }

    except Exception as erro:
        resultado["erro"] = f"{type(erro).__name__}: {erro}"
        print(f"  [ERRO] {par} {intervalo}: {erro}")

    return resultado


def comparar_todos_pares(
    pares: Optional[list] = None,
    timeframes: Optional[list] = None,
    data_inicio: Optional[datetime] = None,
    data_fim: Optional[datetime] = None,
    max_workers: int = 1,
    **kwargs_lstm,
) -> list:
    """
    Executa a comparação de estratégias para todos os pares e timeframes.

    A configuração padrão usa todos os timeframes do TCC. Como o LSTM é
    re-treinado periodicamente, a execução completa é computacionalmente
    mais cara que o backtesting simples.

    Parâmetros:
        pares      : Lista de pares (padrão: PARES_TCC)
        timeframes : Lista de timeframes (padrão: TIMEFRAMES_TCC)
        data_inicio: Início (padrão: DATA_INICIO_TCC)
        data_fim   : Fim (padrão: DATA_FIM_TCC)
        max_workers: Threads (padrão: 1, para evitar competição por CPU)
        **kwargs_lstm: Parâmetros do LSTM

    Retorna:
        Lista de dicionários de comparação, na ordem de entrada.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if pares is None:
        pares = PARES_TCC
    if timeframes is None:
        timeframes = TIMEFRAMES_TCC

    print("\n" + "=" * 60)
    print("  COMPARAÇÃO DE ESTRATÉGIAS — CriptoIA")
    print("=" * 60)

    combinacoes = [(par, tf) for par in pares for tf in timeframes]
    resultados_por_chave = {}

    def _executar(par, tf):
        return comparar_estrategias(
            par, tf, data_inicio, data_fim, **kwargs_lstm
        )

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futuros = {
                executor.submit(_executar, par, tf): (par, tf)
                for par, tf in combinacoes
            }
            for futuro in as_completed(futuros):
                par, tf = futuros[futuro]
                try:
                    resultados_por_chave[(par, tf)] = futuro.result()
                except Exception as erro:
                    print(f"  [ERRO CRÍTICO] {par} {tf}: {erro}")
                    resultados_por_chave[(par, tf)] = {
                        "par": par,
                        "intervalo": tf,
                        "erro": f"{type(erro).__name__}: {erro}",
                    }
    else:
        for par, tf in combinacoes:
            try:
                resultados_por_chave[(par, tf)] = _executar(par, tf)
            except Exception as erro:
                print(f"  [ERRO CRÍTICO] {par} {tf}: {erro}")
                resultados_por_chave[(par, tf)] = {
                    "par": par,
                    "intervalo": tf,
                    "erro": f"{type(erro).__name__}: {erro}",
                }

    return [resultados_por_chave[(par, tf)] for par, tf in combinacoes]


def salvar_resultados_comparacao(
    resultados: list,
    caminho_json: str = "resultados/comparacao_tcc.json",
    caminho_csv: Optional[str] = "resultados/comparacao_tcc.csv",
    periodo: Optional[dict] = None,
) -> dict:
    """Salva os resultados da comparação em JSON e CSV."""
    for caminho in [caminho_json, caminho_csv]:
        if caminho:
            diretorio = os.path.dirname(os.path.abspath(caminho))
            os.makedirs(diretorio, exist_ok=True)

    payload = {
        "gerado_em": datetime.now().isoformat(),
        "periodo": periodo or {
            "inicio": DATA_INICIO_TCC.isoformat(),
            "fim": DATA_FIM_TCC.isoformat(),
        },
        "metodologia": {
            "criptoia": "sinais determinísticos de indicadores (proxy)",
            "buy_and_hold": "passivo",
            "supervisionado": "LSTM PyTorch com walk-forward",
            "janela_comum": "últimos 40% (out-of-sample do LSTM)",
        },
        "resultados": resultados,
    }

    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(payload, arquivo, ensure_ascii=False, indent=2, default=str)

    caminhos = {"json": os.path.abspath(caminho_json)}

    if caminho_csv:
        linhas = []
        for res in resultados:
            comum = res.get("periodo_comum", {})
            linhas.append({
                "par": res.get("par"),
                "intervalo": res.get("intervalo"),
                "erro": res.get("erro"),
                "criptoia_retorno_pct_completo": (res.get("criptoia") or {}).get("retorno_total_pct"),
                "criptoia_acuracia_completo": (res.get("criptoia") or {}).get("acuracia_direcional"),
                "bnh_retorno_pct_completo": (res.get("buy_and_hold") or {}).get("retorno_total_pct"),
                "lstm_retorno_pct": (res.get("supervisionado") or {}).get("retorno_total_pct"),
                "lstm_acuracia": (res.get("supervisionado") or {}).get("acuracia_direcional"),
                "lstm_baseline": (res.get("supervisionado") or {}).get("acuracia_baseline_maioria"),
                "comum_criptoia_retorno_pct": (comum.get("criptoia") or {}).get("retorno_total_pct") if isinstance(comum, dict) else None,
                "comum_criptoia_acuracia": (comum.get("criptoia") or {}).get("acuracia_direcional") if isinstance(comum, dict) else None,
                "comum_bnh_retorno_pct": (comum.get("buy_and_hold") or {}).get("retorno_total_pct") if isinstance(comum, dict) else None,
                "comum_lstm_retorno_pct": (comum.get("supervisionado") or {}).get("retorno_total_pct") if isinstance(comum, dict) else None,
                "comum_lstm_acuracia": (comum.get("supervisionado") or {}).get("acuracia_direcional") if isinstance(comum, dict) else None,
                "periodo_inicio_teste": (res.get("supervisionado") or {}).get("data_inicio_teste"),
                "periodo_fim_teste": (res.get("supervisionado") or {}).get("data_fim_teste"),
            })
        pd.DataFrame(linhas).to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        caminhos["csv"] = os.path.abspath(caminho_csv)

    return caminhos


def formatar_relatorio_comparacao(resultados: list) -> str:
    """
    Formata o relatório comparativo em texto legível.

    Mostra a janela comum (comparação justa) e o período completo
    (contexto) para CriptoIA e Buy-and-Hold.
    """
    linhas = [
        "=" * 80,
        "  RELATÓRIO COMPARATIVO DE ESTRATÉGIAS — CriptoIA",
        f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "=" * 80,
        "",
        "ESTRATÉGIAS COMPARADAS:",
        "  1. CriptoIA      : Sinais determinísticos por indicadores (proxy)",
        "  2. Buy-and-Hold  : Manter o ativo durante todo o período",
        "  3. LSTM          : Rede LSTM PyTorch com walk-forward",
        "",
        "METODOLOGIA:",
        "  - Janela comum = últimos 40% dos dados (out-of-sample do LSTM)",
        "  - LSTM re-treinado periodicamente apenas com dados passados",
        "  - Retorno do LSTM usa o retorno FUTURO (t+1), sem look-ahead",
        "  - Sem custos de transação",
        "",
    ]

    for res in resultados:
        par = res.get("par", "?")
        tf = res.get("intervalo", "?")
        linhas.append(f"{'─' * 70}")
        linhas.append(f"  {par} | {tf} | {res.get('data_inicio_str', '?')} -> {res.get('data_fim_str', '?')}")
        linhas.append(f"{'─' * 70}")

        if "erro" in res:
            linhas.append(f"  ❌ ERRO: {res['erro']}")
            linhas.append("")
            continue

        # Período comum
        comum = res.get("periodo_comum") or {}
        linhas.append("  [Janela comum de avaliação]")
        if isinstance(comum, dict) and "erro" not in comum:
            linhas.append(f"  Período: {comum.get('data_inicio')} -> {comum.get('data_fim')}")
            linhas.append(f"  {'Estratégia':<22} {'Retorno(%)':>12} {'Acurácia(%)':>13} {'Max.DD(%)':>10}")
            linhas.append(f"  {'-'*22} {'-'*12} {'-'*13} {'-'*10}")
            for chave, label in [
                ("criptoia", "CriptoIA (proxy)"),
                ("buy_and_hold", "Buy-and-Hold"),
                ("supervisionado", "Supervisionado (LSTM)"),
            ]:
                bruto = comum.get(chave)
                est = bruto if isinstance(bruto, dict) else {}
                if "erro" in est:
                    linhas.append(f"  {label:<22} {'ERRO':>12}")
                    continue
                ret = est.get("retorno_total_pct", "N/A")
                acu = est.get("acuracia_direcional")
                acu_str = f"{acu*100:.1f}" if acu is not None else "N/A"
                dd = est.get("max_drawdown_pct", "N/A")
                linhas.append(f"  {label:<22} {str(ret):>12} {acu_str:>13} {str(dd):>10}")
        else:
            linhas.append(f"  ❌ {comum.get('erro', 'indisponível') if isinstance(comum, dict) else 'indisponível'}")

        # Contexto completo
        linhas.append("  [Período completo — contexto]")
        for chave, label in [("criptoia", "CriptoIA (proxy)"), ("buy_and_hold", "Buy-and-Hold")]:
            est = res.get(chave) or {}
            if not isinstance(est, dict):
                est = {}
            if "erro" in est:
                linhas.append(f"  {label:<22} {'ERRO':>12}")
                continue
            ret = est.get("retorno_total_pct", "N/A")
            acu = est.get("acuracia_direcional")
            acu_str = f"{acu*100:.1f}%" if acu is not None else "N/A"
            linhas.append(f"  {label:<22} retorno={ret}% | acurácia={acu_str}")

        sup = res.get("supervisionado") or {}
        if not isinstance(sup, dict):
            sup = {}
        if "erro" not in sup:
            linhas.append(
                f"  Supervisionado (LSTM)  retorno={sup.get('retorno_total_pct', 'N/A')}% | "
                f"acurácia={sup.get('acuracia_direcional')} | "
                f"baseline={sup.get('acuracia_baseline_maioria')} | "
                f"seq_len={sup.get('seq_len')} | epochs={sup.get('epochs')}"
            )
        linhas.append("")

    return "\n".join(linhas)
