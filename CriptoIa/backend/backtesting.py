"""
backtesting.py — Módulo de backtesting histórico do CriptoIA.

Avalia a estratégia de sinais do CriptoIA com dados históricos reais,
cobrindo os pares BTC/USDT, ETH/USDT, SOL/USDT e DOGE/USDT no
período jan/2025–jun/2026 nos timeframes 1h, 4h, 1d e 1w.

Metodologia:
    - Sinal baseado nos indicadores técnicos (sem chamar API Gemini
      a cada candle, conforme viável acadêmica e financeiramente).
    - Regras da estratégia CriptoIA:
        COMPRAR : MM20 > MM50  AND  RSI14 < 70  AND  MACD > Sinal_MACD
        VENDER  : MM20 < MM50  AND  RSI14 > 30  AND  MACD < Sinal_MACD
        MANTER  : demais casos
    - Decisão aplicada ao candle SEGUINTE para evitar look-ahead bias:
      o sinal gerado no fechamento do candle t é aplicado ao retorno
      close-to-close do candle t+1 (retorno futuro via shift(-1)).
    - Retorno simulado: 1 unidade de capital (long-only, sem alavancagem,
      sem custos de transação — limitação documentada).
    - Acurácia direcional: % de sinais direcionais (±1) que previram
      corretamente a direção do próximo candle.

Importante (limitação metodológica):
    Este módulo avalia uma estratégia determinística de indicadores,
    usada como PROXY da lógica técnica do CriptoIA. Ele NÃO reproduz as
    recomendações do Gemini, pois isso exigiria uma chamada multimodal
    por candle (inviável em custo e tempo para 4 pares × 4 timeframes).
    Os resultados devem ser apresentados como avaliação da estratégia de
    indicadores, não como acurácia das respostas do Gemini.

Limitações documentadas:
    - Não inclui custos de transação (spread, taxas, slippage).
    - Estratégia long-only; não considera posições vendidas.
    - A API da Binance retorna no máximo 1000 candles por requisição;
      para o timeframe 1h e 18 meses (~13.140 candles) é necessário
      múltiplas requisições (paginação), implementadas neste módulo.
    - Se a paginação falhar, o resultado é marcado como incompleto e não
      é apresentado como se cobrisse todo o período.
    - O período jan/2025–jun/2026 é verificado com os dados disponíveis
      na API; candles futuros não são incluídos.
    - RSI com janela rolling simples (não EMA de Wilder); comportamento
      idêntico ao módulo analise_tecnica.py do projeto.
    - A acurácia usa o candle seguinte; o TCC prevê avaliar o prazo de
      `tempo_estimado` do Gemini, que não existe neste proxy.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

# Garante saída compatível com Windows
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from backend.coleta_dados import obter_cliente_binance
from backend.analise_tecnica import (
    MINIMO_CANDLES_INDICADORES,
    enriquecer_dataframe,
)


# ── Parâmetros do experimento acadêmico (TCC) ─────────────────────────────────
PARES_TCC = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]
TIMEFRAMES_TCC = ["1h", "4h", "1d", "1w"]

# Período do experimento acadêmico
DATA_INICIO_TCC = datetime(2025, 1, 1, tzinfo=timezone.utc)
DATA_FIM_TCC = datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

# Limites de candles por requisição (Binance via CCXT)
LIMITE_CANDLES_POR_REQUISICAO = 1000

# Mínimo de candles para calcular métricas com os indicadores completos
MINIMO_CANDLES_BACKTESTING = 60

# Identificador metodológico usado nos relatórios e na interface
METODOLOGIA_PROXY = "sinais_indicadores_proxy"


# ── Utilidades ────────────────────────────────────────────────────────────────

def _normalizar_data(data: datetime) -> datetime:
    """Garante que uma data seja timezone-aware em UTC."""
    if data.tzinfo is None:
        return data.replace(tzinfo=timezone.utc)
    return data.astimezone(timezone.utc)


def _timestamp_compativel(index: pd.Index, data) -> pd.Timestamp:
    """
    Converte `data` para um Timestamp compatível com o timezone do índice.

    Evita o erro pandas "Invalid comparison between dtype=datetime64[us]
    and Timestamp" em dados sintéticos tz-naive e dados da Binance tz-aware.
    """
    ts = pd.Timestamp(data)
    tz_indice = getattr(index, "tz", None)
    if tz_indice is None:
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    else:
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz_indice)
        else:
            ts = ts.tz_convert(tz_indice)
    return ts


def _validar_parametros(
    par: str,
    intervalo: str,
    data_inicio: datetime,
    data_fim: datetime,
) -> None:
    """Valida os parâmetros mínimos de uma execução de backtesting."""
    if (
        not isinstance(par, str)
        or not re.fullmatch(r"[A-Za-z0-9]{2,20}/[A-Za-z0-9]{2,20}", par.strip())
    ):
        raise ValueError("Par inválido. Informe um par no formato BASE/QUOTE, ex.: BTC/USDT.")
    if not isinstance(intervalo, str) or not intervalo.strip():
        raise ValueError("Timeframe inválido. Informe, por exemplo, 1h, 4h, 1d ou 1w.")
    if _normalizar_data(data_inicio) >= _normalizar_data(data_fim):
        raise ValueError("A data de início deve ser anterior à data de fim.")


# ── Coleta paginada para períodos longos ───────────────────────────────────────

def buscar_dados_historicos_paginado(
    par: str,
    intervalo: str,
    data_inicio: datetime,
    data_fim: datetime,
) -> pd.DataFrame:
    """
    Busca dados OHLCV da Binance para um período específico,
    realizando múltiplas requisições paginadas quando necessário.

    Parâmetros:
        par         : Par de negociação, ex: 'BTC/USDT'
        intervalo   : Timeframe dos candles, ex: '1d', '4h', '1h', '1w'
        data_inicio : Data/hora de início (timezone-aware UTC)
        data_fim    : Data/hora de fim (timezone-aware UTC)

    Retorna:
        DataFrame com colunas: Abertura, Maxima, Minima, Fechamento, Volume
        indexado por timestamp (UTC). Pode ser vazio se não houver dados.

        O DataFrame possui atributos de auditoria:
            - coleta_incompleta (bool)
            - erro_coleta (str | None)
            - periodo_solicitado (tuple)

    Notas:
        - Limita-se a 1000 candles por requisição (limite da Binance).
        - Respeita rate limit com pausa entre requisições.
        - Exclui candles fora do período solicitado.
        - Em caso de falha de rede no meio da paginação, retorna os candles
          já obtidos, mas marca coleta_incompleta=True para que o chamador
          não trate o resultado como se cobrisse todo o período.
    """
    data_inicio = _normalizar_data(data_inicio)
    data_fim = _normalizar_data(data_fim)
    _validar_parametros(par, intervalo, data_inicio, data_fim)

    cliente = obter_cliente_binance()
    todos_candles = []
    erro_coleta = None

    since_ms = int(data_inicio.timestamp() * 1000)
    fim_ms = int(data_fim.timestamp() * 1000)

    print(f"  [Binance] Buscando {par} {intervalo}: "
          f"{data_inicio.strftime('%Y-%m-%d')} -> {data_fim.strftime('%Y-%m-%d')}")

    while since_ms < fim_ms:
        try:
            candles = cliente.fetch_ohlcv(
                par,
                timeframe=intervalo,
                since=since_ms,
                limit=LIMITE_CANDLES_POR_REQUISICAO,
            )
        except Exception as erro:
            msg = str(erro).lower()
            if "signature" in msg or "key" in msg or "1022" in msg:
                cliente = obter_cliente_binance(forcar_publico=True)
                try:
                    candles = cliente.fetch_ohlcv(
                        par,
                        timeframe=intervalo,
                        since=since_ms,
                        limit=LIMITE_CANDLES_POR_REQUISICAO,
                    )
                except Exception as e2:
                    erro_coleta = str(e2)
                    print(f"  [AVISO] Erro ao buscar dados: {e2}. Coleta marcada como incompleta.")
                    break
            else:
                erro_coleta = str(erro)
                print(f"  [AVISO] Erro ao buscar dados: {erro}. Coleta marcada como incompleta.")
                break

        if not candles:
            break

        todos_candles.extend(candles)

        ultimo_ts = candles[-1][0]
        if ultimo_ts >= fim_ms or len(candles) < LIMITE_CANDLES_POR_REQUISICAO:
            break

        since_ms = ultimo_ts + 1
        time.sleep(0.3)  # Respeita rate limit da Binance

    if not todos_candles:
        print(f"  [AVISO] Nenhum dado retornado para {par} {intervalo}.")
        df_vazio = pd.DataFrame()
        df_vazio.attrs["coleta_incompleta"] = erro_coleta is not None
        df_vazio.attrs["erro_coleta"] = erro_coleta
        df_vazio.attrs["periodo_solicitado"] = (data_inicio, data_fim)
        return df_vazio

    df = pd.DataFrame(
        todos_candles,
        columns=["timestamp", "Abertura", "Maxima", "Minima", "Fechamento", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp")
    df = df.set_index("timestamp")
    df = df.sort_index()

    # Filtra exatamente pelo período solicitado (inclusive nas duas bordas)
    inicio_pd = pd.Timestamp(data_inicio)
    fim_pd = pd.Timestamp(data_fim)
    df = df[(df.index >= inicio_pd) & (df.index <= fim_pd)]

    df.attrs["coleta_incompleta"] = erro_coleta is not None
    df.attrs["erro_coleta"] = erro_coleta
    df.attrs["periodo_solicitado"] = (data_inicio, data_fim)

    print(f"  [Binance] {len(df)} candles carregados ({par} {intervalo}).")
    if erro_coleta:
        print(f"  [AVISO] Coleta incompleta: {erro_coleta}")
    return df


# ── Geração de sinais ─────────────────────────────────────────────────────────

def gerar_sinais(df_enriquecido: pd.DataFrame) -> pd.Series:
    """
    Aplica as regras da estratégia CriptoIA para gerar sinais de negociação.

    Regras (baseadas nos indicadores calculados em analise_tecnica.py):
        COMPRAR (+1) : MM20 > MM50  AND  RSI14 < 70  AND  MACD > Sinal_MACD
        VENDER  (-1) : MM20 < MM50  AND  RSI14 > 30  AND  MACD < Sinal_MACD
        MANTER  (0)  : demais casos

    Parâmetros:
        df_enriquecido : DataFrame com indicadores (saída de enriquecer_dataframe)

    Retorna:
        Series com valores -1, 0 ou +1 (mesmo índice do DataFrame)
    """
    df = df_enriquecido.copy()

    cond_comprar = (
        (df["MM20"] > df["MM50"]) &
        (df["RSI14"] < 70) &
        (df["MACD"] > df["Sinal_MACD"])
    )

    cond_vender = (
        (df["MM20"] < df["MM50"]) &
        (df["RSI14"] > 30) &
        (df["MACD"] < df["Sinal_MACD"])
    )

    sinais = pd.Series(0, index=df.index, dtype=int, name="Sinal")
    sinais[cond_comprar] = 1
    sinais[cond_vender] = -1

    return sinais


# ── Cálculo de métricas ───────────────────────────────────────────────────────

def calcular_metricas(
    df_enriquecido: pd.DataFrame,
    data_inicio_avaliacao: Optional[datetime] = None,
) -> dict:
    """
    Calcula as métricas de desempenho do backtesting para um par/timeframe.

    Anti look-ahead bias:
        - O sinal gerado no candle t é aplicado ao retorno do candle t+1.
        - Usamos o retorno futuro (close[t+1]/close[t]-1, via shift(-1))
          e o sinal corrente. O sinal usa apenas indicadores até t; o retorno
          futuro é usado somente para medir o acerto, nunca para decidir.

    Métricas calculadas:
        - n_candles              : Total de candles no período
        - n_candles_validos      : Candles usados após remover warm-up/NaN
        - n_sinais_compra/venda/manter : Contagens no período avaliado
        - distribuicao_acoes     : Frequência relativa (%) de cada ação
        - acuracia_direcional    : % de sinais direcionais (±1) corretos
        - retorno_acumulado      : Retorno percentual acumulado (long-only)
        - retorno_total_pct      : Igual a retorno_acumulado × 100 (%)
        - max_drawdown_pct       : Máximo drawdown percentual
        - n_operacoes            : Mudanças de posição (entradas/saídas)
        - n_entradas             : Transições para posição comprada

    Parâmetros:
        df_enriquecido          : DataFrame com indicadores
        data_inicio_avaliacao   : Início opcional da janela de avaliação
                                  (usado para comparar estratégias no mesmo
                                  período out-of-sample)

    Retorna:
        Dicionário com as métricas calculadas
    """
    if df_enriquecido is None or df_enriquecido.empty or len(df_enriquecido) < MINIMO_CANDLES_BACKTESTING:
        return {
            "erro": (
                "Dados insuficientes para backtesting "
                f"(mínimo {MINIMO_CANDLES_BACKTESTING} candles)"
            )
        }

    df = df_enriquecido.copy()

    colunas_indicadores = ["MM20", "MM50", "RSI14", "MACD", "Sinal_MACD"]
    colunas_ausentes = [c for c in colunas_indicadores if c not in df.columns]
    if colunas_ausentes:
        return {"erro": f"Indicadores ausentes: {', '.join(colunas_ausentes)}"}

    # Indicadores completos: evita contar o warm-up (MM50 etc.) como MANTER
    mask_indicadores = df[colunas_indicadores].notna().all(axis=1)

    # Sinal no candle t, aplicado ao retorno close-to-close de t+1.
    # O retorno futuro NÃO é usado para gerar o sinal; serve apenas para
    # medir o resultado da decisão já tomada.
    sinais = gerar_sinais(df)
    df["Sinal"] = sinais
    df["Retorno_Futuro"] = df["Fechamento"].pct_change().shift(-1)

    mask_avaliacao = mask_indicadores & df["Retorno_Futuro"].notna()

    if data_inicio_avaliacao is not None:
        inicio_ts = _timestamp_compativel(df.index, data_inicio_avaliacao)
        mask_avaliacao = mask_avaliacao & (df.index >= inicio_ts)

    df_avaliacao = df.loc[mask_avaliacao].copy()

    if df_avaliacao.empty:
        return {"erro": "Sem dados válidos após remoção de NaN/warm-up"}

    # Retorno da estratégia: long-only; sinal +1 comprado, 0/-1 em caixa.
    # O sinal de t é multiplicado pelo retorno de t -> t+1.
    df_avaliacao["Retorno_Estrategia"] = np.where(
        df_avaliacao["Sinal"] == 1,
        df_avaliacao["Retorno_Futuro"],
        0.0,
    )

    # ── Acurácia direcional ────────────────────────────────────────────────────
    # Considera apenas sinais direcionais (±1); MANTER não tem direção
    df_dir = df_avaliacao[df_avaliacao["Sinal"].isin([-1, 1])]

    if not df_dir.empty:
        direcao_mercado = (df_dir["Retorno_Futuro"] > 0).astype(int)
        direcao_sinal = (df_dir["Sinal"] == 1).astype(int)
        acuracia = float((direcao_sinal == direcao_mercado).mean())
    else:
        acuracia = float("nan")

    # ── Retorno acumulado ──────────────────────────────────────────────────────
    retorno_acumulado = float(
        (1 + df_avaliacao["Retorno_Estrategia"]).prod() - 1
    )

    # ── Máximo drawdown ────────────────────────────────────────────────────────
    curva_capital = (1 + df_avaliacao["Retorno_Estrategia"]).cumprod()
    pico = curva_capital.cummax()
    drawdown = (curva_capital - pico) / pico
    max_drawdown = float(drawdown.min())

    # ── Contagem de operações e distribuição de ações ─────────────────────────
    sinais_avaliados = df_avaliacao["Sinal"]
    total_sinais = len(sinais_avaliados)
    n_compra = int((sinais_avaliados == 1).sum())
    n_venda = int((sinais_avaliados == -1).sum())
    n_manter = int((sinais_avaliados == 0).sum())

    # Entradas: de caixa/venda para compra
    posicao = df_avaliacao["Sinal"]
    entradas = ((posicao == 1) & (posicao.shift(1).fillna(0) != 1)).sum()
    mudancas = posicao.diff().fillna(0).ne(0).sum()

    return {
        "n_candles": int(len(df)),
        "n_candles_validos": int(len(df_avaliacao)),
        "n_candles_indicadores": int(mask_indicadores.sum()),
        "n_sinais_compra": n_compra,
        "n_sinais_venda": n_venda,
        "n_sinais_manter": n_manter,
        "distribuicao_acoes": {
            "COMPRAR_pct": round(100 * n_compra / total_sinais, 2) if total_sinais else 0.0,
            "VENDER_pct": round(100 * n_venda / total_sinais, 2) if total_sinais else 0.0,
            "MANTER_pct": round(100 * n_manter / total_sinais, 2) if total_sinais else 0.0,
        },
        "acuracia_direcional": round(acuracia, 4) if not np.isnan(acuracia) else None,
        "retorno_acumulado": round(retorno_acumulado, 4),
        "retorno_total_pct": round(retorno_acumulado * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "n_operacoes": int(mudancas),
        "n_entradas": int(entradas),
        "data_inicio_avaliacao": (
            df_avaliacao.index[0].isoformat()
            if data_inicio_avaliacao is not None
            else None
        ),
        "data_fim_avaliacao": df_avaliacao.index[-1].isoformat(),
    }


# ── Execução principal ─────────────────────────────────────────────────────────

def executar_backtesting(
    par: str,
    intervalo: str,
    data_inicio: Optional[datetime] = None,
    data_fim: Optional[datetime] = None,
) -> dict:
    """
    Executa o backtesting completo para um par e timeframe específicos.

    Parâmetros:
        par         : Par de negociação (ex: 'BTC/USDT')
        intervalo   : Timeframe (ex: '1d', '4h', '1h', '1w')
        data_inicio : Início do período (padrão: DATA_INICIO_TCC = jan/2025)
        data_fim    : Fim do período (padrão: DATA_FIM_TCC = jun/2026)

    Retorna:
        Dicionário com:
            par, intervalo, data_inicio_str, data_fim_str,
            n_candles, métricas e possivelmente 'erro'
    """
    if data_inicio is None:
        data_inicio = DATA_INICIO_TCC
    if data_fim is None:
        data_fim = DATA_FIM_TCC

    data_inicio = _normalizar_data(data_inicio)
    data_fim = _normalizar_data(data_fim)

    print(f"\n[Backtesting] {par} | {intervalo}")
    resultado = {
        "par": par,
        "intervalo": intervalo,
        "data_inicio_str": data_inicio.strftime("%Y-%m-%d"),
        "data_fim_str": data_fim.strftime("%Y-%m-%d"),
        "metodologia": METODOLOGIA_PROXY,
        "avalia_gemini": False,
        "observacao_metodologica": (
            "Estratégia determinística de indicadores usada como proxy; "
            "não representa as recomendações multimodais do Gemini."
        ),
    }

    try:
        _validar_parametros(par, intervalo, data_inicio, data_fim)

        df_bruto = buscar_dados_historicos_paginado(
            par, intervalo, data_inicio, data_fim
        )

        if df_bruto.empty:
            resultado["erro"] = (
                "Nenhum dado retornado pela API para o período solicitado. "
                "Verifique o par, o timeframe e a conexão com a Binance."
            )
            return resultado

        if df_bruto.attrs.get("coleta_incompleta"):
            resultado["erro"] = (
                "Coleta histórica incompleta; o resultado não cobre todo o "
                f"período solicitado. Detalhe: {df_bruto.attrs.get('erro_coleta')}"
            )
            resultado["n_candles_coletados"] = int(len(df_bruto))
            return resultado

        if len(df_bruto) < MINIMO_CANDLES_INDICADORES:
            resultado["erro"] = (
                f"Dados insuficientes: {len(df_bruto)} candles coletados; "
                f"mínimo de {MINIMO_CANDLES_INDICADORES} para os indicadores."
            )
            return resultado

        df_enriquecido = enriquecer_dataframe(df_bruto)
        metricas = calcular_metricas(df_enriquecido)

        if "erro" in metricas:
            resultado.update(metricas)
            return resultado

        resultado.update(metricas)
        resultado.update({
            "n_candles_coletados": int(len(df_bruto)),
            "primeiro_candle": df_bruto.index[0].isoformat(),
            "ultimo_candle": df_bruto.index[-1].isoformat(),
            "cobertura_solicitada_pct": round(
                100 * len(df_bruto) / max(1, _estimar_candles(data_inicio, data_fim, intervalo)),
                2,
            ),
        })

    except Exception as erro:
        resultado["erro"] = f"{type(erro).__name__}: {erro}"
        print(f"  [ERRO] {par} {intervalo}: {erro}")

    return resultado


def _estimar_candles(data_inicio: datetime, data_fim: datetime, intervalo: str) -> int:
    """Estima a quantidade esperada de candles para uma cobertura."""
    segundos_por_tf = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800,
    }
    segundos = segundos_por_tf.get(intervalo, 86400)
    duracao = (data_fim - data_inicio).total_seconds()
    # +1 porque o período é inclusivo nas duas bordas
    return max(1, int(duracao / segundos) + 1)


def executar_backtesting_completo(
    pares: Optional[list] = None,
    timeframes: Optional[list] = None,
    data_inicio: Optional[datetime] = None,
    data_fim: Optional[datetime] = None,
    max_workers: int = 2,
) -> list:
    """
    Executa backtesting para todos os pares e timeframes especificados.

    Parâmetros:
        pares       : Lista de pares (padrão: PARES_TCC)
        timeframes  : Lista de timeframes (padrão: TIMEFRAMES_TCC)
        data_inicio : Início do período (padrão: DATA_INICIO_TCC)
        data_fim    : Fim do período (padrão: DATA_FIM_TCC)
        max_workers : Número de threads para coleta/processamento (padrão: 2).
                      Use 1 para execução sequencial e menor pressão na API.

    Retorna:
        Lista de dicionários de resultados por par/timeframe, na ordem
        de entrada. Erros individuais são capturados e não interrompem
        os demais experimentos.
    """
    if pares is None:
        pares = PARES_TCC
    if timeframes is None:
        timeframes = TIMEFRAMES_TCC

    print("\n" + "=" * 60)
    print("  BACKTESTING HISTÓRICO — CriptoIA")
    print("=" * 60)

    combinacoes = [(par, tf) for par in pares for tf in timeframes]
    resultados_por_chave = {}
    total = len(combinacoes)
    atual = 0

    def _executar(par: str, tf: str) -> dict:
        return executar_backtesting(par, tf, data_inicio, data_fim)

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
                    resultados_por_chave[(par, tf)] = {
                        "par": par,
                        "intervalo": tf,
                        "erro": f"{type(erro).__name__}: {erro}",
                    }
                atual += 1
                print(f"  [{atual}/{total}] {par} ({tf}) processado.")
    else:
        for par, tf in combinacoes:
            atual += 1
            print(f"\n  [{atual}/{total}] Processando {par} ({tf})...")
            try:
                resultados_por_chave[(par, tf)] = _executar(par, tf)
            except Exception as erro:
                print(f"  [ERRO CRÍTICO] {par} {tf}: {erro}")
                resultados_por_chave[(par, tf)] = {
                    "par": par,
                    "intervalo": tf,
                    "erro": f"{type(erro).__name__}: {erro}",
                }

    resultados = [resultados_por_chave[(par, tf)] for par, tf in combinacoes]

    print(f"\n  Backtesting concluído: {len(resultados)} combinações processadas.")
    return resultados


def resumir_criterios_sucesso(resultados: list) -> dict:
    """
    Resume os critérios da Seção IV do TCC:
      - acurácia direcional > 50% em pelo menos 3 dos 4 ativos;
      - retorno acumulado simulado não negativo no período.

    A avaliação é feita por par, usando o timeframe 1d quando disponível;
    caso contrário, usa a média dos timeframes válidos. O TCC1 não define
    qual timeframe deve representar o resumo dos critérios; o recorte 1d é
    uma escolha operacional deste trabalho, não um parâmetro do TCC. O
    resumo é apenas informativo e não altera os resultados brutos.
    """
    por_par = {}
    for res in resultados:
        if "erro" in res or res.get("acuracia_direcional") is None:
            continue
        por_par.setdefault(res["par"], []).append(res)

    resumo = {}
    for par, lista in por_par.items():
        # Preferência pelo diário; se ausente, usa todos os timeframes
        diarios = [r for r in lista if r.get("intervalo") == "1d"]
        base = diarios if diarios else lista
        acuracias = [r["acuracia_direcional"] for r in base if r.get("acuracia_direcional") is not None]
        retornos = [r["retorno_total_pct"] for r in base if r.get("retorno_total_pct") is not None]
        resumo[par] = {
            "acuracia_media": round(float(np.mean(acuracias)), 4) if acuracias else None,
            "retorno_medio_pct": round(float(np.mean(retornos)), 2) if retornos else None,
            "acuracia_acima_50": bool(acuracias and np.mean(acuracias) > 0.5),
            "retorno_nao_negativo": bool(retornos and np.mean(retornos) >= 0),
            "timeframes_avaliados": [r.get("intervalo") for r in base],
        }

    ativos_acima_50 = sum(1 for v in resumo.values() if v["acuracia_acima_50"])
    ativos_retorno_ok = sum(1 for v in resumo.values() if v["retorno_nao_negativo"])

    return {
        "por_par": resumo,
        "ativos_avaliados": len(resumo),
        "ativos_acuracia_acima_50": ativos_acima_50,
        "ativos_retorno_nao_negativo": ativos_retorno_ok,
        "criterio_acuracia_atendido": ativos_acima_50 >= 3,
        "criterio_retorno_atendido": ativos_retorno_ok == len(resumo) and len(resumo) > 0,
        "recorte_principal": "1d" if any(
            r.get("intervalo") == "1d" for r in resultados
        ) else "média dos timeframes válidos",
        "nota_timeframe": (
            "O TCC1 não especifica o timeframe do resumo dos critérios. "
            "O recorte 1d (ou todos os timeframes válidos quando não há 1d) "
            "é uma escolha operacional deste trabalho, não um parâmetro do TCC."
        ),
        "observacao": (
            "Critério aplicado à estratégia de indicadores (proxy), "
            "não às recomendações do Gemini."
        ),
    }


def salvar_resultados_backtesting(
    resultados: list,
    caminho_json: str = "resultados/backtesting_tcc.json",
    caminho_csv: Optional[str] = "resultados/backtesting_tcc.csv",
    periodo: Optional[dict] = None,
) -> dict:
    """
    Salva os resultados brutos em JSON e, opcionalmente, CSV.

    Retorna os caminhos absolutos gerados.
    """
    for caminho in [caminho_json, caminho_csv]:
        if caminho:
            diretorio = os.path.dirname(os.path.abspath(caminho))
            os.makedirs(diretorio, exist_ok=True)

    payload = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": periodo or {
            "inicio": DATA_INICIO_TCC.isoformat(),
            "fim": DATA_FIM_TCC.isoformat(),
        },
        "metodologia": METODOLOGIA_PROXY,
        "avalia_gemini": False,
        "resultados": resultados,
        "criterios_sucesso": resumir_criterios_sucesso(resultados),
    }

    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(payload, arquivo, ensure_ascii=False, indent=2, default=str)

    caminhos = {"json": os.path.abspath(caminho_json)}

    if caminho_csv:
        linhas = []
        for res in resultados:
            linhas.append({
                "par": res.get("par"),
                "intervalo": res.get("intervalo"),
                "erro": res.get("erro"),
                "n_candles": res.get("n_candles"),
                "n_candles_validos": res.get("n_candles_validos"),
                "n_sinais_compra": res.get("n_sinais_compra"),
                "n_sinais_venda": res.get("n_sinais_venda"),
                "n_sinais_manter": res.get("n_sinais_manter"),
                "acuracia_direcional": res.get("acuracia_direcional"),
                "retorno_total_pct": res.get("retorno_total_pct"),
                "max_drawdown_pct": res.get("max_drawdown_pct"),
                "n_operacoes": res.get("n_operacoes"),
                "n_entradas": res.get("n_entradas"),
                "cobertura_solicitada_pct": res.get("cobertura_solicitada_pct"),
                "primeiro_candle": res.get("primeiro_candle"),
                "ultimo_candle": res.get("ultimo_candle"),
            })
        df = pd.DataFrame(linhas)
        df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        caminhos["csv"] = os.path.abspath(caminho_csv)

    return caminhos


def formatar_relatorio_backtesting(resultados: list) -> str:
    """
    Formata os resultados do backtesting em um relatório texto legível.

    Parâmetros:
        resultados : Lista de dicionários retornados por executar_backtesting_completo

    Retorna:
        String formatada com o relatório
    """
    linhas = [
        "=" * 80,
        "  RELATÓRIO DE BACKTESTING — CriptoIA",
        f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "=" * 80,
        "",
        "METODOLOGIA:",
        "  - Estratégia: Sinais determinísticos baseados em indicadores técnicos",
        "  - COMPRAR: MM20>MM50 AND RSI<70 AND MACD>Sinal",
        "  - VENDER:  MM20<MM50 AND RSI>30 AND MACD<Sinal",
        "  - Anti look-ahead: sinal do candle t aplicado ao retorno de t+1",
        "  - Long-only, sem custos de transação",
        "",
        "LIMITAÇÕES:",
        "  - Proxy determinístico; NÃO avalia as respostas multimodais do Gemini",
        "  - Sem custos de transação (spread, taxas, slippage)",
        "  - Estratégia long-only (não opera vendido)",
        "  - Acurácia no candle seguinte, não no prazo de tempo_estimado",
        "  - Se a coleta falhar no meio, o resultado é marcado como incompleto",
        "  - Máximo 1000 candles/requisição (paginação implementada)",
        "",
    ]

    for res in resultados:
        par = res.get("par", "?")
        tf = res.get("intervalo", "?")
        linhas.append(f"{'─' * 60}")
        linhas.append(f"  {par} | {tf}")
        linhas.append(f"{'─' * 60}")

        if "erro" in res:
            linhas.append(f"  ❌ ERRO: {res['erro']}")
        else:
            linhas.append(f"  Período      : {res.get('data_inicio_str')} -> {res.get('data_fim_str')}")
            linhas.append(f"  Candles      : {res.get('n_candles', 'N/A')} total | {res.get('n_candles_validos', 'N/A')} válidos")
            linhas.append(f"  Sinais       : ▲ Comprar={res.get('n_sinais_compra')} | "
                          f"▼ Vender={res.get('n_sinais_venda')} | "
                          f"⏸ Manter={res.get('n_sinais_manter')}")
            dist = res.get("distribuicao_acoes", {})
            if dist:
                linhas.append(
                    f"  Distribuição : {dist.get('COMPRAR_pct', 0):.1f}% compra | "
                    f"{dist.get('VENDER_pct', 0):.1f}% venda | "
                    f"{dist.get('MANTER_pct', 0):.1f}% manter"
                )
            acu = res.get("acuracia_direcional")
            acu_str = f"{acu*100:.1f}%" if acu is not None else "N/A"
            linhas.append(f"  Acurácia Dir.: {acu_str}")
            linhas.append(f"  Retorno Acum.: {res.get('retorno_total_pct', 'N/A')}%")
            linhas.append(f"  Max Drawdown : {res.get('max_drawdown_pct', 'N/A')}%")
            linhas.append(f"  Operações    : {res.get('n_operacoes', 'N/A')}")
            if res.get("cobertura_solicitada_pct") is not None:
                linhas.append(f"  Cobertura    : {res.get('cobertura_solicitada_pct')}% do período estimado")

        linhas.append("")

    criterios = resumir_criterios_sucesso(resultados)
    linhas.extend([
        "=" * 80,
        "  CRITÉRIOS DE SUCESSO (Seção IV do TCC)",
        "=" * 80,
        f"  Ativos avaliados              : {criterios['ativos_avaliados']}",
        f"  Acurácia > 50% (por ativo)    : {criterios['ativos_acuracia_acima_50']}",
        f"  Retorno não negativo (por ativo): {criterios['ativos_retorno_nao_negativo']}",
        f"  Critério de acurácia atendido : {criterios['criterio_acuracia_atendido']}",
        f"  Critério de retorno atendido  : {criterios['criterio_retorno_atendido']}",
        f"  Recorte do resumo             : {criterios['recorte_principal']}",
        f"  Nota sobre timeframe          : {criterios['nota_timeframe']}",
        f"  Observação                    : {criterios['observacao']}",
    ])

    return "\n".join(linhas)
