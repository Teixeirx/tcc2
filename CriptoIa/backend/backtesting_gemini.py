"""
backtesting_gemini.py — Módulo de Backtesting Multimodal do Google Gemini.

Avalia o modelo Google Gemini (IA multimodal) com inferência real sobre
gráficos PNG históricos e indicadores técnicos numéricos em uma amostra
sistemática do período do TCC (jan/2025–jun/2026), no timeframe diário (1d).

Principais características:
    - Amostragem sistemática periódica (ex.: a cada 7 dias) sem look-ahead bias;
    - Em cada ponto t, gera o gráfico PNG com exatamente 90 candles até t;
    - Envia imagem + indicadores ao Gemini (gemini-3.5-flash-lite / configurável);
    - Avalia a recomendação contra o futuro real:
        * Acurácia no candle t+1 (direção imediata);
        * Acurácia no horizonte de tempo estimado pelo modelo (tempo_estimado);
        * Taxa de atingimento do preço-alvo projetado (preco_alvo);
        * Retorno simulado e drawdown máximo de cada operação;
        * Buy-and-Hold calculado na mesma janela efetiva do experimento;
        * Comparação direta ponto a ponto com o proxy determinístico de indicadores;
    - Cache persistente (resultados/gemini_backtest_cache.json) com suporte
      a retomada e prevenção de chamadas duplicadas.
"""

import os
import re
import sys
import json
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# Garante saída compatível com UTF-8 no Windows
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.coleta_dados import obter_cliente_binance
from backend.analise_tecnica import enriquecer_dataframe, resumir_indicadores
from backend.gerador_grafico import gerar_grafico_candlestick
from backend.analisador_ia import analisar_mercado_com_gemini
from backend.backtesting import (
    buscar_dados_historicos_paginado,
    DATA_INICIO_TCC,
    DATA_FIM_TCC,
    PARES_TCC,
)

DIR_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultados")
ARQUIVO_CACHE_PADRAO = os.path.join(DIR_RESULTADOS, "gemini_backtest_cache.json")

# Quantidade de candles por dia em cada timeframe suportado.
CANDLES_POR_DIA = {
    "1h": 24,
    "4h": 6,
    "1d": 1,
    "1w": 1 / 7,
}


def converter_passo_para_candles(intervalo: str, passo_dias: int) -> int:
    """
    Converte um passo em dias para a quantidade equivalente de candles.

    O parâmetro --passo do CLI é expresso em dias (ex.: 7 = amostragem
    semanal). Para 1d, 7 dias = 7 candles; para 1h, 7 dias = 168 candles;
    para 4h, 7 dias = 42 candles; para 1w, 7 dias = 1 candle.
    """
    if passo_dias <= 0:
        raise ValueError("passo_amostragem deve ser maior que zero.")
    if intervalo not in CANDLES_POR_DIA:
        raise ValueError(f"Timeframe não suportado: {intervalo}")

    fator = CANDLES_POR_DIA[intervalo]
    return max(1, int(round(passo_dias * fator)))


# ── Parsers de Preço-Alvo e Horizonte Temporal ────────────────────────────────

def extrair_janela_dias(tempo_estimado: Optional[str]) -> Tuple[int, int]:
    """
    Converte a estimativa textual de tempo do Gemini em um intervalo de dias (min_dias, max_dias).
    
    Exemplos:
        '5 a 10 dias'     -> (5, 10)
        '7 a 14 dias'     -> (7, 14)
        '2 a 4 semanas'   -> (14, 28)
        '1 semana'        -> (1, 7)
        '24 a 48 horas'   -> (1, 2)
        '3-7 dias úteis'  -> (3, 7)
    """
    if not tempo_estimado or not isinstance(tempo_estimado, str):
        return 7, 14

    texto = tempo_estimado.lower().strip()

    # Caso em semanas (ex.: 2 a 4 semanas, 1-2 semanas)
    m_sem = re.search(r"(\d+)\s*(?:a|-|ate|à)\s*(\d+)\s*semana", texto)
    if m_sem:
        return int(m_sem.group(1)) * 7, int(m_sem.group(2)) * 7

    m_sem_u = re.search(r"(\d+)\s*semana", texto)
    if m_sem_u:
        s = int(m_sem_u.group(1))
        return max(1, (s - 1) * 7), s * 7

    # Caso em dias (ex.: 7 a 14 dias, 5-10 dias)
    m_dias = re.search(r"(\d+)\s*(?:a|-|ate|à)\s*(\d+)\s*dia", texto)
    if m_dias:
        return int(m_dias.group(1)), int(m_dias.group(2))

    m_dias_u = re.search(r"(\d+)\s*dia", texto)
    if m_dias_u:
        d = int(m_dias_u.group(1))
        return max(1, d // 2), d

    # Caso em horas (ex.: 24 a 48 horas)
    m_horas = re.search(r"(\d+)\s*(?:a|-|ate|à)\s*(\d+)\s*hora", texto)
    if m_horas:
        return 1, max(1, int(m_horas.group(2)) // 24)

    return 7, 14


def extrair_preco_alvo(preco_alvo_str: Optional[str], preco_referencia: float) -> Optional[float]:
    """
    Extrai e normaliza o valor numérico de preço-alvo retornado pelo Gemini.
    Suporta formatos brasileiros (98.500,00), americanos (98,500.00) e percentuais (+5%).
    """
    if not preco_alvo_str or not isinstance(preco_alvo_str, str):
        return None

    s = preco_alvo_str.replace("USDT", "").replace("usd", "").replace("R$", "").strip()

    # Percentual relativo (ex.: +5%)
    if "%" in s:
        m = re.search(r"([+-]?\d+(?:[\.,]\d+)?)%", s)
        if m:
            pct = float(m.group(1).replace(",", "."))
            return round(preco_referencia * (1 + pct / 100), 4)

    # Extrai parte numérica com separadores
    numeros = re.findall(r"[\d\.,]+", s)
    if not numeros:
        return None

    c = numeros[0].strip(".,")
    if not c:
        return None

    # Detecta formato brasileiro (ex.: 98.500,00) vs formato americano (ex.: 98,500.00)
    if "." in c and "," in c:
        if c.rfind(",") > c.rfind("."):
            c = c.replace(".", "").replace(",", ".")
        else:
            c = c.replace(",", "")
    elif "," in c:
        partes = c.split(",")
        if len(partes) == 2 and len(partes[1]) in (1, 2, 3, 4):
            c = partes[0] + "." + partes[1]
        else:
            c = c.replace(",", "")

    try:
        val = float(c)
        # Sanity check: deve estar dentro de limites plausíveis em relação ao preço atual
        if 0.1 * preco_referencia <= val <= 10.0 * preco_referencia:
            return round(val, 4)
    except (ValueError, TypeError):
        pass

    return None


# ── Gerenciamento do Cache Persistente ─────────────────────────────────────────

def carregar_cache_gemini(caminho_cache: str = ARQUIVO_CACHE_PADRAO) -> Dict[str, Any]:
    """Carrega o cache de inferências do Gemini em disco."""
    if os.path.exists(caminho_cache):
        try:
            with open(caminho_cache, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Cache] Erro ao ler cache {caminho_cache}: {e}. Criando novo cache.")
    return {}


def salvar_cache_gemini(cache: Dict[str, Any], caminho_cache: str = ARQUIVO_CACHE_PADRAO) -> None:
    """Salva atomicamente o cache de inferências no disco."""
    os.makedirs(os.path.dirname(caminho_cache), exist_ok=True)
    caminho_tmp = f"{caminho_cache}.tmp"
    with open(caminho_tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, default=str)
    os.replace(caminho_tmp, caminho_cache)


# ── Avaliador de Sinal Proxy para Comparação Direta ───────────────────────────

def calcular_buy_and_hold(
    df: pd.DataFrame,
    indice_inicio: int,
    indice_fim: int,
) -> Dict[str, Any]:
    """
    Calcula o retorno e o drawdown máximo de Buy-and-Hold na mesma janela
    efetivamente avaliada pelo backtesting Gemini (compra no fechamento do
    primeiro ponto de decisão e manutenção até o fim do último horizonte).
    """
    indice_inicio = max(0, min(int(indice_inicio), len(df) - 1))
    indice_fim = max(indice_inicio, min(int(indice_fim), len(df) - 1))

    serie = df["Fechamento"].iloc[indice_inicio : indice_fim + 1].astype(float)
    preco_inicial = float(serie.iloc[0])
    preco_final = float(serie.iloc[-1])
    retorno_pct = ((preco_final - preco_inicial) / preco_inicial) * 100.0

    picos = serie.cummax()
    drawdowns = (serie - picos) / picos
    max_drawdown_pct = float(drawdowns.min()) * 100.0

    return {
        "estrategia": "buy_and_hold",
        "preco_inicial": round(preco_inicial, 8),
        "preco_final": round(preco_final, 8),
        "retorno_total_pct": round(retorno_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "data_inicio": df.index[indice_inicio].isoformat(),
        "data_fim": df.index[indice_fim].isoformat(),
        "n_candles": len(serie),
    }


def calcular_sinal_proxy_ponto(df_passado: pd.DataFrame) -> str:
    """Calcula o sinal do proxy determinístico no fechamento do candle atual."""
    if len(df_passado) < 50:
        return "MANTER"
    ultima = df_passado.iloc[-1]
    mm20 = ultima.get("MM20")
    mm50 = ultima.get("MM50")
    rsi14 = ultima.get("RSI14")
    macd = ultima.get("MACD")
    sinal = ultima.get("Sinal_MACD")

    if pd.isna(mm20) or pd.isna(mm50) or pd.isna(rsi14) or pd.isna(macd) or pd.isna(sinal):
        return "MANTER"

    if mm20 > mm50 and rsi14 < 70 and macd > sinal:
        return "COMPRAR"
    elif mm20 < mm50 and rsi14 > 30 and macd < sinal:
        return "VENDER"
    return "MANTER"


# ── Execução de Backtesting Amostral para um Par ───────────────────────────────

def executar_backtest_gemini_par(
    par: str = "BTC/USDT",
    intervalo: str = "1d",
    passo_amostragem: int = 7,
    max_amostras: Optional[int] = None,
    delay_segundos: float = 2.0,
    caminho_cache: str = ARQUIVO_CACHE_PADRAO,
    data_inicio: Optional[datetime] = None,
    data_fim: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Executa a amostragem histórica e inferência multimodal do Gemini para um par.

    ``data_inicio`` e ``data_fim`` permitem executar recortes temporais
    exploratórios sem alterar o período oficial do TCC. Quando omitidos,
    usa-se o período completo definido em ``backtesting.py``.
    """
    data_inicio = data_inicio or DATA_INICIO_TCC
    data_fim = data_fim or DATA_FIM_TCC
    passo_candles = converter_passo_para_candles(intervalo, passo_amostragem)

    print(f"\n{'='*70}")
    print(f"  Iniciando Backtesting Multimodal Gemini: {par} ({intervalo})")
    print(f"  Período: {data_inicio.strftime('%Y-%m-%d')} a {data_fim.strftime('%Y-%m-%d')}")
    print(f"  Amostragem: a cada {passo_amostragem} dias (~{passo_candles} candles) | Delay: {delay_segundos:.1f}s")
    print(f"{'='*70}")

    df = buscar_dados_historicos_paginado(par, intervalo, data_inicio, data_fim)
    if df.empty or len(df) < 110:
        return {"erro": f"Dados insuficientes para {par} ({len(df)} candles)."}

    # Carrega cache
    cache = carregar_cache_gemini(caminho_cache)

    # Pontos de amostragem: precisamos de 90 candles para trás e pelo menos 14 candles para frente
    inicio_amostragem = 90
    fim_amostragem = len(df) - 15

    indices = list(range(inicio_amostragem, fim_amostragem, passo_candles))
    if max_amostras and len(indices) > max_amostras:
        indices = indices[:max_amostras]

    print(f"[Amostragem] {len(indices)} pontos de decisão selecionados.")

    registros = []
    amostras_novas = 0
    amostras_em_cache = 0

    for i, t in enumerate(indices, start=1):
        data_decisao = df.index[t]
        data_str = data_decisao.strftime("%Y-%m-%d")
        chave_cache = f"{par}_{intervalo}_{data_str}"

        preco_entrada = float(df["Fechamento"].iloc[t])

        # Verifica se já está em cache
        if chave_cache in cache:
            dados_ponto = cache[chave_cache]
            amostras_em_cache += 1
            print(f"  [{i}/{len(indices)}] {data_str} — Carregado do cache ({dados_ponto['resposta_ia']['acao']})")
        else:
            # Janela passada (estritamente até t, sem look-ahead bias)
            df_passado = df.iloc[t - 89 : t + 1].copy()
            df_enriquecido = enriquecer_dataframe(df_passado)
            indicadores = resumir_indicadores(df_enriquecido)

            # Gera gráfico temporário
            clean_par = par.replace("/", "_")
            nome_grafico = f"temp_gemini_{clean_par}_{data_str}.png"
            caminho_grafico = gerar_grafico_candlestick(
                df_enriquecido,
                nome_arquivo=nome_grafico,
                titulo=f"{par} — {data_str} ({intervalo})"
            )

            print(f"  [{i}/{len(indices)}] {data_str} — Chamando Gemini para {par} (${preco_entrada:,.2f})...")
            
            # Chamada ao Gemini com retry se ocorrer fallback ou exceção
            resposta_ia = None
            for tentativa in range(1, 4):
                try:
                    resposta_ia = analisar_mercado_com_gemini(
                        caminho_grafico=caminho_grafico,
                        indicadores=indicadores,
                        par=par,
                    )
                    if resposta_ia and not resposta_ia.get("fallback"):
                        break
                    motivo = resposta_ia.get("motivo_fallback", "") if resposta_ia else "vazio"
                    print(f"    [Aviso] Fallback retornado (tentativa {tentativa}/3): {motivo[:60]}...")
                    time.sleep(3 * tentativa)
                except Exception as erro:
                    print(f"    [Aviso] Falha na chamada (tentativa {tentativa}/3): {erro}")
                    time.sleep(3 * tentativa)

            # Limpa imagem temporária
            if os.path.exists(caminho_grafico):
                try:
                    os.remove(caminho_grafico)
                except Exception:
                    pass

            if not resposta_ia:
                resposta_ia = {
                    "acao": "MANTER",
                    "justificativa": "Falha de comunicação persistente com a API.",
                    "preco_alvo": None,
                    "tempo_estimado": None,
                    "fallback": True,
                }

            # Sinal do proxy determinístico no mesmo candle para comparação
            sinal_proxy = calcular_sinal_proxy_ponto(df_enriquecido)

            # Avaliação futura
            min_dias, max_dias = extrair_janela_dias(resposta_ia.get("tempo_estimado"))
            preco_alvo = extrair_preco_alvo(resposta_ia.get("preco_alvo"), preco_entrada)

            # Fatia futura real
            fim_futuro = min(len(df), t + 1 + max_dias)
            df_futuro = df.iloc[t + 1 : fim_futuro]

            preco_t1 = float(df["Fechamento"].iloc[t + 1])
            retorno_t1_pct = ((preco_t1 - preco_entrada) / preco_entrada) * 100.0

            acao = resposta_ia.get("acao", "MANTER").upper()

            acertou_t1 = None
            acertou_prazo = None
            atingiu_alvo = False
            dias_ate_alvo = None
            retorno_trade_pct = 0.0
            max_dd_trade_pct = 0.0

            if acao == "COMPRAR":
                acertou_t1 = bool(preco_t1 > preco_entrada)
                if not df_futuro.empty:
                    max_futura = float(df_futuro["Maxima"].max())
                    min_futura = float(df_futuro["Minima"].min())
                    fechamento_fim = float(df_futuro["Fechamento"].iloc[-1])

                    max_dd_trade_pct = ((min_futura - preco_entrada) / preco_entrada) * 100.0

                    if preco_alvo and max_futura >= preco_alvo:
                        atingiu_alvo = True
                        acertou_prazo = True
                        # Encontra dia em que bateu o alvo
                        idx_hit = df_futuro[df_futuro["Maxima"] >= preco_alvo].index
                        dias_ate_alvo = int((idx_hit[0] - data_decisao).total_seconds() / 86400)
                        retorno_trade_pct = ((preco_alvo - preco_entrada) / preco_entrada) * 100.0
                    else:
                        acertou_prazo = bool(fechamento_fim > preco_entrada)
                        retorno_trade_pct = ((fechamento_fim - preco_entrada) / preco_entrada) * 100.0

            elif acao == "VENDER":
                acertou_t1 = bool(preco_t1 < preco_entrada)
                if not df_futuro.empty:
                    min_futura = float(df_futuro["Minima"].min())
                    max_futura = float(df_futuro["Maxima"].max())
                    fechamento_fim = float(df_futuro["Fechamento"].iloc[-1])

                    max_dd_trade_pct = ((preco_entrada - max_futura) / preco_entrada) * 100.0

                    if preco_alvo and min_futura <= preco_alvo:
                        atingiu_alvo = True
                        acertou_prazo = True
                        idx_hit = df_futuro[df_futuro["Minima"] <= preco_alvo].index
                        dias_ate_alvo = int((idx_hit[0] - data_decisao).total_seconds() / 86400)
                    else:
                        acertou_prazo = bool(fechamento_fim < preco_entrada)

                # Long-only fica em caixa (retorno 0)
                retorno_trade_pct = 0.0

            else:  # MANTER
                retorno_trade_pct = 0.0

            dados_ponto = {
                "chave": chave_cache,
                "par": par,
                "intervalo": intervalo,
                "data_decisao": data_str,
                "preco_fechamento": preco_entrada,
                "resposta_ia": resposta_ia,
                "sinal_proxy": sinal_proxy,
                "concordou_com_proxy": bool(acao == sinal_proxy),
                "avaliacao": {
                    "preco_entrada": preco_entrada,
                    "preco_t1": preco_t1,
                    "retorno_t1_pct": round(retorno_t1_pct, 2),
                    "acertou_t1": acertou_t1,
                    "janela_dias_estimada": [min_dias, max_dias],
                    "preco_alvo_normalizado": preco_alvo,
                    "atingiu_alvo": atingiu_alvo,
                    "dias_ate_alvo": dias_ate_alvo,
                    "acertou_prazo": acertou_prazo,
                    "retorno_trade_pct": round(retorno_trade_pct, 2),
                    "max_dd_trade_pct": round(max_dd_trade_pct, 2),
                }
            }

            cache[chave_cache] = dados_ponto
            salvar_cache_gemini(cache, caminho_cache)
            amostras_novas += 1

            # Pausa educada para respeitar limites de taxa
            time.sleep(delay_segundos)

        registros.append(dados_ponto)

    # ── Métricas Consolidadas do Par ──────────────────────────────────────────
    total_pontos = len(registros)
    comprar = [r for r in registros if r["resposta_ia"]["acao"] == "COMPRAR"]
    vender = [r for r in registros if r["resposta_ia"]["acao"] == "VENDER"]
    manter = [r for r in registros if r["resposta_ia"]["acao"] == "MANTER"]

    direcionais = comprar + vender

    # Acurácia t+1 (direção no candle seguinte)
    acertos_t1 = [r for r in direcionais if r["avaliacao"]["acertou_t1"] is True]
    acuracia_t1 = (len(acertos_t1) / len(direcionais) * 100.0) if direcionais else None

    # Acurácia no prazo estimado pelo Gemini
    acertos_prazo = [r for r in direcionais if r["avaliacao"]["acertou_prazo"] is True]
    acuracia_prazo = (len(acertos_prazo) / len(direcionais) * 100.0) if direcionais else None

    # Taxa de atingimento do alvo (apenas compras com alvo explícito)
    compras_com_alvo = [r for r in comprar if r["avaliacao"]["preco_alvo_normalizado"] is not None]
    alvos_atingidos = [r for r in compras_com_alvo if r["avaliacao"]["atingiu_alvo"] is True]
    taxa_alvo_compras = (len(alvos_atingidos) / len(compras_com_alvo) * 100.0) if compras_com_alvo else None

    # Retorno acumulado composto da estratégia (long-only).
    # Observação: como a amostragem é semanal e o horizonte pode chegar a
    # 14 dias, os retornos de decisões consecutivas podem se sobrepor.
    capital = 1.0
    for r in registros:
        ret = r["avaliacao"]["retorno_trade_pct"] / 100.0
        capital *= (1.0 + ret)
    retorno_acumulado_pct = (capital - 1.0) * 100.0

    # Buy-and-Hold na mesma janela efetiva do experimento.
    buy_and_hold = None
    if registros:
        ultimo_horizonte = max(
            r["avaliacao"]["janela_dias_estimada"][1] for r in registros
        )
        indice_fim_avaliacao = min(len(df) - 1, indices[-1] + 1 + ultimo_horizonte)
        buy_and_hold = calcular_buy_and_hold(df, indices[0], indice_fim_avaliacao)

    # Concordância com o Proxy
    concordancias = [r for r in registros if r.get("concordou_com_proxy") is True]
    taxa_concordancia = (len(concordancias) / total_pontos * 100.0) if total_pontos else 0.0

    resultado = {
        "par": par,
        "intervalo": intervalo,
        "periodo": {
            "inicio": data_inicio.isoformat(),
            "fim": data_fim.isoformat(),
        },
        "passo_amostragem_dias": passo_amostragem,
        "passo_amostragem_candles": passo_candles,
        "total_amostras": total_pontos,
        "amostras_novas": amostras_novas,
        "amostras_em_cache": amostras_em_cache,
        "retorno_metodologia": (
            "Retorno composto por decisão amostral; horizontes de 7-14 dias "
            "podem se sobrepor. Buy-and-Hold calculado na mesma janela efetiva."
        ),
        "buy_and_hold": buy_and_hold,
        "distribuicao_acoes": {
            "COMPRAR": len(comprar),
            "VENDER": len(vender),
            "MANTER": len(manter),
            "COMPRAR_pct": round(len(comprar) / total_pontos * 100.0, 1) if total_pontos else 0.0,
            "VENDER_pct": round(len(vender) / total_pontos * 100.0, 1) if total_pontos else 0.0,
            "MANTER_pct": round(len(manter) / total_pontos * 100.0, 1) if total_pontos else 0.0,
        },
        "acuracia_t1_pct": round(acuracia_t1, 1) if acuracia_t1 is not None else None,
        "acuracia_prazo_estimado_pct": round(acuracia_prazo, 1) if acuracia_prazo is not None else None,
        "taxa_atingimento_alvo_pct": round(taxa_alvo_compras, 1) if taxa_alvo_compras is not None else None,
        "retorno_acumulado_pct": round(retorno_acumulado_pct, 2),
        "taxa_concordancia_proxy_pct": round(taxa_concordancia, 1),
        "registros": registros,
    }

    print(f"\n[Resultado Par {par}]")
    print(f"  Acurácia Prazo Estimado: {resultado['acuracia_prazo_estimado_pct']}% "
          f"({len(acertos_prazo)}/{len(direcionais)})")
    print(f"  Acurácia Candle t+1:     {resultado['acuracia_t1_pct']}%")
    print(f"  Taxa Atingimento Alvo:   {resultado['taxa_atingimento_alvo_pct']}%")
    print(f"  Retorno Acumulado:       {resultado['retorno_acumulado_pct']:+.2f}%")
    if buy_and_hold:
        print(f"  Buy-and-Hold (mesma janela): {buy_and_hold['retorno_total_pct']:+.2f}%")
    print(f"  Concordância com Proxy:  {resultado['taxa_concordancia_proxy_pct']}%")

    return resultado


# ── Execução em Grade Completa (4 Ativos) ──────────────────────────────────────

def executar_backtest_gemini_grade(
    pares: Optional[List[str]] = None,
    passo_amostragem: int = 7,
    max_amostras_por_par: Optional[int] = None,
    delay_segundos: float = 2.0,
    caminho_cache: str = ARQUIVO_CACHE_PADRAO,
    intervalos: Optional[List[str]] = None,
    data_inicio: Optional[datetime] = None,
    data_fim: Optional[datetime] = None,
    prefixo_saida: Optional[str] = None,
    escopo: Optional[str] = None,
    observacao: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa o backtesting multimodal do Gemini em um ou mais timeframes.

    Por padrão, reproduz o Experimento 2 oficial: 4 pares, timeframe 1d,
    período integral do TCC e saída em ``backtesting_gemini_tcc``.

    Para o complemento exploratório intraday, informe ``intervalos=["1h", "4h"]``,
    um período menor em ``data_inicio``/``data_fim`` e um ``prefixo_saida``
    próprio, evitando sobrescrever os resultados oficiais.
    """
    if pares is None:
        pares = PARES_TCC
    if intervalos is None:
        intervalos = ["1d"]

    data_inicio = data_inicio or DATA_INICIO_TCC
    data_fim = data_fim or DATA_FIM_TCC

    periodo_oficial = (
        data_inicio.isoformat() == DATA_INICIO_TCC.isoformat()
        and data_fim.isoformat() == DATA_FIM_TCC.isoformat()
    )
    if escopo is None:
        escopo = "oficial" if periodo_oficial and intervalos == ["1d"] else "exploratorio"
    if prefixo_saida is None:
        prefixo_saida = (
            "backtesting_gemini_tcc"
            if escopo == "oficial"
            else "backtesting_gemini_exploratorio"
        )
    if observacao is None:
        if escopo == "oficial":
            observacao = (
                "Inferência real restrita ao 1d por quota do modelo gratuito do Gemini; "
                "a grade 4x4 usa o proxy determinístico."
            )
        else:
            observacao = (
                "Experimento exploratório com período reduzido; complementa, mas não "
                "substitui o Experimento 2 oficial de 18 meses em 1d."
            )

    resultados_pares = []
    tempo_inicio = time.time()

    for intervalo in intervalos:
        for par in pares:
            res = executar_backtest_gemini_par(
                par=par,
                intervalo=intervalo,
                passo_amostragem=passo_amostragem,
                max_amostras=max_amostras_por_par,
                delay_segundos=delay_segundos,
                caminho_cache=caminho_cache,
                data_inicio=data_inicio,
                data_fim=data_fim,
            )
            resultados_pares.append(res)

    tempo_total = time.time() - tempo_inicio
    combinacoes_avaliadas = len(resultados_pares)
    pares_unicos = sorted({r["par"] for r in resultados_pares if "par" in r})

    # Consolidação Geral
    relatorio_geral = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "escopo": escopo,
        "periodo": {
            "inicio": data_inicio.isoformat(),
            "fim": data_fim.isoformat(),
        },
        "intervalos": intervalos,
        "metodologia": (
            "inferencia_real_gemini_multimodal_exploratoria"
            if escopo == "exploratorio"
            else "inferencia_real_gemini_multimodal"
        ),
        "benchmark": "buy_and_hold_na_mesma_janela_efetiva",
        "retorno_metodologia": (
            "Retorno composto por decisão amostral; horizontes podem se sobrepor."
        ),
        "observacao_quota": observacao,
        "modelo": os.getenv("MODELO_GEMINI", "gemini-3.5-flash-lite"),
        "passo_amostragem_dias": passo_amostragem,
        "tempo_execucao_segundos": round(tempo_total, 1),
        "pares_avaliados": len(pares_unicos),
        "combinacoes_avaliadas": combinacoes_avaliadas,
        "resultados": resultados_pares,
    }

    # Salva JSON completo
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    caminho_json = os.path.join(DIR_RESULTADOS, f"{prefixo_saida}.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(relatorio_geral, f, ensure_ascii=False, indent=2, default=str)

    # Salva CSV resumido. Em execuções multi-timeframe, inclui a coluna TF.
    caminho_csv = os.path.join(DIR_RESULTADOS, f"{prefixo_saida}.csv")
    incluir_intervalo = len(intervalos) > 1 or intervalos != ["1d"]
    cabecalho_csv = (
        ("Intervalo," if incluir_intervalo else "")
        + "Par,Amostras,Comprar,Vender,Manter,Acuracia_t1_pct,"
        + "Acuracia_Prazo_pct,Taxa_Alvo_pct,Retorno_Acumulado_pct,"
        + "Buy_and_Hold_pct,Concordancia_Proxy_pct\n"
    )
    linhas_csv = [cabecalho_csv]
    for r in resultados_pares:
        if "erro" not in r:
            dist = r["distribuicao_acoes"]
            bh = r.get("buy_and_hold") or {}
            prefixo_linha = f"{r['intervalo']}," if incluir_intervalo else ""
            linhas_csv.append(
                f"{prefixo_linha}{r['par']},{r['total_amostras']},{dist['COMPRAR']},{dist['VENDER']},{dist['MANTER']},"
                f"{r['acuracia_t1_pct']},{r['acuracia_prazo_estimado_pct']},{r['taxa_atingimento_alvo_pct']},"
                f"{r['retorno_acumulado_pct']},{bh.get('retorno_total_pct', '')},{r['taxa_concordancia_proxy_pct']}\n"
            )
    with open(caminho_csv, "w", encoding="utf-8") as f:
        f.writelines(linhas_csv)

    # Salva Relatório Textual Resumo
    nome_resumo = (
        "backtesting_gemini_resumo.txt"
        if prefixo_saida == "backtesting_gemini_tcc"
        else f"{prefixo_saida}_resumo.txt"
    )
    caminho_resumo = os.path.join(DIR_RESULTADOS, nome_resumo)
    intervalos_txt = ", ".join(intervalos)
    with open(caminho_resumo, "w", encoding="utf-8") as f:
        f.write("RESUMO DO BACKTESTING MULTIMODAL REAL — GOOGLE GEMINI (TCC II)\n")
        f.write("=" * 105 + "\n")
        f.write(
            f"Modelo: {relatorio_geral['modelo']} | "
            f"Período: {data_inicio.strftime('%Y-%m-%d')} a {data_fim.strftime('%Y-%m-%d')} | "
            f"Timeframe(s): {intervalos_txt} | Escopo: {escopo}\n"
        )
        f.write("-" * 105 + "\n")
        if incluir_intervalo:
            f.write(f"{'TF':<4} {'Par':<10} {'Amostras':>8} {'Comp':>5} {'Vend':>5} {'Mant':>5} {'Acur.t+1%':>10} {'Acur.Prazo%':>12} {'Alvo.Hit%':>10} {'Retorno%':>10} {'B&H%':>9} {'Proxy.Conc%':>12}\n")
        else:
            f.write(f"{'Par':<10} {'Amostras':>8} {'Comp':>5} {'Vend':>5} {'Mant':>5} {'Acur.t+1%':>10} {'Acur.Prazo%':>12} {'Alvo.Hit%':>10} {'Retorno%':>10} {'B&H%':>9} {'Proxy.Conc%':>12}\n")
        f.write("-" * 105 + "\n")
        for r in resultados_pares:
            if "erro" not in r:
                d = r["distribuicao_acoes"]
                ac_t1 = f"{r['acuracia_t1_pct']:.1f}%" if r['acuracia_t1_pct'] is not None else "N/A"
                ac_pz = f"{r['acuracia_prazo_estimado_pct']:.1f}%" if r['acuracia_prazo_estimado_pct'] is not None else "N/A"
                hit = f"{r['taxa_atingimento_alvo_pct']:.1f}%" if r['taxa_atingimento_alvo_pct'] is not None else "N/A"
                ret = f"{r['retorno_acumulado_pct']:+.2f}%"
                bh = r.get("buy_and_hold") or {}
                bh_txt = f"{bh['retorno_total_pct']:+.2f}%" if bh.get("retorno_total_pct") is not None else "N/A"
                conc = f"{r['taxa_concordancia_proxy_pct']:.1f}%"
                prefixo_tf = f"{r['intervalo']:<4} " if incluir_intervalo else ""
                f.write(f"{prefixo_tf}{r['par']:<10} {r['total_amostras']:>8} {d['COMPRAR']:>5} {d['VENDER']:>5} {d['MANTER']:>5} {ac_t1:>10} {ac_pz:>12} {hit:>10} {ret:>10} {bh_txt:>9} {conc:>12}\n")
        f.write("=" * 105 + "\n")
        f.write("Nota: B&H = Buy-and-Hold na mesma janela efetiva do experimento. Retorno Gemini composto por decisao amostral; horizontes podem se sobrepor.\n")
        f.write(f"Nota: {observacao}\n")

    print(f"\n[Backtesting Gemini Concluído com Sucesso!]")
    print(f"  JSON completo: {caminho_json}")
    print(f"  CSV exportado: {caminho_csv}")
    print(f"  Resumo TXT:   {caminho_resumo}")

    return relatorio_geral


if __name__ == "__main__":
    executar_backtest_gemini_grade()
