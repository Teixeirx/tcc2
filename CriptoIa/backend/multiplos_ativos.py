"""
multiplos_ativos.py — Módulo de análise paralela de múltiplos pares.

Permite executar análises simultâneas de múltiplos pares e timeframes
usando ThreadPoolExecutor, gerando um relatório comparativo consolidado.

Funcionalidades:
    - Análise paralela de N pares × M timeframes
    - Tratamento individual de erros (falha em um par não interrompe os demais)
    - Relatório comparativo em CSV, JSON e texto

Nota: O GIL do Python limita o paralelismo real em operações CPU-bound,
mas para operações I/O-bound (chamadas de API) ThreadPoolExecutor é eficiente.
"""

import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd

# Adiciona o diretório raiz ao path para importações
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Diretório raiz do projeto (CriptoIa/)
_DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from backend.coleta_dados import buscar_dados_historicos, buscar_preco_atual
from backend.analise_tecnica import enriquecer_dataframe, resumir_indicadores
from backend.gerador_grafico import gerar_grafico_candlestick
from backend.analisador_ia import analisar_mercado_com_gemini


# ── Análise de um único par ────────────────────────────────────────────────────

def analisar_par(
    par: str,
    intervalo: str = "1d",
    quantidade: int = 90,
    gerar_grafico: bool = True,
    usar_ia: bool = True,
) -> dict:
    """
    Executa a análise completa (sem dashboard) para um único par.

    Parâmetros:
        par          : Par de negociação (ex: 'BTC/USDT')
        intervalo    : Timeframe (ex: '1d', '4h', '1h', '1w')
        quantidade   : Número de candles a buscar
        gerar_grafico: Se True, gera o gráfico PNG (necessário para IA)
        usar_ia      : Se True, chama o Gemini para análise

    Retorna:
        Dicionário com: par, intervalo, indicadores, analise_ia, erro (se houver)
    """
    resultado = {
        "par": par,
        "intervalo": intervalo,
        "quantidade": quantidade,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Coleta e análise técnica
        df = buscar_dados_historicos(par, intervalo, quantidade)
        ticker = buscar_preco_atual(par)
        df_enriquecido = enriquecer_dataframe(df)
        indicadores = resumir_indicadores(df_enriquecido)

        resultado["indicadores"] = indicadores
        resultado["ticker"] = ticker
        resultado["n_candles"] = len(df)

        # Gráfico e análise IA
        if gerar_grafico:
            nome_arquivo = f"grafico_{par.replace('/', '_')}_{intervalo}.png"
            caminho_grafico = gerar_grafico_candlestick(
                df_enriquecido,
                nome_arquivo=nome_arquivo,
                titulo=f"{par} — Análise Técnica ({intervalo})"
            )
            resultado["caminho_grafico"] = caminho_grafico

            if usar_ia:
                analise_ia = analisar_mercado_com_gemini(
                    caminho_grafico, indicadores, par=par
                )
                resultado["analise_ia"] = analise_ia

    except Exception as erro:
        resultado["erro"] = f"{type(erro).__name__}: {erro}"
        resultado["analise_ia"] = {
            "acao": "MANTER",
            "justificativa": f"Falha ao analisar {par} ({intervalo}): {erro}",
            "preco_alvo": None,
            "tempo_estimado": None,
            "fallback": True,
            "motivo_fallback": str(erro),
        }

    return resultado


# ── Análise paralela ───────────────────────────────────────────────────────────

def analisar_multiplos(
    combinacoes: list,
    max_workers: int = 3,
    gerar_graficos: bool = True,
    usar_ia: bool = True,
) -> list:
    """
    Executa análises em paralelo para múltiplos pares e timeframes.

    Parâmetros:
        combinacoes  : Lista de dicts com chaves 'par', 'intervalo', 'quantidade'
                       Ex: [{"par": "BTC/USDT", "intervalo": "1d", "quantidade": 90}]
        max_workers  : Número máximo de threads paralelas (padrão: 3)
        gerar_graficos: Se True, gera gráficos PNG para cada análise
        usar_ia      : Se True, chama o Gemini para cada par

    Retorna:
        Lista de dicionários de resultados (um por combinação)
        Erros individuais são capturados e incluídos no resultado,
        não interrompendo as demais análises.
    """
    if not combinacoes:
        return []

    print(f"\n[MultiplosAtivos] Iniciando {len(combinacoes)} análises com {max_workers} workers...")
    resultados = [None] * len(combinacoes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(
                analisar_par,
                comb["par"],
                comb.get("intervalo", "1d"),
                comb.get("quantidade", 90),
                gerar_graficos,
                usar_ia,
            ): (indice, comb)
            for indice, comb in enumerate(combinacoes)
        }

        for futuro in as_completed(futuros):
            indice, comb = futuros[futuro]
            try:
                res = futuro.result(timeout=120)
                resultados[indice] = res
                status = "✅" if "erro" not in res else "❌"
                print(f"  {status} {res.get('par')} {res.get('intervalo')}: "
                      f"{res.get('analise_ia', {}).get('acao', 'N/A')}")
            except Exception as erro:
                print(f"  [TIMEOUT/ERRO] {comb}: {erro}")
                resultados[indice] = {
                    "par": comb.get("par", "?"),
                    "intervalo": comb.get("intervalo", "?"),
                    "erro": f"{type(erro).__name__}: {erro}",
                    "analise_ia": {
                        "acao": "MANTER",
                        "justificativa": f"Falha na análise paralela: {erro}",
                        "preco_alvo": None,
                        "tempo_estimado": None,
                        "fallback": True,
                        "motivo_fallback": str(erro),
                    },
                }

    print(f"\n[MultiplosAtivos] Concluído: {len(resultados)}/{len(combinacoes)} análises.")
    return resultados


# ── Relatório consolidado ──────────────────────────────────────────────────────

def gerar_relatorio_consolidado(resultados: list) -> dict:
    """
    Gera um relatório comparativo consolidado dos resultados.

    Parâmetros:
        resultados : Lista de dicionários retornados por analisar_multiplos

    Retorna:
        Dicionário com:
            - resumo_texto : Relatório em texto legível
            - dataframe    : pandas DataFrame com resultados tabulados
            - json_dados   : JSON com todos os resultados
    """
    linhas_tabela = []

    for res in resultados:
        linha = {
            "Par": res.get("par", "?"),
            "Timeframe": res.get("intervalo", "?"),
            "Ação_IA": res.get("analise_ia", {}).get("acao", "MANTER"),
            "RSI14": res.get("indicadores", {}).get("rsi14", "N/A"),
            "MACD": res.get("indicadores", {}).get("macd", "N/A"),
            "MM20": res.get("indicadores", {}).get("mm20", "N/A"),
            "MM50": res.get("indicadores", {}).get("mm50", "N/A"),
            "Preço": res.get("ticker", {}).get("preco", "N/A"),
            "Variação_24h(%)": res.get("ticker", {}).get("variacao_24h", "N/A"),
            "Erro": res.get("erro", ""),
        }
        linhas_tabela.append(linha)

    df_tabela = pd.DataFrame(linhas_tabela)

    # Texto resumido
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    linhas_txt = [
        "=" * 70,
        f"  RELATÓRIO CONSOLIDADO — Múltiplos Ativos — {timestamp}",
        "=" * 70,
        "",
    ]

    for res in resultados:
        par = res.get("par", "?")
        tf = res.get("intervalo", "?")
        acao = res.get("analise_ia", {}).get("acao", "N/A")
        preco_alvo = res.get("analise_ia", {}).get("preco_alvo", "N/A")

        if "erro" in res:
            linhas_txt.append(f"  ❌ {par} ({tf}): ERRO — {res['erro']}")
        else:
            rsi = res.get("indicadores", {}).get("rsi14", "N/A")
            preco = res.get("ticker", {}).get("preco", "N/A")
            preco_str = f"${preco:,.4f}" if isinstance(preco, (int, float)) else str(preco)
            alvo_str = f" | Alvo: {preco_alvo}" if preco_alvo else ""
            linhas_txt.append(
                f"  {'▲' if acao == 'COMPRAR' else '▼' if acao == 'VENDER' else '⏸'} "
                f"{par} ({tf}): {acao} | RSI={rsi} | Preço={preco_str}{alvo_str}"
            )

    linhas_txt.append("")
    linhas_txt.append(f"  Total analisado: {len(resultados)} combinações")
    linhas_txt.append(
        f"  Com erro: {sum(1 for r in resultados if 'erro' in r)}"
    )

    return {
        "resumo_texto": "\n".join(linhas_txt),
        "dataframe": df_tabela,
        "json_dados": json.dumps(resultados, ensure_ascii=False, default=str, indent=2),
        "timestamp": timestamp,
    }


def salvar_relatorio_csv(resultados: list, caminho: str = os.path.join(_DIR_RAIZ, "resultados", "relatorio_multiplos.csv")) -> str:
    """
    Salva os resultados em arquivo CSV.

    Parâmetros:
        resultados : Lista de dicionários retornados por analisar_multiplos
        caminho    : Caminho do arquivo CSV de saída

    Retorna:
        Caminho absoluto do arquivo gerado
    """
    relatorio = gerar_relatorio_consolidado(resultados)
    df = relatorio["dataframe"]
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"[MultiplosAtivos] Relatório CSV salvo em: {caminho}")
    return os.path.abspath(caminho)
