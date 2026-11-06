"""
gerador_grafico.py — Módulo de geração de gráfico de candlestick.

Gera um gráfico técnico completo com candlesticks, médias móveis,
volume e Bandas de Bollinger usando mplfinance e matplotlib.
O gráfico é salvo como PNG para ser enviado ao modelo Gemini.
"""

import os

import matplotlib
matplotlib.use("Agg")  # Backend sem interface gráfica (headless)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplfinance as mpf
import pandas as pd


# Paleta de cores do tema escuro
COR_FUNDO = "#0d1117"
COR_FUNDO_GRAFICO = "#161b22"
COR_GRADE = "#30363d"
COR_TEXTO = "#c9d1d9"
COR_ALTA = "#3fb950"     # Verde para candles de alta
COR_BAIXA = "#f85149"    # Vermelho para candles de baixa
COR_MM20 = "#f0883e"     # Laranja para MM20
COR_MM50 = "#79c0ff"     # Azul claro para MM50
COR_BOLLINGER = "#8b949e" # Cinza para Bandas de Bollinger
COR_VOLUME = "#388bfd"   # Azul para volume
COR_RSI = "#d2a8ff"      # Roxo para RSI

# Diretório raiz do projeto (CriptoIa/)
_DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gerar_grafico_candlestick(
    df: pd.DataFrame,
    nome_arquivo: str = os.path.join(_DIR_RAIZ, "resultados", "grafico_bitcoin.png"),
    titulo: str = "Bitcoin (BTC/USDT) — Análise Técnica"
) -> str:
    """
    Gera um gráfico técnico completo com:
        - Candlesticks com cores de alta/baixa
        - Médias Móveis (MM20, MM50)
        - Bandas de Bollinger
        - Volume
        - RSI (painel separado abaixo)

    Parâmetros:
        df           : DataFrame com colunas OHLCV + indicadores (de analise_tecnica.py)
        nome_arquivo : Caminho do arquivo PNG de saída
        titulo       : Título exibido no gráfico

    Retorna:
        Caminho do arquivo gerado
    """
    print(f"[Gráfico] Gerando gráfico técnico: {nome_arquivo}...")

    # Renomeia colunas para o formato esperado pelo mplfinance
    df_mpf = df[["Abertura", "Maxima", "Minima", "Fechamento", "Volume"]].copy()
    df_mpf.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Cores dos candles
    cores_mercado = mpf.make_marketcolors(
        up=COR_ALTA,
        down=COR_BAIXA,
        edge="inherit",
        wick="inherit",
        volume={"up": COR_ALTA, "down": COR_BAIXA}
    )

    estilo = mpf.make_mpf_style(
        marketcolors=cores_mercado,
        facecolor=COR_FUNDO_GRAFICO,
        edgecolor=COR_GRADE,
        figcolor=COR_FUNDO,
        gridcolor=COR_GRADE,
        gridstyle="--",
        gridaxis="both",
        y_on_right=False,
        rc={
            "axes.labelcolor": COR_TEXTO,
            "xtick.color": COR_TEXTO,
            "ytick.color": COR_TEXTO,
            "text.color": COR_TEXTO,
            "font.family": "DejaVu Sans",
        }
    )

    # Plota adicionais: MM20, MM50 e Bandas de Bollinger
    plots_adicionais = []

    if "MM20" in df.columns and df["MM20"].notna().any():
        plots_adicionais.append(
            mpf.make_addplot(df["MM20"].rename("MM20"), color=COR_MM20, width=1.5)
        )

    if "MM50" in df.columns and df["MM50"].notna().any():
        plots_adicionais.append(
            mpf.make_addplot(df["MM50"].rename("MM50"), color=COR_MM50, width=1.5)
        )

    if "Bollinger_Superior" in df.columns and df["Bollinger_Superior"].notna().any():
        plots_adicionais.append(
            mpf.make_addplot(
                df["Bollinger_Superior"], color=COR_BOLLINGER,
                width=0.8, linestyle="--", alpha=0.7
            )
        )
        plots_adicionais.append(
            mpf.make_addplot(
                df["Bollinger_Inferior"], color=COR_BOLLINGER,
                width=0.8, linestyle="--", alpha=0.7
            )
        )

    # RSI no painel inferior
    if "RSI14" in df.columns and df["RSI14"].notna().any():
        plots_adicionais.append(
            mpf.make_addplot(
                df["RSI14"], panel=2, color=COR_RSI,
                width=1.5, ylabel="RSI"
            )
        )

    # Gera o gráfico
    figura, paineis = mpf.plot(
        df_mpf,
        type="candle",
        style=estilo,
        title=f"\n{titulo}",
        volume=True,
        addplot=plots_adicionais if plots_adicionais else None,
        panel_ratios=(4, 1.5, 1.5) if "RSI14" in df.columns else (4, 1.5),
        figsize=(14, 10),
        returnfig=True,
    )

    # Adiciona legenda ao painel principal
    painel_principal = paineis[0]
    linhas_legenda = []
    rotulos_legenda = []

    for linha in painel_principal.get_lines():
        rotulos_legenda.append(linha.get_label())
        linhas_legenda.append(linha)

    # Legenda manual
    from matplotlib.lines import Line2D
    elementos_legenda = [
        Line2D([0], [0], color=COR_MM20, linewidth=1.5, label="MM 20"),
        Line2D([0], [0], color=COR_MM50, linewidth=1.5, label="MM 50"),
        Line2D([0], [0], color=COR_BOLLINGER, linewidth=0.8,
               linestyle="--", label="Bollinger"),
    ]
    painel_principal.legend(
        handles=elementos_legenda,
        loc="upper left",
        facecolor=COR_FUNDO,
        edgecolor=COR_GRADE,
        labelcolor=COR_TEXTO,
        fontsize=9,
        framealpha=0.8,
    )

    # Linhas de referência do RSI (30 e 70)
    if len(paineis) >= 5:
        painel_rsi = paineis[4]
        painel_rsi.axhline(70, color=COR_BAIXA, linewidth=0.8, linestyle="--", alpha=0.7)
        painel_rsi.axhline(30, color=COR_ALTA, linewidth=0.8, linestyle="--", alpha=0.7)
        painel_rsi.set_ylim(0, 100)

    figura.savefig(nome_arquivo, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close(figura)

    print(f"[Gráfico] Salvo em: {nome_arquivo}")
    return nome_arquivo
