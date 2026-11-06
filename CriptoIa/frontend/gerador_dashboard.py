"""
gerador_dashboard.py — Módulo de geração do dashboard web interativo.

Gera um arquivo HTML completo com:
  - Preço atual do Bitcoin com variação 24h
  - Gráfico interativo de candlestick com Plotly (usando dados do Pandas)
  - Indicadores técnicos em cards
  - Recomendação da IA com justificativa, preço-alvo e tempo estimado
  - Design premium com tema escuro e animações
"""

import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Diretório raiz do projeto (dois níveis acima de frontend/)
_DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _formatar_preco(valor: float) -> str:
    """Formata um valor numérico como string de preço em dólar."""
    return f"${valor:,.2f}"


def _obter_cor_acao(acao: str) -> dict:
    """
    Retorna as cores CSS associadas à ação recomendada.

    Parâmetros:
        acao : 'COMPRAR', 'VENDER' ou 'MANTER'

    Retorna:
        Dicionário com 'primaria', 'glow' e 'fundo'
    """
    mapa_cores = {
        "COMPRAR": {
            "primaria": "#3fb950",
            "glow": "rgba(63, 185, 80, 0.4)",
            "fundo": "rgba(63, 185, 80, 0.08)",
            "emoji": "📈",
        },
        "VENDER": {
            "primaria": "#f85149",
            "glow": "rgba(248, 81, 73, 0.4)",
            "fundo": "rgba(248, 81, 73, 0.08)",
            "emoji": "📉",
        },
        "MANTER": {
            "primaria": "#f0883e",
            "glow": "rgba(240, 136, 62, 0.4)",
            "fundo": "rgba(240, 136, 62, 0.08)",
            "emoji": "⏸️",
        },
    }
    return mapa_cores.get(acao.upper(), mapa_cores["MANTER"])


def _gerar_grafico_plotly(df: pd.DataFrame, include_plotlyjs: bool | str = True) -> str:
    """
    Gera o gráfico interativo de candlestick com Plotly usando dados do Pandas.

    Cria um gráfico com dois painéis:
      - Painel superior: candlesticks + MM20 + MM50 + Bandas de Bollinger
      - Painel inferior: Volume + RSI

    Parâmetros:
        df : DataFrame enriquecido com indicadores técnicos

    Retorna:
        String HTML do gráfico Plotly (div + script)
    """
    figura = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Preço (BTC/USDT)", "Volume", "RSI (14)")
    )

    # ── Painel 1: Candlesticks ──────────────────────────────────────────────
    figura.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Abertura"],
            high=df["Maxima"],
            low=df["Minima"],
            close=df["Fechamento"],
            name="BTC/USDT",
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950",
            decreasing_fillcolor="#f85149",
        ),
        row=1, col=1
    )

    # MM20
    if "MM20" in df.columns:
        figura.add_trace(
            go.Scatter(
                x=df.index, y=df["MM20"],
                name="MM 20",
                line=dict(color="#f0883e", width=1.5),
                opacity=0.9
            ),
            row=1, col=1
        )

    # MM50
    if "MM50" in df.columns:
        figura.add_trace(
            go.Scatter(
                x=df.index, y=df["MM50"],
                name="MM 50",
                line=dict(color="#79c0ff", width=1.5),
                opacity=0.9
            ),
            row=1, col=1
        )

    # Bandas de Bollinger (com preenchimento entre elas)
    if "Bollinger_Superior" in df.columns:
        figura.add_trace(
            go.Scatter(
                x=df.index, y=df["Bollinger_Superior"],
                name="Bollinger Sup.",
                line=dict(color="#8b949e", width=0.8, dash="dot"),
                opacity=0.6,
            ),
            row=1, col=1
        )
        figura.add_trace(
            go.Scatter(
                x=df.index, y=df["Bollinger_Inferior"],
                name="Bollinger Inf.",
                line=dict(color="#8b949e", width=0.8, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(139, 148, 158, 0.05)",
                opacity=0.6,
            ),
            row=1, col=1
        )

    # ── Painel 2: Volume ────────────────────────────────────────────────────
    cores_volume = [
        "#3fb950" if c >= a else "#f85149"
        for c, a in zip(df["Fechamento"], df["Abertura"])
    ]
    figura.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=cores_volume,
            opacity=0.7,
            showlegend=False,
        ),
        row=2, col=1
    )

    # ── Painel 3: RSI ───────────────────────────────────────────────────────
    if "RSI14" in df.columns:
        figura.add_trace(
            go.Scatter(
                x=df.index, y=df["RSI14"],
                name="RSI 14",
                line=dict(color="#d2a8ff", width=1.5),
            ),
            row=3, col=1
        )
        # Linhas de referência RSI
        figura.add_hline(y=70, line_dash="dash", line_color="#f85149",
                         line_width=0.8, opacity=0.6, row=3, col=1)
        figura.add_hline(y=30, line_dash="dash", line_color="#3fb950",
                         line_width=0.8, opacity=0.6, row=3, col=1)

    # ── Layout ──────────────────────────────────────────────────────────────
    figura.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, system-ui, sans-serif", color="#c9d1d9", size=12),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.9)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    # Estilo dos eixos
    for i in range(1, 4):
        figura.update_xaxes(
            row=i, col=1,
            gridcolor="#21262d",
            linecolor="#30363d",
            zerolinecolor="#30363d",
        )
        figura.update_yaxes(
            row=i, col=1,
            gridcolor="#21262d",
            linecolor="#30363d",
            zerolinecolor="#30363d",
        )

    figura.update_yaxes(row=3, col=1, range=[0, 100])

    return figura.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
            "displaylogo": False,
            "scrollZoom": True,
        }
    )


def _gerar_card_indicador(rotulo: str, valor: str, descricao: str = "") -> str:
    """Gera o HTML de um card de indicador técnico."""
    return f"""
        <div class="card-indicador">
            <span class="rotulo-indicador">{rotulo}</span>
            <span class="valor-indicador">{valor}</span>
            {f'<span class="desc-indicador">{descricao}</span>' if descricao else ''}
        </div>
    """


def gerar_dashboard(
    df: pd.DataFrame,
    dados_ticker: dict,
    analise_ia: dict,
    indicadores: dict,
    par: str = "BTC/USDT",
    horario_analise: str = "",
    caminho_saida: str = os.path.join(_DIR_RAIZ, "resultados", "dashboard.html"),
    include_plotlyjs: bool | str = True
) -> str:
    """
    Gera o arquivo HTML do dashboard completo.

    Parâmetros:
        df              : DataFrame enriquecido com dados históricos e indicadores
        dados_ticker    : Dicionário com preço atual e variação 24h
        analise_ia      : Dicionário com acao, justificativa, preco_alvo, tempo_estimado
        indicadores     : Dicionário com resumo dos indicadores técnicos
        par             : Par de negociação (ex: 'BTC/USDT')
        horario_analise : Data/hora da análise formatada

    Retorna:
        Caminho do arquivo 'dashboard.html' gerado
    """
    preco_atual = dados_ticker.get("preco", 0.0)
    variacao_24h = dados_ticker.get("variacao_24h", 0.0)
    maxima_24h = dados_ticker.get("maxima_24h", 0.0)
    minima_24h = dados_ticker.get("minima_24h", 0.0)
    volume_24h = dados_ticker.get("volume_24h", 0.0)

    acao = analise_ia.get("acao", "MANTER")
    justificativa = analise_ia.get("justificativa", "")
    preco_alvo = analise_ia.get("preco_alvo")
    tempo_estimado = analise_ia.get("tempo_estimado")

    cores = _obter_cor_acao(acao)

    sinal_variacao = "+" if variacao_24h >= 0 else ""
    cor_variacao = "#3fb950" if variacao_24h >= 0 else "#f85149"

    grafico_html = _gerar_grafico_plotly(df, include_plotlyjs=include_plotlyjs)

    rsi_valor = indicadores.get("rsi14")
    rsi_status = (
        "Sobrecomprado" if rsi_valor and rsi_valor > 70
        else "Sobrevendido" if rsi_valor and rsi_valor < 30
        else "Neutro"
    )
    rsi_cor = (
        "#f85149" if rsi_valor and rsi_valor > 70
        else "#3fb950" if rsi_valor and rsi_valor < 30
        else "#f0883e"
    )

    cards_indicadores = ""
    if indicadores.get("rsi14"):
        cards_indicadores += _gerar_card_indicador(
            "RSI (14)", f"{indicadores['rsi14']:.1f}", rsi_status
        )
    if indicadores.get("macd"):
        tendencia_macd = "Alta ▲" if indicadores["macd"] > indicadores.get("sinal_macd", 0) else "Baixa ▼"
        cards_indicadores += _gerar_card_indicador(
            "MACD", f"{indicadores['macd']:.4f}", tendencia_macd
        )
    if indicadores.get("mm20"):
        cards_indicadores += _gerar_card_indicador(
            "MM 20", _formatar_preco(indicadores["mm20"])
        )
    if indicadores.get("mm50"):
        cards_indicadores += _gerar_card_indicador(
            "MM 50", _formatar_preco(indicadores["mm50"])
        )
    if indicadores.get("bollinger_superior"):
        cards_indicadores += _gerar_card_indicador(
            "Bollinger Sup.", _formatar_preco(indicadores["bollinger_superior"])
        )
    if indicadores.get("bollinger_inferior"):
        cards_indicadores += _gerar_card_indicador(
            "Bollinger Inf.", _formatar_preco(indicadores["bollinger_inferior"])
        )

    bloco_previsao = ""
    if preco_alvo:
        bloco_previsao += f"""
            <div class="previsao-item">
                <span class="previsao-rotulo">🎯 Preço Alvo</span>
                <span class="previsao-valor" style="color: {cores['primaria']};">{preco_alvo}</span>
            </div>
        """
    if tempo_estimado:
        bloco_previsao += f"""
            <div class="previsao-item">
                <span class="previsao-rotulo">⏱️ Tempo Estimado para Lucro</span>
                <span class="previsao-valor">{tempo_estimado}</span>
            </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Dashboard de análise técnica do Bitcoin com inteligência artificial (Google Gemini). Exibe preço atual, gráfico interativo e recomendação de compra, venda ou manutenção.">
    <title>CriptoIA — Análise Bitcoin com IA</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* ═══════════════════════════════════════════════════
           VARIÁVEIS E RESET
        ═══════════════════════════════════════════════════ */
        :root {{
            --fundo-pagina: #0d1117;
            --fundo-card: rgba(22, 27, 34, 0.85);
            --fundo-card-hover: rgba(30, 37, 48, 0.9);
            --borda: rgba(48, 54, 61, 0.8);
            --texto-primario: #e6edf3;
            --texto-secundario: #8b949e;
            --texto-terciario: #484f58;
            --cor-acao: {cores['primaria']};
            --glow-acao: {cores['glow']};
            --fundo-acao: {cores['fundo']};
            --verde: #3fb950;
            --vermelho: #f85149;
            --laranja: #f0883e;
            --azul: #79c0ff;
            --roxo: #d2a8ff;
            --sombra: 0 8px 32px rgba(0, 0, 0, 0.4);
            --radius: 12px;
            --radius-sm: 8px;
            --transicao: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        html {{ scroll-behavior: smooth; }}

        body {{
            background-color: var(--fundo-pagina);
            background-image:
                radial-gradient(ellipse 80% 60% at 20% -10%, rgba(63, 185, 80, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse 60% 50% at 80% 110%, rgba(121, 192, 255, 0.05) 0%, transparent 60%);
            color: var(--texto-primario);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            line-height: 1.6;
        }}

        /* ═══════════════════════════════════════════════════
           LAYOUT PRINCIPAL
        ═══════════════════════════════════════════════════ */
        .pagina {{
            max-width: 1280px;
            margin: 0 auto;
            padding: 0 24px 60px;
        }}

        /* ═══════════════════════════════════════════════════
           CABEÇALHO
        ═══════════════════════════════════════════════════ */
        .cabecalho {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 28px 0 36px;
            border-bottom: 1px solid var(--borda);
            margin-bottom: 40px;
            animation: deslizarBaixo 0.6s ease-out;
        }}

        .cabecalho-logo {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .logo-icone {{
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background: linear-gradient(135deg, #f0883e, #f85149);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: 0 0 20px rgba(240, 136, 62, 0.3);
        }}

        .logo-texto h1 {{
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, #e6edf3, #8b949e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .logo-texto p {{
            font-size: 0.75rem;
            color: var(--texto-secundario);
            margin-top: 1px;
        }}

        .horario-badge {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            background: var(--fundo-card);
            border: 1px solid var(--borda);
            border-radius: 20px;
            font-size: 0.8rem;
            color: var(--texto-secundario);
        }}

        .ponto-ao-vivo {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--verde);
            animation: piscar 2s infinite;
        }}

        /* ═══════════════════════════════════════════════════
           GRADE DE PREÇO (TOPO)
        ═══════════════════════════════════════════════════ */
        .grade-preco {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 16px;
            margin-bottom: 32px;
            animation: deslizarBaixo 0.7s ease-out 0.1s both;
        }}

        @media (max-width: 900px) {{ .grade-preco {{ grid-template-columns: 1fr 1fr; }} }}
        @media (max-width: 500px) {{ .grade-preco {{ grid-template-columns: 1fr; }} }}

        .card-preco {{
            background: var(--fundo-card);
            border: 1px solid var(--borda);
            border-radius: var(--radius);
            padding: 20px 24px;
            backdrop-filter: blur(12px);
            transition: var(--transicao);
            position: relative;
            overflow: hidden;
        }}

        .card-preco::before {{
            content: '';
            position: absolute;
            inset: 0;
            border-radius: var(--radius);
            background: linear-gradient(135deg, rgba(255,255,255,0.02), transparent);
            pointer-events: none;
        }}

        .card-preco:hover {{
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: var(--sombra);
        }}

        .card-preco-rotulo {{
            font-size: 0.72rem;
            font-weight: 500;
            color: var(--texto-secundario);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }}

        .card-preco-valor {{
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--texto-primario);
            letter-spacing: -0.02em;
        }}

        .card-preco-valor.destaque {{
            font-size: 2.2rem;
        }}

        .badge-variacao {{
            display: inline-flex;
            align-items: center;
            gap: 3px;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 6px;
            background: rgba({('63, 185, 80' if variacao_24h >= 0 else '248, 81, 73')}, 0.15);
            color: {cor_variacao};
        }}

        /* ═══════════════════════════════════════════════════
           GRÁFICO INTERATIVO
        ═══════════════════════════════════════════════════ */
        .secao-grafico {{
            background: var(--fundo-card);
            border: 1px solid var(--borda);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 32px;
            backdrop-filter: blur(12px);
            animation: deslizarBaixo 0.8s ease-out 0.2s both;
        }}

        .secao-titulo {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--texto-secundario);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .secao-titulo::before {{
            content: '';
            display: inline-block;
            width: 3px;
            height: 14px;
            background: var(--azul);
            border-radius: 2px;
        }}

        /* ═══════════════════════════════════════════════════
           GRADE INFERIOR: INDICADORES + RECOMENDAÇÃO IA
        ═══════════════════════════════════════════════════ */
        .grade-inferior {{
            display: grid;
            grid-template-columns: 1fr 1.6fr;
            gap: 20px;
            animation: deslizarBaixo 0.9s ease-out 0.3s both;
        }}

        @media (max-width: 800px) {{ .grade-inferior {{ grid-template-columns: 1fr; }} }}

        /* ── Indicadores ── */
        .card-indicadores {{
            background: var(--fundo-card);
            border: 1px solid var(--borda);
            border-radius: var(--radius);
            padding: 24px;
            backdrop-filter: blur(12px);
        }}

        .grade-indicadores {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
        }}

        .card-indicador {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--borda);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: var(--transicao);
        }}

        .card-indicador:hover {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.1);
        }}

        .rotulo-indicador {{
            font-size: 0.7rem;
            color: var(--texto-secundario);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .valor-indicador {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--texto-primario);
            font-variant-numeric: tabular-nums;
        }}

        .desc-indicador {{
            font-size: 0.7rem;
            color: var(--texto-secundario);
        }}

        /* ── Recomendação IA ── */
        .card-recomendacao {{
            background: var(--fundo-card);
            border: 1px solid var(--borda);
            border-radius: var(--radius);
            padding: 28px;
            backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            gap: 20px;
            position: relative;
            overflow: hidden;
        }}

        .card-recomendacao::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--cor-acao), transparent);
        }}

        .recomendacao-acao-wrapper {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .recomendacao-emoji {{
            font-size: 2.8rem;
            filter: drop-shadow(0 0 12px var(--glow-acao));
        }}

        .recomendacao-acao {{
            font-size: 2.8rem;
            font-weight: 800;
            color: var(--cor-acao);
            text-shadow: 0 0 30px var(--glow-acao);
            letter-spacing: -0.02em;
            line-height: 1;
        }}

        .recomendacao-subtitulo {{
            font-size: 0.75rem;
            color: var(--texto-secundario);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 4px;
        }}

        .divider {{
            height: 1px;
            background: var(--borda);
        }}

        .recomendacao-justificativa {{
            font-size: 0.92rem;
            color: #adbac7;
            line-height: 1.75;
        }}

        .grade-previsao {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}

        .previsao-item {{
            background: var(--fundo-acao);
            border: 1px solid rgba(var(--cor-acao), 0.2);
            border-color: color-mix(in srgb, var(--cor-acao) 25%, transparent);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .previsao-rotulo {{
            font-size: 0.72rem;
            color: var(--texto-secundario);
            font-weight: 500;
        }}

        .previsao-valor {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--texto-primario);
        }}

        .badge-ia {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            background: rgba(210, 168, 255, 0.1);
            border: 1px solid rgba(210, 168, 255, 0.2);
            border-radius: 20px;
            font-size: 0.75rem;
            color: var(--roxo);
            font-weight: 500;
            width: fit-content;
        }}

        /* ═══════════════════════════════════════════════════
           RODAPÉ
        ═══════════════════════════════════════════════════ */
        .rodape {{
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--borda);
            text-align: center;
            color: var(--texto-terciario);
            font-size: 0.78rem;
            animation: deslizarBaixo 1s ease-out 0.4s both;
        }}

        .rodape a {{
            color: var(--texto-secundario);
            text-decoration: none;
        }}

        /* ═══════════════════════════════════════════════════
           ANIMAÇÕES
        ═══════════════════════════════════════════════════ */
        @keyframes deslizarBaixo {{
            from {{ opacity: 0; transform: translateY(-16px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes piscar {{
            0%, 100% {{ opacity: 1; box-shadow: 0 0 6px #3fb950; }}
            50%       {{ opacity: 0.4; box-shadow: none; }}
        }}

        @keyframes pulsar {{
            0%, 100% {{ box-shadow: 0 0 0 0 var(--glow-acao); }}
            50%       {{ box-shadow: 0 0 20px 4px var(--glow-acao); }}
        }}

        .recomendacao-acao {{
            animation: pulsar 3s ease-in-out infinite;
        }}
    </style>
</head>
<body>
<div class="pagina">

    <!-- ── CABEÇALHO ─────────────────────────────────────────────────── -->
    <header class="cabecalho" id="cabecalho-principal">
        <div class="cabecalho-logo">
            <div class="logo-icone">₿</div>
            <div class="logo-texto">
                <h1>CriptoIA</h1>
                <p>Análise técnica com inteligência artificial</p>
            </div>
        </div>
        <div class="horario-badge" id="horario-analise">
            <div class="ponto-ao-vivo"></div>
            Análise gerada em: {horario_analise}
        </div>
    </header>

    <!-- ── GRADE DE PREÇO ─────────────────────────────────────────────── -->
    <section class="grade-preco" aria-label="Resumo de preço">
        <div class="card-preco" id="card-preco-atual">
            <div class="card-preco-rotulo">💰 Preço Atual</div>
            <div class="card-preco-valor destaque">{_formatar_preco(preco_atual)}</div>
            <div class="badge-variacao">
                {'▲' if variacao_24h >= 0 else '▼'} {sinal_variacao}{variacao_24h:.2f}% (24h)
            </div>
        </div>
        <div class="card-preco" id="card-maxima-24h">
            <div class="card-preco-rotulo">📈 Máxima 24h</div>
            <div class="card-preco-valor">{_formatar_preco(maxima_24h)}</div>
        </div>
        <div class="card-preco" id="card-minima-24h">
            <div class="card-preco-rotulo">📉 Mínima 24h</div>
            <div class="card-preco-valor">{_formatar_preco(minima_24h)}</div>
        </div>
        <div class="card-preco" id="card-volume-24h">
            <div class="card-preco-rotulo">📊 Volume 24h (USDT)</div>
            <div class="card-preco-valor" style="font-size: 1.3rem;">
                ${volume_24h / 1_000_000:.1f}M
            </div>
        </div>
    </section>

    <!-- ── GRÁFICO INTERATIVO ─────────────────────────────────────────── -->
    <section class="secao-grafico" id="secao-grafico-principal">
        <div class="secao-titulo">📊 Gráfico Interativo — {par} (90 dias)</div>
        {grafico_html}
    </section>

    <!-- ── INDICADORES + RECOMENDAÇÃO ────────────────────────────────── -->
    <div class="grade-inferior">

        <!-- Indicadores Técnicos -->
        <div class="card-indicadores" id="card-indicadores-tecnicos">
            <div class="secao-titulo">⚙️ Indicadores Técnicos</div>
            <div class="grade-indicadores">
                {cards_indicadores}
            </div>
        </div>

        <!-- Recomendação da IA -->
        <div class="card-recomendacao" id="card-recomendacao-ia">
            <div class="recomendacao-acao-wrapper">
                <span class="recomendacao-emoji">{cores['emoji']}</span>
                <div>
                    <div class="recomendacao-acao" id="acao-ia">{acao}</div>
                    <div class="recomendacao-subtitulo">Recomendação da Inteligência Artificial</div>
                </div>
            </div>

            <div class="divider"></div>

            <p class="recomendacao-justificativa" id="justificativa-ia">
                {justificativa}
            </p>

            {f'<div class="divider"></div><div class="grade-previsao">{bloco_previsao}</div>' if bloco_previsao else ''}
        </div>

    </div>

    <!-- ── RODAPÉ ─────────────────────────────────────────────────────── -->
    <footer class="rodape">
        <p>
            CriptoIA · Trabalho de Conclusão de Curso I · Ciência da Computação<br>
            Dados fornecidos pela <strong>Binance</strong> via CCXT ·
            Análise por <strong>Google Gemini</strong> ·
            <em>Não constitui recomendação financeira.</em>
        </p>
    </footer>

</div>
</body>
</html>
"""

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write(html)

    print(f"[Dashboard] Gerado com sucesso: {caminho_saida}")
    return caminho_saida


# Constante usada no template HTML
MODELO_GEMINI = "gemini-3.5-flash-lite"
