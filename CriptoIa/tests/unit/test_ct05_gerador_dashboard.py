"""
tests/unit/test_ct05_gerador_dashboard.py

CT05 — RF06/RF07: Geração e Validação do Dashboard
====================================================
Verifica que o módulo gerador_dashboard.py:
  - Gera um arquivo HTML > 500 KB
  - O HTML contém o gráfico Plotly (identificável pela string 'plotly')
  - O HTML contém os indicadores técnicos
  - O HTML contém a recomendação da IA
  - Não depende do navegador real (usa validação de conteúdo, não webbrowser)

Todos os testes são UNITÁRIOS — não dependem de internet ou navegador real.
"""

import os
import pytest
import numpy as np
import pandas as pd

from frontend.gerador_dashboard import gerar_dashboard
from backend.analise_tecnica import enriquecer_dataframe


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _gerar_df_enriquecido(n: int = 90) -> pd.DataFrame:
    """Gera DataFrame enriquecido com dados sintéticos."""
    np.random.seed(7)
    datas = pd.date_range(start="2025-01-01", periods=n, freq="D")
    precos = 50000.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    df = pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.02,
        "Minima": precos * 0.98,
        "Fechamento": precos,
        "Volume": np.random.uniform(1e9, 5e9, n),
    }, index=datas)
    return enriquecer_dataframe(df)


@pytest.fixture(scope="module")
def df_enriquecido():
    return _gerar_df_enriquecido(90)


@pytest.fixture
def dados_ticker():
    return {
        "preco": 95000.0,
        "variacao_24h": 2.5,
        "volume_24h": 1234567890.0,
        "maxima_24h": 96500.0,
        "minima_24h": 93000.0,
    }


@pytest.fixture
def analise_ia_comprar():
    return {
        "acao": "COMPRAR",
        "justificativa": (
            "RSI em zona neutra com MACD acima da linha de sinal. "
            "MM20 cruzou acima da MM50 configurando golden cross. "
            "Boa oportunidade de entrada."
        ),
        "preco_alvo": "$98.500,00",
        "tempo_estimado": "7 a 14 dias",
    }


@pytest.fixture
def indicadores():
    return {
        "preco_fechamento": 95000.0,
        "mm20": 94000.0,
        "mm50": 91000.0,
        "rsi14": 58.5,
        "macd": 120.0,
        "sinal_macd": 95.0,
        "bollinger_superior": 98000.0,
        "bollinger_inferior": 90000.0,
    }


@pytest.fixture
def caminho_html_temp(tmp_path):
    return str(tmp_path / "dashboard_teste.html")


# ── CT05: Testes de geração do dashboard ───────────────────────────────────────

class TestCT05GeradorDashboard:

    def test_ct05_arquivo_html_criado(
        self, df_enriquecido, dados_ticker, analise_ia_comprar,
        indicadores, caminho_html_temp
    ):
        """CT05: Arquivo HTML deve ser criado."""
        resultado = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia_comprar,
            indicadores=indicadores,
            par="BTC/USDT",
            horario_analise="01/01/2025 às 12:00:00",
            caminho_saida=caminho_html_temp,
        )
        assert os.path.exists(resultado), (
            f"CT05 falhou: arquivo HTML não foi criado em '{resultado}'"
        )

    def test_ct05_tamanho_minimo_500kb(
        self, df_enriquecido, dados_ticker, analise_ia_comprar,
        indicadores, caminho_html_temp
    ):
        """CT05: Arquivo HTML deve ter > 500 KB."""
        resultado = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia_comprar,
            indicadores=indicadores,
            par="BTC/USDT",
            horario_analise="01/01/2025 às 12:00:00",
            caminho_saida=caminho_html_temp,
        )
        tamanho_bytes = os.path.getsize(resultado)
        assert tamanho_bytes > 500_000, (
            f"CT05 falhou: HTML tem {tamanho_bytes/1024:.1f} KB. Esperado > 500 KB."
        )

    def test_ct05_html_contem_plotly(
        self, df_enriquecido, dados_ticker, analise_ia_comprar,
        indicadores, caminho_html_temp
    ):
        """CT05: HTML deve conter o gráfico Plotly."""
        resultado = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia_comprar,
            indicadores=indicadores,
            par="BTC/USDT",
            horario_analise="01/01/2025 às 12:00:00",
            caminho_saida=caminho_html_temp,
        )
        with open(resultado, "r", encoding="utf-8") as f:
            conteudo = f.read()

        assert "plotly" in conteudo.lower(), (
            "CT05 falhou: HTML não contém referência ao Plotly (gráfico ausente)"
        )

    def test_ct05_html_contem_indicadores(
        self, df_enriquecido, dados_ticker, analise_ia_comprar,
        indicadores, caminho_html_temp
    ):
        """CT05: HTML deve conter indicadores técnicos."""
        resultado = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia_comprar,
            indicadores=indicadores,
            par="BTC/USDT",
            horario_analise="01/01/2025 às 12:00:00",
            caminho_saida=caminho_html_temp,
        )
        with open(resultado, "r", encoding="utf-8") as f:
            conteudo = f.read()

        # Verifica que pelo menos um dos indicadores está no HTML
        assert any(
            termo in conteudo for termo in ["RSI", "MACD", "Bollinger", "MM20"]
        ), "CT05 falhou: HTML não contém indicadores técnicos"

    def test_ct05_html_contem_recomendacao_ia(
        self, df_enriquecido, dados_ticker, analise_ia_comprar,
        indicadores, caminho_html_temp
    ):
        """CT05: HTML deve conter a recomendação da IA."""
        resultado = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia_comprar,
            indicadores=indicadores,
            par="BTC/USDT",
            horario_analise="01/01/2025 às 12:00:00",
            caminho_saida=caminho_html_temp,
        )
        with open(resultado, "r", encoding="utf-8") as f:
            conteudo = f.read()

        assert "COMPRAR" in conteudo, (
            "CT05 falhou: HTML não contém a recomendação COMPRAR da IA"
        )

    def test_ct05_html_valido_tag_basica(
        self, df_enriquecido, dados_ticker, analise_ia_comprar,
        indicadores, caminho_html_temp
    ):
        """CT05: HTML deve ter estrutura básica válida."""
        resultado = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia_comprar,
            indicadores=indicadores,
            par="BTC/USDT",
            horario_analise="01/01/2025 às 12:00:00",
            caminho_saida=caminho_html_temp,
        )
        with open(resultado, "r", encoding="utf-8") as f:
            conteudo = f.read()

        assert "<!DOCTYPE html>" in conteudo or "<html" in conteudo.lower(), (
            "CT05 falhou: HTML não tem tag DOCTYPE ou <html>"
        )
        assert "</html>" in conteudo.lower(), (
            "CT05 falhou: HTML não tem tag </html> de fechamento"
        )

    def test_ct05_diferentes_acoes(
        self, df_enriquecido, dados_ticker, indicadores, caminho_html_temp
    ):
        """CT05: Dashboard deve funcionar com COMPRAR, VENDER e MANTER."""
        for acao in ["COMPRAR", "VENDER", "MANTER"]:
            analise = {
                "acao": acao,
                "justificativa": f"Teste de ação {acao}",
                "preco_alvo": "$95.000,00" if acao != "MANTER" else None,
                "tempo_estimado": "7 dias" if acao != "MANTER" else None,
            }
            caminho = caminho_html_temp.replace(".html", f"_{acao}.html")
            resultado = gerar_dashboard(
                df=df_enriquecido,
                dados_ticker=dados_ticker,
                analise_ia=analise,
                indicadores=indicadores,
                par="BTC/USDT",
                horario_analise="01/01/2025 às 12:00:00",
                caminho_saida=caminho,
            )
            assert os.path.exists(resultado), (
                f"CT05 falhou: dashboard com acao={acao} não foi criado"
            )
            with open(resultado, "r", encoding="utf-8") as f:
                conteudo = f.read()
            assert acao in conteudo, (
                f"CT05 falhou: ação '{acao}' não encontrada no HTML"
            )
