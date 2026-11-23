"""
tests/unit/test_multiplos_ativos.py

Testes unitários para o módulo backend/multiplos_ativos.py:
  - Análise de único par (com mock)
  - Análise em paralelo com ThreadPoolExecutor
  - Resiliência: falha em um par não interrompe os demais
  - Geração de relatório consolidado (DataFrame e texto)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from backend.multiplos_ativos import (
    analisar_par,
    analisar_multiplos,
    gerar_relatorio_consolidado,
    salvar_relatorio_csv,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def df_simulado():
    n = 60
    datas = pd.date_range("2025-01-01", periods=n, freq="D")
    precos = np.linspace(90000, 95000, n)
    return pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.01,
        "Minima": precos * 0.99,
        "Fechamento": precos,
        "Volume": np.full(n, 1000.0),
    }, index=datas)


@pytest.fixture
def ticker_simulado():
    return {
        "preco": 95000.0,
        "variacao_24h": 1.5,
        "volume_24h": 500000.0,
        "maxima_24h": 96000.0,
        "minima_24h": 94000.0,
    }


# ── Testes ─────────────────────────────────────────────────────────────────────

class TestMultiplosAtivos:

    @patch("backend.multiplos_ativos.buscar_dados_historicos")
    @patch("backend.multiplos_ativos.buscar_preco_atual")
    def test_analisar_par_sem_ia(self, mock_ticker, mock_dados, df_simulado, ticker_simulado):
        """Verifica a análise de um único par sem chamada de IA."""
        mock_dados.return_value = df_simulado
        mock_ticker.return_value = ticker_simulado

        res = analisar_par("BTC/USDT", "1d", 60, gerar_grafico=False, usar_ia=False)
        assert res["par"] == "BTC/USDT"
        assert "indicadores" in res
        assert "ticker" in res
        assert "erro" not in res

    def test_lista_vazia_retorna_vazio(self):
        """Sem combinações, não deve criar executor nem falhar."""
        assert analisar_multiplos([], max_workers=1) == []

    @patch("backend.multiplos_ativos.buscar_dados_historicos")
    @patch("backend.multiplos_ativos.buscar_preco_atual")
    def test_paralelismo_multiplos_pares(self, mock_ticker, mock_dados, df_simulado, ticker_simulado):
        """Verifica se múltiplos pares são executados em paralelo."""
        mock_dados.return_value = df_simulado
        mock_ticker.return_value = ticker_simulado

        combinacoes = [
            {"par": "BTC/USDT", "intervalo": "1d", "quantidade": 60},
            {"par": "ETH/USDT", "intervalo": "1d", "quantidade": 60},
            {"par": "SOL/USDT", "intervalo": "1d", "quantidade": 60},
        ]
        resultados = analisar_multiplos(
            combinacoes=combinacoes,
            max_workers=3,
            gerar_graficos=False,
            usar_ia=False,
        )
        assert len(resultados) == 3
        assert [r["par"] for r in resultados] == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        pares_retornados = {r["par"] for r in resultados}
        assert pares_retornados == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}

    @patch("backend.multiplos_ativos.buscar_dados_historicos")
    @patch("backend.multiplos_ativos.buscar_preco_atual")
    def test_resiliencia_falha_individual(self, mock_ticker, mock_dados, df_simulado, ticker_simulado):
        """Falha em um ativo não deve interromper os outros."""
        def side_effect_dados(par, intervalo, quantidade):
            if "FAIL" in par:
                raise ConnectionError("Exchange indisponível para este par")
            return df_simulado

        mock_dados.side_effect = side_effect_dados
        mock_ticker.return_value = ticker_simulado

        combinacoes = [
            {"par": "BTC/USDT", "intervalo": "1d", "quantidade": 60},
            {"par": "FAIL/USDT", "intervalo": "1d", "quantidade": 60},
            {"par": "SOL/USDT", "intervalo": "1d", "quantidade": 60},
        ]
        resultados = analisar_multiplos(
            combinacoes=combinacoes,
            max_workers=2,
            gerar_graficos=False,
            usar_ia=False,
        )
        assert len(resultados) == 3
        erros = [r for r in resultados if "erro" in r]
        sucessos = [r for r in resultados if "erro" not in r]
        assert len(erros) == 1
        assert "FAIL/USDT" == erros[0]["par"]
        assert len(sucessos) == 2

    def test_gerar_relatorio_consolidado(self):
        """Verifica a consolidação dos resultados em DataFrame e texto."""
        resultados = [
            {
                "par": "BTC/USDT",
                "intervalo": "1d",
                "ticker": {"preco": 95000.0, "variacao_24h": 2.1},
                "indicadores": {"rsi14": 55.0, "mm20": 93000.0, "mm50": 90000.0},
                "analise_ia": {"acao": "COMPRAR", "preco_alvo": "$98,000"},
            },
            {
                "par": "ETH/USDT",
                "intervalo": "1d",
                "erro": "Timeout na API",
            }
        ]
        relatorio = gerar_relatorio_consolidado(resultados)
        assert "dataframe" in relatorio
        assert "resumo_texto" in relatorio
        df_tabela = relatorio["dataframe"]
        assert isinstance(df_tabela, pd.DataFrame)
        assert len(df_tabela) == 2
        assert "Par" in df_tabela.columns
        assert "Ação_IA" in df_tabela.columns

    @patch("backend.multiplos_ativos.analisar_mercado_com_gemini")
    @patch("backend.multiplos_ativos.gerar_grafico_candlestick")
    @patch("backend.multiplos_ativos.buscar_dados_historicos")
    @patch("backend.multiplos_ativos.buscar_preco_atual")
    def test_analisar_par_com_ia(self, mock_ticker, mock_dados, mock_grafico,
                                   mock_gemini, df_simulado, ticker_simulado):
        """Com IA habilitada, o gráfico e a análise devem ser retornados."""
        mock_dados.return_value = df_simulado
        mock_ticker.return_value = ticker_simulado
        mock_grafico.return_value = "grafico_BTC_USDT_1d.png"
        mock_gemini.return_value = {
            "acao": "COMPRAR", "justificativa": "Teste",
            "preco_alvo": "$100.000", "tempo_estimado": "7 dias",
        }

        res = analisar_par("BTC/USDT", "1d", 60, gerar_grafico=True, usar_ia=True)

        assert res["caminho_grafico"] == "grafico_BTC_USDT_1d.png"
        assert res["analise_ia"]["acao"] == "COMPRAR"
        mock_gemini.assert_called_once()

    def test_relatorio_aceita_preco_alvo_numerico(self):
        """preco_alvo numérico não deve causar TypeError na concatenação."""
        relatorio = gerar_relatorio_consolidado([{
            "par": "BTC/USDT", "intervalo": "1d",
            "ticker": {"preco": 95000.0},
            "indicadores": {"rsi14": 55.0},
            "analise_ia": {"acao": "COMPRAR", "preco_alvo": 100000.0},
        }])
        assert "Alvo: 100000.0" in relatorio["resumo_texto"]

    def test_salvar_relatorio_csv(self, tmp_path):
        caminho = tmp_path / "relatorio.csv"
        retorno = salvar_relatorio_csv([{
            "par": "BTC/USDT", "intervalo": "1d",
            "ticker": {"preco": 95000.0},
            "indicadores": {"rsi14": 55.0},
            "analise_ia": {"acao": "MANTER"},
        }], str(caminho))
        assert caminho.exists()
        assert retorno == str(caminho.resolve())
