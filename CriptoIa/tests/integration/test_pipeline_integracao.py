"""
tests/integration/test_pipeline_integracao.py

Testes de integração para o sistema CriptoIA:
  - Integração do pipeline completo (coleta -> indicadores -> gráfico -> IA -> dashboard)
  - Integração de backtesting e comparação
  - Integração de múltiplos ativos paralelos
  - Sem dependência de internet ou navegador real (mocks na camada de I/O de rede)
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from backend.coleta_dados import buscar_dados_historicos, buscar_preco_atual
from backend.analise_tecnica import enriquecer_dataframe, resumir_indicadores
from backend.gerador_grafico import gerar_grafico_candlestick
from backend.analisador_ia import analisar_mercado_com_gemini
from frontend.gerador_dashboard import gerar_dashboard
from backend.backtesting import gerar_sinais, calcular_metricas
from backend.comparacao import comparar_estrategias
from backend.multiplos_ativos import analisar_multiplos


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv_mock_data():
    """Gera dados OHLCV para 140 candles (garantindo dados suficientes após MM50)."""
    np.random.seed(123)
    n = 140
    datas = pd.date_range("2025-01-01", periods=n, freq="D")
    precos = 90000.0 * np.cumprod(1 + np.random.normal(0.001, 0.015, n))
    df = pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.015,
        "Minima": precos * 0.985,
        "Fechamento": precos,
        "Volume": np.random.uniform(500, 2000, n),
    }, index=datas)
    return df


# ── Testes de Integração ───────────────────────────────────────────────────────

class TestPipelineIntegracao:

    @patch("backend.analisador_ia.os.getenv", return_value="chave_teste")
    @patch("backend.analisador_ia.genai.Client")
    def test_pipeline_completa_ponta_a_ponta(
        self, mock_genai_client, mock_getenv, ohlcv_mock_data, tmp_path
    ):
        """Executa a pipeline completa de ponta a ponta gerando todos os artefatos intermediários."""
        # 1. Mock do retorno do Gemini
        mock_resposta = MagicMock()
        mock_resposta.text = (
            '{"acao": "COMPRAR", "justificativa": "RSI neutro e MM20 acima de MM50.", '
            '"preco_alvo": "$98.000,00", "tempo_estimado": "7 dias"}'
        )
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.return_value = mock_resposta
        mock_genai_client.return_value = mock_cliente

        # 2. Análise Técnica
        df_enriquecido = enriquecer_dataframe(ohlcv_mock_data)
        indicadores = resumir_indicadores(df_enriquecido)
        assert "rsi14" in indicadores
        assert "mm20" in indicadores

        # 3. Geração do gráfico PNG
        caminho_grafico = str(tmp_path / "grafico_pipeline.png")
        gerar_grafico_candlestick(df_enriquecido, nome_arquivo=caminho_grafico)
        assert os.path.exists(caminho_grafico)
        assert os.path.getsize(caminho_grafico) > 50_000

        # 4. Análise com Gemini (mockado)
        analise_ia = analisar_mercado_com_gemini(caminho_grafico, indicadores)
        assert analise_ia["acao"] == "COMPRAR"

        # 5. Geração do Dashboard HTML
        caminho_dashboard = str(tmp_path / "dashboard_pipeline.html")
        dados_ticker = {
            "preco": float(ohlcv_mock_data["Fechamento"].iloc[-1]),
            "variacao_24h": 1.25,
            "volume_24h": 1000000.0,
            "maxima_24h": 95000.0,
            "minima_24h": 92000.0,
        }
        res_dashboard = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia,
            indicadores=indicadores,
            par="BTC/USDT",
            caminho_saida=caminho_dashboard,
        )
        assert os.path.exists(res_dashboard)
        assert os.path.getsize(res_dashboard) > 500_000

    def test_integracao_backtesting_comparacao(self, ohlcv_mock_data):
        """Testa integração entre enriquecimento, backtesting e comparação de estratégias."""
        df_enriquecido = enriquecer_dataframe(ohlcv_mock_data)
        res_comp = comparar_estrategias(par="BTC/USDT", intervalo="1d", df_precarregado=df_enriquecido)

        assert "criptoia" in res_comp
        assert "buy_and_hold" in res_comp
        assert "supervisionado" in res_comp

        # Retornos numéricos válidos
        assert isinstance(res_comp["criptoia"]["retorno_total_pct"], (int, float))
        assert isinstance(res_comp["buy_and_hold"]["retorno_total_pct"], (int, float))
        assert isinstance(res_comp["supervisionado"]["retorno_total_pct"], (int, float))
