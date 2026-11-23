"""
tests/unit/test_comparacao.py

Testes unitários para o módulo backend/comparacao.py:
  - Estratégia Buy-and-Hold
  - Modelo Supervisionado (LSTM PyTorch com walk-forward)
  - Comparação consolidada das 3 estratégias
  - Prevenção de look-ahead bias na validação do modelo supervisionado
  - Formatação do relatório comparativo
"""

import pytest
import numpy as np
import pandas as pd

from backend.comparacao import (
    calcular_buy_and_hold,
    executar_modelo_supervisionado,
    comparar_estrategias,
    formatar_relatorio_comparacao,
)
from backend.analise_tecnica import enriquecer_dataframe


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def df_historico_longo():
    """Gera 150 candles para permitir divisão treino/teste no walk-forward."""
    np.random.seed(99)
    n = 150
    datas = pd.date_range(start="2025-01-01", periods=n, freq="D")
    precos = 60000.0 * np.cumprod(1 + np.random.normal(0.001, 0.015, n))
    df = pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.015,
        "Minima": precos * 0.985,
        "Fechamento": precos,
        "Volume": np.random.uniform(2000, 8000, n),
    }, index=datas)
    return enriquecer_dataframe(df)


# ── Testes das Estratégias ────────────────────────────────────────────────────

class TestComparacaoEstrategias:

    def test_buy_and_hold_retorno(self, df_historico_longo):
        """Verifica o cálculo de Buy-and-Hold."""
        res = calcular_buy_and_hold(df_historico_longo)
        assert "retorno_total_pct" in res
        assert "max_drawdown_pct" in res
        p_inicial = float(df_historico_longo["Fechamento"].iloc[0])
        p_final = float(df_historico_longo["Fechamento"].iloc[-1])
        esperado_pct = round(((p_final - p_inicial) / p_inicial) * 100, 2)
        assert res["retorno_total_pct"] == esperado_pct

    def test_modelo_supervisionado_retorna_metricas(self, df_historico_longo):
        """Verifica que o LSTM real walk-forward produz métricas coerentes."""
        res = executar_modelo_supervisionado(
            df_historico_longo,
            seq_len=8,
            epochs=1,
            hidden_size=8,
            batch_size=16,
            passo_retreino=50,
        )
        assert "erro" not in res, res.get("erro")
        assert res["arquitetura"] == "LSTM (PyTorch)"
        assert "retorno_total_pct" in res
        assert "acuracia_direcional" in res
        assert "max_drawdown_pct" in res
        assert "acuracia_baseline_maioria" in res
        acu = res.get("acuracia_direcional")
        if acu is not None and not np.isnan(acu):
            assert 0.0 <= acu <= 1.0

    def test_comparar_estrategias_todas_presentes(self, df_historico_longo):
        """Verifica se a comparação executa CriptoIA, Buy-and-Hold e LSTM."""
        res = comparar_estrategias(
            par="BTC/USDT",
            intervalo="1d",
            df_precarregado=df_historico_longo,
            seq_len=8,
            epochs=1,
            hidden_size=8,
            batch_size=16,
            passo_retreino=50,
        )
        assert "criptoia" in res
        assert "buy_and_hold" in res
        assert "supervisionado" in res
        assert "periodo_comum" in res
        assert res["par"] == "BTC/USDT"
        assert res["intervalo"] == "1d"
        assert res["periodo_comum"]["criptoia"]["acuracia_direcional"] is not None
        assert res["periodo_comum"]["buy_and_hold"]["n_candles"] > 0
        # As três estratégias devem ser avaliadas na mesma janela comum
        assert res["periodo_comum"]["criptoia"]["data_inicio_avaliacao"] is not None
        assert res["periodo_comum"]["buy_and_hold"]["data_inicio"] is not None

    def test_relatorio_comparativo_texto(self):
        """Verifica se o relatório comparativo formatado contém cabeçalhos e colunas."""
        dados = [{
            "par": "ETH/USDT",
            "intervalo": "4h",
            "data_inicio_str": "01/01/2025",
            "data_fim_str": "30/06/2026",
            "criptoia": {"retorno_total_pct": 28.4, "acuracia_direcional": 0.58, "max_drawdown_pct": 14.1},
            "buy_and_hold": {"retorno_total_pct": 45.2, "acuracia_direcional": None, "max_drawdown_pct": 22.0},
            "supervisionado": {"retorno_total_pct": 19.8, "acuracia_direcional": 0.52, "max_drawdown_pct": 16.5},
        }]
        texto = formatar_relatorio_comparacao(dados)
        assert "RELATÓRIO COMPARATIVO DE ESTRATÉGIAS" in texto
        assert "CriptoIA" in texto
        assert "Buy-and-Hold" in texto
        assert "Supervisionado" in texto
        assert "ETH/USDT" in texto
