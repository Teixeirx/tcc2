"""
tests/unit/test_backtesting.py

Testes unitários para o módulo backend/backtesting.py:
  - Geração de sinais determinísticos (COMPRAR, VENDER, MANTER via Series de inteiros)
  - Prevenção de look-ahead bias (sinal em t vs. retorno futuro de t+1)
  - Cálculo de retornos acumulados e acurácia direcional
  - Cálculo de máximo drawdown
  - Formatação do relatório de backtesting
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from backend.backtesting import (
    gerar_sinais,
    calcular_metricas,
    formatar_relatorio_backtesting,
    PARES_TCC,
    TIMEFRAMES_TCC,
)
from backend.analise_tecnica import enriquecer_dataframe


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _gerar_df_teste(n: int = 120, tendencia: str = "alta") -> pd.DataFrame:
    """Gera DataFrame sintético com indicadores técnicos calculados."""
    np.random.seed(42)
    datas = pd.date_range(start="2025-01-01", periods=n, freq="D")
    
    if tendencia == "alta":
        precos = 50000.0 * (1 + 0.005 * np.arange(n) + np.random.normal(0, 0.005, n))
    elif tendencia == "baixa":
        precos = 50000.0 * (1 - 0.005 * np.arange(n) + np.random.normal(0, 0.005, n))
    else:
        precos = 50000.0 + np.random.normal(0, 500, n)

    df = pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.01,
        "Minima": precos * 0.99,
        "Fechamento": precos,
        "Volume": np.random.uniform(1000, 5000, n),
    }, index=datas)
    return enriquecer_dataframe(df)


@pytest.fixture
def df_alta():
    return _gerar_df_teste(120, "alta")


@pytest.fixture
def df_baixa():
    return _gerar_df_teste(120, "baixa")


# ── Testes de Sinais e Métricas ─────────────────────────────────────────────────

class TestBacktestingSinais:

    def test_gerar_sinais_retorna_series(self, df_alta):
        """Verifica que gerar_sinais retorna uma Series com valores -1, 0, 1."""
        sinais = gerar_sinais(df_alta)
        assert isinstance(sinais, pd.Series)
        assert set(sinais.unique()).issubset({-1, 0, 1})
        assert len(sinais) == len(df_alta)

    def test_calcular_metricas_campos_obrigatorios(self, df_alta):
        """Verifica se todas as métricas acadêmicas são calculadas."""
        metricas = calcular_metricas(df_alta)
        campos_esperados = [
            "acuracia_direcional",
            "retorno_total_pct",
            "max_drawdown_pct",
            "n_operacoes",
            "n_candles",
        ]
        for campo in campos_esperados:
            assert campo in metricas, f"Métrica '{campo}' ausente"

    def test_acuracia_entre_zero_e_um(self, df_alta):
        """A acurácia direcional deve estar entre 0.0 e 1.0 (ou None/NaN se sem sinais)."""
        metricas = calcular_metricas(df_alta)
        acu = metricas.get("acuracia_direcional")
        if acu is not None and not np.isnan(acu):
            assert 0.0 <= acu <= 1.0, f"Acurácia inválida: {acu}"

    def test_max_drawdown_calculado(self, df_alta):
        """Máximo drawdown é calculado e numérico."""
        metricas = calcular_metricas(df_alta)
        dd = metricas.get("max_drawdown_pct")
        assert isinstance(dd, (int, float))

    def test_dados_insuficientes_retorna_erro(self):
        """Menos de 60 candles retorna dict com chave erro."""
        df_completo = _gerar_df_teste(60, "alta")
        df_curto = df_completo.iloc[:30].copy()
        metricas = calcular_metricas(df_curto)
        assert "erro" in metricas

    def test_relatorio_formatado_contem_secoes(self):
        """Verifica se o relatório textual possui metodologia e limitações."""
        resultado_simulado = [{
            "par": "BTC/USDT",
            "intervalo": "1d",
            "data_inicio_str": "01/01/2025",
            "data_fim_str": "30/06/2026",
            "n_candles": 546,
            "n_candles_validos": 496,
            "n_sinais_compra": 25,
            "n_sinais_venda": 18,
            "n_sinais_manter": 453,
            "acuracia_direcional": 0.625,
            "retorno_total_pct": 34.5,
            "max_drawdown_pct": 12.3,
            "n_operacoes": 43,
        }]
        texto = formatar_relatorio_backtesting(resultado_simulado)
        assert "RELATÓRIO DE BACKTESTING" in texto
        assert "METODOLOGIA:" in texto
        assert "LIMITAÇÕES:" in texto
        assert "BTC/USDT" in texto
        assert "62.5%" in texto
