"""
tests/unit/test_ct02_analise_tecnica.py

CT02 — RF03: Cálculo dos Indicadores Técnicos
==============================================
Verifica que o módulo analise_tecnica.py:
  - Requer pelo menos 50 candles (não produz resultados significativos com menos)
  - Produz as colunas obrigatórias: MM20, MM50, RSI14, MACD, Sinal_MACD,
    Bollinger_Superior, Bollinger_Inferior
  - RSI fica entre 0 e 100 nos valores não-NaN
  - Bollinger_Superior é sempre maior que Bollinger_Inferior nos valores não-NaN

Todos os testes são UNITÁRIOS — não dependem de internet.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.analise_tecnica import (
    enriquecer_dataframe,
    calcular_rsi,
    calcular_macd,
    calcular_bandas_bollinger,
    calcular_media_movel,
    resumir_indicadores,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _gerar_df_candles(n: int, preco_inicial: float = 50000.0) -> pd.DataFrame:
    """Gera DataFrame de candles OHLCV sintético para testes."""
    np.random.seed(42)
    datas = pd.date_range(start="2025-01-01", periods=n, freq="D")
    precos = preco_inicial * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    df = pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.02,
        "Minima": precos * 0.98,
        "Fechamento": precos,
        "Volume": np.random.uniform(1000, 5000, n),
    }, index=datas)
    return df


@pytest.fixture
def df_90_candles():
    """DataFrame com 90 candles para testes de indicadores."""
    return _gerar_df_candles(90)


@pytest.fixture
def df_50_candles():
    """DataFrame com exatamente 50 candles (mínimo aceito)."""
    return _gerar_df_candles(50)


@pytest.fixture
def df_enriquecido_90(df_90_candles):
    """DataFrame enriquecido com indicadores (90 candles)."""
    return enriquecer_dataframe(df_90_candles)


# ── CT02.1: Colunas obrigatórias presentes ─────────────────────────────────────

class TestCT02Indicadores:

    def test_ct02_colunas_obrigatorias_presentes(self, df_enriquecido_90):
        """CT02: Todas as colunas de indicadores devem estar presentes."""
        colunas_obrigatorias = [
            "MM20", "MM50", "RSI14", "MACD", "Sinal_MACD",
            "Bollinger_Superior", "Bollinger_Inferior",
        ]
        for col in colunas_obrigatorias:
            assert col in df_enriquecido_90.columns, (
                f"CT02 falhou: coluna '{col}' ausente no DataFrame enriquecido. "
                f"Colunas presentes: {list(df_enriquecido_90.columns)}"
            )

    # ── CT02.2: RSI entre 0 e 100 ──────────────────────────────────────────────

    def test_ct02_rsi_intervalo_valido(self, df_enriquecido_90):
        """CT02: RSI14 deve estar entre 0 e 100 em todos os valores não-NaN."""
        rsi_valido = df_enriquecido_90["RSI14"].dropna()

        assert (rsi_valido >= 0).all() and (rsi_valido <= 100).all(), (
            f"CT02 falhou: RSI fora do intervalo [0, 100]. "
            f"Min={rsi_valido.min():.4f}, Max={rsi_valido.max():.4f}"
        )

    # ── CT02.3: Bollinger Superior > Inferior ──────────────────────────────────

    def test_ct02_bollinger_superior_maior_inferior(self, df_enriquecido_90):
        """CT02: Bollinger_Superior deve ser sempre maior que Bollinger_Inferior."""
        df = df_enriquecido_90.dropna(subset=["Bollinger_Superior", "Bollinger_Inferior"])

        violacoes = (df["Bollinger_Superior"] <= df["Bollinger_Inferior"]).sum()
        assert violacoes == 0, (
            f"CT02 falhou: {violacoes} linha(s) com Bollinger_Superior <= Bollinger_Inferior"
        )

    # ── CT02.4: MM20 e MM50 têm valores numéricos ──────────────────────────────

    def test_ct02_medias_moveis_numericas(self, df_enriquecido_90):
        """CT02: MM20 e MM50 devem ter valores numéricos (float), não strings."""
        for col in ["MM20", "MM50"]:
            valores_validos = df_enriquecido_90[col].dropna()
            assert len(valores_validos) > 0, (
                f"CT02 falhou: coluna {col} não tem valores válidos"
            )
            assert pd.api.types.is_float_dtype(valores_validos), (
                f"CT02 falhou: {col} não é float. Dtype: {valores_validos.dtype}"
            )

    # ── CT02.5: Pelo menos 50 candles necessários ──────────────────────────────

    def test_ct02_minimo_50_candles_gera_indicadores(self, df_50_candles):
        """CT02: Com 50 candles, os indicadores devem ser calculados sem erro."""
        df_enr = enriquecer_dataframe(df_50_candles)
        # Com 50 candles, MM50 terá apenas 1 valor não-NaN (o último)
        assert "MM50" in df_enr.columns, "CT02 falhou: MM50 ausente com 50 candles"
        assert "RSI14" in df_enr.columns, "CT02 falhou: RSI14 ausente com 50 candles"

    # ── CT02.6: Com menos de 20 candles, MM20 será todo NaN ───────────────────

    def test_ct02_menos_de_50_candles_rejeitado(self):
        """CT02: Com menos de 50 candles, a análise deve ser rejeitada com erro claro."""
        df_10 = _gerar_df_candles(10)
        with pytest.raises(ValueError, match="pelo menos 50 candles"):
            enriquecer_dataframe(df_10)

    # ── CT02.7: resumir_indicadores retorna dict com chaves corretas ───────────

    def test_ct02_resumir_indicadores_chaves(self, df_enriquecido_90):
        """CT02: resumir_indicadores deve retornar dict com todas as chaves."""
        resumo = resumir_indicadores(df_enriquecido_90)
        chaves_esperadas = [
            "preco_fechamento", "mm20", "mm50", "rsi14",
            "macd", "sinal_macd", "bollinger_superior", "bollinger_inferior",
        ]
        for chave in chaves_esperadas:
            assert chave in resumo, (
                f"CT02 falhou: chave '{chave}' ausente em resumir_indicadores"
            )

    # ── CT02.8: MACD calculado corretamente (sinal suavizado) ─────────────────

    def test_ct02_macd_sinal_correlacionados(self, df_enriquecido_90):
        """CT02: MACD e Sinal_MACD devem ter alta correlação (EMA suaviza MACD)."""
        df_valido = df_enriquecido_90.dropna(subset=["MACD", "Sinal_MACD"])
        if len(df_valido) < 2:
            pytest.skip("Dados insuficientes para correlação")

        correlacao = df_valido["MACD"].corr(df_valido["Sinal_MACD"])
        assert correlacao > 0.9, (
            f"CT02: correlação MACD/Sinal baixa ({correlacao:.4f}). "
            "Isso pode indicar erro no cálculo."
        )

    # ── CT02.9: Bollinger Media é a média de Bollinger Superior e Inferior ─────

    def test_ct02_bollinger_media_entre_bandas(self, df_enriquecido_90):
        """CT02: Bollinger_Media deve estar entre Superior e Inferior."""
        df = df_enriquecido_90.dropna(
            subset=["Bollinger_Superior", "Bollinger_Media", "Bollinger_Inferior"]
        )
        assert (df["Bollinger_Media"] <= df["Bollinger_Superior"]).all(), (
            "CT02: Bollinger_Media deve ser <= Bollinger_Superior"
        )
        assert (df["Bollinger_Media"] >= df["Bollinger_Inferior"]).all(), (
            "CT02: Bollinger_Media deve ser >= Bollinger_Inferior"
        )
