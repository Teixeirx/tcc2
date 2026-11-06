"""
tests/unit/test_ct03_gerador_grafico.py

CT03 — RF04: Geração do Gráfico PNG
=====================================
Verifica que o módulo gerador_grafico.py:
  - Gera um arquivo grafico.png em disco
  - O arquivo tem tamanho > 50 KB
  - O arquivo é uma imagem válida abrível com Pillow

Nota: Este teste realmente gera um arquivo PNG (não usa mock para o arquivo),
pois a validação de uma imagem real é o critério de sucesso do CT03.
Não depende de internet.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from backend.gerador_grafico import gerar_grafico_candlestick
from backend.analise_tecnica import enriquecer_dataframe


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _gerar_df_enriquecido(n: int = 90) -> pd.DataFrame:
    """Gera DataFrame enriquecido com dados sintéticos."""
    np.random.seed(0)
    datas = pd.date_range(start="2025-01-01", periods=n, freq="D")
    precos = 50000.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
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
    """DataFrame enriquecido compartilhado no módulo."""
    return _gerar_df_enriquecido(90)


@pytest.fixture
def caminho_grafico_temp(tmp_path):
    """Retorna um caminho temporário para o gráfico."""
    return str(tmp_path / "grafico_teste.png")


# ── CT03: Testes de geração de gráfico ─────────────────────────────────────────

class TestCT03GeradorGrafico:

    def test_ct03_arquivo_criado(self, df_enriquecido, caminho_grafico_temp):
        """CT03: O arquivo PNG deve ser criado no caminho especificado."""
        resultado = gerar_grafico_candlestick(
            df_enriquecido,
            nome_arquivo=caminho_grafico_temp,
            titulo="BTC/USDT — Teste CT03"
        )
        assert os.path.exists(resultado), (
            f"CT03 falhou: arquivo não foi criado em {resultado}"
        )

    def test_ct03_tamanho_minimo_50kb(self, df_enriquecido, caminho_grafico_temp):
        """CT03: O arquivo PNG deve ter tamanho mínimo de 50 KB."""
        resultado = gerar_grafico_candlestick(
            df_enriquecido,
            nome_arquivo=caminho_grafico_temp,
            titulo="BTC/USDT — Teste CT03 Tamanho"
        )
        tamanho_bytes = os.path.getsize(resultado)
        assert tamanho_bytes > 50_000, (
            f"CT03 falhou: arquivo PNG tem apenas {tamanho_bytes} bytes "
            f"({tamanho_bytes/1024:.1f} KB). Esperado > 50 KB."
        )

    def test_ct03_imagem_valida_pillow(self, df_enriquecido, caminho_grafico_temp):
        """CT03: O arquivo deve ser uma imagem válida e abrível com Pillow."""
        gerar_grafico_candlestick(
            df_enriquecido,
            nome_arquivo=caminho_grafico_temp,
            titulo="BTC/USDT — Teste CT03 Pillow"
        )
        try:
            img = Image.open(caminho_grafico_temp)
            img.verify()  # Lança exceção se a imagem for corrompida
        except Exception as e:
            pytest.fail(
                f"CT03 falhou: Pillow não conseguiu abrir a imagem. Erro: {e}"
            )

    def test_ct03_formato_png(self, df_enriquecido, caminho_grafico_temp):
        """CT03: O arquivo deve ser no formato PNG."""
        gerar_grafico_candlestick(
            df_enriquecido,
            nome_arquivo=caminho_grafico_temp,
        )
        # Verifica os primeiros bytes (PNG magic bytes: 89 50 4E 47)
        with open(caminho_grafico_temp, "rb") as f:
            header = f.read(4)
        assert header == b"\x89PNG", (
            f"CT03 falhou: arquivo não é PNG válido. Header: {header!r}"
        )

    def test_ct03_retorna_caminho_correto(self, df_enriquecido, caminho_grafico_temp):
        """CT03: A função deve retornar o caminho do arquivo gerado."""
        retorno = gerar_grafico_candlestick(
            df_enriquecido,
            nome_arquivo=caminho_grafico_temp,
        )
        assert retorno == caminho_grafico_temp, (
            f"CT03 falhou: caminho retornado '{retorno}' != '{caminho_grafico_temp}'"
        )

    def test_ct03_imagem_dimensoes_razoaveis(self, df_enriquecido, caminho_grafico_temp):
        """CT03: A imagem deve ter dimensões razoáveis (largura > 800px)."""
        gerar_grafico_candlestick(
            df_enriquecido,
            nome_arquivo=caminho_grafico_temp,
        )
        img = Image.open(caminho_grafico_temp)
        largura, altura = img.size
        assert largura > 800, (
            f"CT03 falhou: largura da imagem é {largura}px, esperado > 800px"
        )
        assert altura > 400, (
            f"CT03 falhou: altura da imagem é {altura}px, esperado > 400px"
        )
