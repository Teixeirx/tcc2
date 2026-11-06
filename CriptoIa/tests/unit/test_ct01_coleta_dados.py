"""
tests/unit/test_ct01_coleta_dados.py

CT01 — RF01: Coleta de Dados Históricos
========================================
Verifica que o módulo coleta_dados.py:
  - Retorna um DataFrame com a quantidade solicitada de candles
  - Possui as colunas obrigatórias: Abertura, Maxima, Minima, Fechamento, Volume
  - Não contém valores nulos
  - Trata falhas de conexão de forma adequada (sem exceção não tratada)

Todos os testes são UNITÁRIOS com mocks — não dependem de internet.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from backend.coleta_dados import (
    buscar_dados_historicos,
    buscar_preco_atual,
    obter_cliente_binance,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _gerar_candles_mock(n: int = 90, preco_inicial: float = 90000.0) -> list:
    """Gera lista de candles OHLCV mock para simular resposta da Binance."""
    candles = []
    ts_base = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    preco = preco_inicial
    for i in range(n):
        abertura = preco
        maxima = preco * 1.02
        minima = preco * 0.98
        fechamento = preco * (1.001 if i % 2 == 0 else 0.999)
        volume = 100.0 + i
        ts = ts_base + i * 86400000  # +1 dia em ms
        candles.append([ts, abertura, maxima, minima, fechamento, volume])
        preco = fechamento
    return candles


@pytest.fixture
def df_mock_90():
    """DataFrame mock com 90 candles BTC/USDT 1d."""
    candles = _gerar_candles_mock(90)
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "Abertura", "Maxima", "Minima", "Fechamento", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# ── CT01.1: Quantidade correta de candles ───────────────────────────────────────

class TestCT01ColetaDados:

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct01_retorna_90_candles(self, mock_cliente):
        """CT01: DataFrame retornado deve ter exatamente 90 linhas."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _gerar_candles_mock(90)
        mock_cliente.return_value = mock_exchange

        df = buscar_dados_historicos("BTC/USDT", "1d", 90)

        assert len(df) == 90, (
            f"Esperado 90 candles, obtido {len(df)}. "
            "CT01 falhou: quantidade de candles incorreta."
        )

    # ── CT01.2: Colunas obrigatórias ────────────────────────────────────────────

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct01_colunas_obrigatorias(self, mock_cliente):
        """CT01: DataFrame deve conter as colunas OHLCV corretas."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _gerar_candles_mock(90)
        mock_cliente.return_value = mock_exchange

        df = buscar_dados_historicos("BTC/USDT", "1d", 90)

        colunas_esperadas = ["Abertura", "Maxima", "Minima", "Fechamento", "Volume"]
        for col in colunas_esperadas:
            assert col in df.columns, (
                f"CT01 falhou: coluna '{col}' ausente. "
                f"Colunas presentes: {list(df.columns)}"
            )

    # ── CT01.3: Ausência de valores nulos ───────────────────────────────────────

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct01_sem_valores_nulos(self, mock_cliente):
        """CT01: DataFrame não deve conter valores nulos nas colunas OHLCV."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _gerar_candles_mock(90)
        mock_cliente.return_value = mock_exchange

        df = buscar_dados_historicos("BTC/USDT", "1d", 90)

        colunas = ["Abertura", "Maxima", "Minima", "Fechamento", "Volume"]
        nulos = df[colunas].isnull().sum()
        assert nulos.sum() == 0, (
            f"CT01 falhou: valores nulos encontrados:\n{nulos[nulos > 0]}"
        )

    # ── CT01.4: Índice é datetime ────────────────────────────────────────────────

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct01_indice_datetime(self, mock_cliente):
        """CT01: Índice do DataFrame deve ser do tipo DatetimeIndex."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _gerar_candles_mock(90)
        mock_cliente.return_value = mock_exchange

        df = buscar_dados_historicos("BTC/USDT", "1d", 90)

        assert isinstance(df.index, pd.DatetimeIndex), (
            f"CT01 falhou: índice não é DatetimeIndex. Tipo: {type(df.index)}"
        )

    # ── CT01.5: Tratamento de falha de conexão ──────────────────────────────────

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct01_tratamento_falha_conexao(self, mock_cliente):
        """CT01: Falha de conexão deve levantar exceção com mensagem descritiva."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ConnectionError("Sem conexão com a Binance")
        mock_cliente.return_value = mock_exchange

        with pytest.raises(ConnectionError) as exc_info:
            buscar_dados_historicos("BTC/USDT", "1d", 90)

        mensagem = str(exc_info.value)
        assert "Binance" in mensagem or "conexão" in mensagem.lower(), (
            "CT01: exceção deve ter mensagem descritiva sobre falha de conexão"
        )

    # ── CT01.6: buscar_preco_atual retorna campos esperados ─────────────────────

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct01_preco_atual_campos(self, mock_cliente):
        """CT01: buscar_preco_atual deve retornar dict com campos obrigatórios."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {
            "last": 95000.0,
            "percentage": 1.5,
            "quoteVolume": 1234567890.0,
            "high": 96000.0,
            "low": 93000.0,
        }
        mock_cliente.return_value = mock_exchange

        resultado = buscar_preco_atual("BTC/USDT")

        campos_esperados = ["preco", "variacao_24h", "volume_24h", "maxima_24h", "minima_24h"]
        for campo in campos_esperados:
            assert campo in resultado, (
                f"CT01 falhou: campo '{campo}' ausente em buscar_preco_atual"
            )

    # ── CT01.7: Par customizado ──────────────────────────────────────────────────

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct01_par_customizado(self, mock_cliente):
        """CT01: O sistema deve aceitar qualquer par válido informado pelo usuário."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _gerar_candles_mock(50, preco_inicial=200.0)
        mock_cliente.return_value = mock_exchange

        # Testa com ETH/USDT (não apenas BTC)
        df = buscar_dados_historicos("ETH/USDT", "4h", 50)

        assert len(df) == 50, (
            f"CT01 falhou: par customizado ETH/USDT retornou {len(df)} candles, esperado 50"
        )

    # ── CT01.8: Cliente público/autenticado ──────────────────────────────────────

    def test_ct01_cliente_publico_sem_chaves(self, monkeypatch):
        """CT01: sem chaves, o cliente Binance deve ser criado em modo público."""
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_SECRET_KEY", raising=False)
        with patch("backend.coleta_dados.ccxt.binance") as mock_binance:
            obter_cliente_binance()
        config = mock_binance.call_args.args[0]
        assert config["apiKey"] is None
        assert config["secret"] is None
        assert config["enableRateLimit"] is True

    def test_ct01_cliente_autenticado_com_chaves_validas(self, monkeypatch):
        """CT01: com chaves válidas, elas devem ser repassadas ao ccxt."""
        monkeypatch.setenv("BINANCE_API_KEY", "chave_valida_1234567890")
        monkeypatch.setenv("BINANCE_SECRET_KEY", "segredo_valido_1234567890")
        with patch("backend.coleta_dados.ccxt.binance") as mock_binance:
            obter_cliente_binance()
        config = mock_binance.call_args.args[0]
        assert config["apiKey"] == "chave_valida_1234567890"
        assert config["secret"] == "segredo_valido_1234567890"
