"""
tests/unit/test_ct06_tratamento_falhas.py

CT06 — RNF06: Tratamento de Falhas de Conexão e API
====================================================
Verifica que o sistema:
  - Exibe mensagem de erro descritiva em caso de falha
  - Não encerra com exceção não tratada (traceback)
  - Retorna fallback JSON estruturado com acao="MANTER" em falha de API
  - Lida com múltiplos cenários de falha de forma resiliente

Todos os testes são UNITÁRIOS com mocks — não fazem chamadas reais.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.analisador_ia import analisar_mercado_com_gemini
from backend.coleta_dados import buscar_dados_historicos, buscar_preco_atual


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def indicadores_validos():
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


# ── CT06: Tratamento de falhas ─────────────────────────────────────────────────

class TestCT06TratamentoFalhas:

    # ── CT06.1: Falha de conexão API Gemini ─────────────────────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct06_falha_conexao_gemini_nao_levanta_excecao(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos
    ):
        """CT06: Falha na API Gemini não deve levantar exceção não tratada."""
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.side_effect = ConnectionError(
            "Network unreachable"
        )
        mock_genai_client.return_value = mock_cliente

        # Não deve levantar exceção
        try:
            resultado = analisar_mercado_com_gemini(
                "grafico_fake.png", indicadores_validos
            )
        except Exception as e:
            pytest.fail(
                f"CT06 falhou: exceção não tratada levantada: {type(e).__name__}: {e}"
            )

    # ── CT06.2: Fallback tem acao="MANTER" em falha ────────────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct06_fallback_estruturado_em_falha(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos
    ):
        """CT06: Em falha de API, retornar dict estruturado com acao='MANTER'."""
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.side_effect = RuntimeError(
            "API quota exceeded"
        )
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        assert isinstance(resultado, dict), (
            "CT06 falhou: retorno não é dict estruturado"
        )
        assert resultado.get("acao") == "MANTER", (
            f"CT06 falhou: acao='{resultado.get('acao')}' em falha de API"
        )
        assert resultado.get("fallback") is True

    # ── CT06.3: Fallback tem mensagem de erro descritiva ────────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct06_fallback_tem_justificativa_descritiva(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos
    ):
        """CT06: Fallback deve ter justificativa com informação sobre o erro."""
        mensagem_erro = "Connection timeout after 30 seconds"
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.side_effect = TimeoutError(mensagem_erro)
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        justif = resultado.get("justificativa", "")
        assert len(justif) > 0, (
            "CT06 falhou: justificativa vazia no fallback de erro"
        )

    # ── CT06.4: Falha de conexão Binance ────────────────────────────────────────

    @patch("backend.coleta_dados.obter_cliente_binance")
    def test_ct06_falha_binance_levanta_excecao_descritiva(self, mock_cliente):
        """CT06: Falha na Binance deve levantar exceção com mensagem descritiva."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = Exception(
            "Failed to connect to Binance"
        )
        mock_cliente.return_value = mock_exchange

        with pytest.raises(Exception) as exc_info:
            buscar_dados_historicos("BTC/USDT", "1d", 90)

        # A exceção deve ter mensagem (não ser genérica sem informação)
        assert str(exc_info.value), (
            "CT06 falhou: exceção da Binance não tem mensagem descritiva"
        )

    # ── CT06.5: Chave API inválida retorna fallback (não exceção) ───────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_invalida_12345")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct06_chave_invalida_retorna_fallback(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos
    ):
        """CT06: Chave de API inválida deve retornar fallback, não exceção."""
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.side_effect = PermissionError(
            "Invalid API key"
        )
        mock_genai_client.return_value = mock_cliente

        try:
            resultado = analisar_mercado_com_gemini(
                "grafico_fake.png", indicadores_validos
            )
            assert isinstance(resultado, dict), (
                "CT06: retorno deve ser dict, mesmo com chave inválida"
            )
        except Exception as e:
            pytest.fail(
                f"CT06 falhou: exceção não tratada com chave inválida: {e}"
            )

    # ── CT06.6: Sem chave API, fallback com acao="MANTER" ───────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="")
    def test_ct06_sem_chave_api_fallback_estruturado(
        self, mock_getenv, indicadores_validos
    ):
        """CT06: Sem chave API configurada, retornar fallback estruturado MANTER."""
        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )
        assert isinstance(resultado, dict), "CT06: retorno deve ser dict"
        assert resultado.get("acao") == "MANTER", (
            f"CT06: esperado MANTER, obtido {resultado.get('acao')}"
        )
        assert resultado.get("fallback") is True
        # Todos os campos obrigatórios devem estar presentes
        for campo in ["acao", "justificativa", "preco_alvo", "tempo_estimado"]:
            assert campo in resultado, (
                f"CT06 falhou: campo '{campo}' ausente no fallback sem chave API"
            )

    # ── CT06.7: Múltiplos ativos — erro individual não interrompe os demais ─────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct06_erro_individual_nao_interrompe_analise(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos
    ):
        """CT06: Erro em um par não deve impedir análise dos demais."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from backend.multiplos_ativos import analisar_multiplos
        from unittest.mock import patch as patch2

        # Simula coleta de dados que falha para o primeiro par
        def analisar_par_mock(par, intervalo, quantidade, gerar_grafico, usar_ia):
            if par == "BTC/USDT":
                raise RuntimeError("Erro simulado para BTC/USDT")
            return {
                "par": par,
                "intervalo": intervalo,
                "indicadores": indicadores_validos,
                "analise_ia": {"acao": "MANTER", "justificativa": "OK"},
            }

        combinacoes = [
            {"par": "BTC/USDT", "intervalo": "1d", "quantidade": 90},
            {"par": "ETH/USDT", "intervalo": "1d", "quantidade": 90},
        ]

        with patch2("backend.multiplos_ativos.analisar_par", side_effect=analisar_par_mock):
            resultados = analisar_multiplos(combinacoes, max_workers=2)

        # Deve ter resultado para os dois pares (mesmo que um com erro)
        assert len(resultados) == 2, (
            f"CT06 falhou: esperado 2 resultados, obtido {len(resultados)}"
        )
