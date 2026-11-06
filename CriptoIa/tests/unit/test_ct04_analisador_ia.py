"""
tests/unit/test_ct04_analisador_ia.py

CT04 — RF05: Análise por IA Gemini
====================================
Verifica que o módulo analisador_ia.py:
  - Retorna JSON com os campos obrigatórios: acao, justificativa,
    preco_alvo e tempo_estimado
  - acao é sempre um de: COMPRAR, VENDER, MANTER
  - justificativa não é vazia
  - Lida corretamente com JSON inválido (fallback)
  - Lida com campos ausentes (fallback)
  - Lida com valores inválidos de acao (normalização para MANTER)
  - Lida com timeout e falha de conexão (fallback estruturado)
  - Retorna fallback com acao="MANTER" quando API key ausente

Todos os testes são UNITÁRIOS com mocks — não fazem chamadas reais ao Gemini.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from backend.analisador_ia import (
    analisar_mercado_com_gemini,
    _extrair_json_da_resposta,
    _construir_prompt,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def indicadores_validos():
    """Dicionário de indicadores técnicos válidos para teste."""
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
def resposta_json_valida():
    """Resposta JSON válida simulando o Gemini."""
    return json.dumps({
        "acao": "COMPRAR",
        "justificativa": "RSI está em zona neutra com MACD positivo. "
                         "MM20 acima da MM50 indica tendência de alta.",
        "preco_alvo": "$98.500,00",
        "tempo_estimado": "7 a 14 dias",
    })


# ── CT04.1: JSON válido retorna campos obrigatórios ────────────────────────────

class TestCT04AnalisadorIA:

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake_para_teste")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct04_json_valido_campos_obrigatorios(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos, resposta_json_valida
    ):
        """CT04: Resposta JSON válida deve conter todos os campos obrigatórios."""
        mock_resposta = MagicMock()
        mock_resposta.text = resposta_json_valida
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.return_value = mock_resposta
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        campos = ["acao", "justificativa", "preco_alvo", "tempo_estimado"]
        for campo in campos:
            assert campo in resultado, (
                f"CT04 falhou: campo '{campo}' ausente na resposta"
            )

    # ── CT04.2: acao é valor válido ─────────────────────────────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake_para_teste")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct04_acao_valor_valido(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos, resposta_json_valida
    ):
        """CT04: campo 'acao' deve ser COMPRAR, VENDER ou MANTER."""
        mock_resposta = MagicMock()
        mock_resposta.text = resposta_json_valida
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.return_value = mock_resposta
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        assert resultado["acao"] in {"COMPRAR", "VENDER", "MANTER"}, (
            f"CT04 falhou: acao='{resultado['acao']}' não é um valor válido"
        )

    # ── CT04.3: justificativa não vazia ─────────────────────────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake_para_teste")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct04_justificativa_nao_vazia(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos, resposta_json_valida
    ):
        """CT04: campo 'justificativa' não deve ser vazio."""
        mock_resposta = MagicMock()
        mock_resposta.text = resposta_json_valida
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.return_value = mock_resposta
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        justif = resultado.get("justificativa", "")
        assert len(justif) > 10, (
            f"CT04 falhou: justificativa vazia ou muito curta: '{justif}'"
        )

    # ── CT04.4: JSON inválido retorna fallback ──────────────────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake_para_teste")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct04_json_invalido_fallback_manter(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos
    ):
        """CT04: JSON inválido deve retornar fallback com acao='MANTER'."""
        mock_resposta = MagicMock()
        mock_resposta.text = "Isso não é JSON válido { sem fechamento"
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.return_value = mock_resposta
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        assert resultado["acao"] == "MANTER", (
            f"CT04 falhou: JSON inválido não retornou fallback MANTER. acao='{resultado['acao']}'"
        )

    # ── CT04.5: acao inválida é normalizada para MANTER ─────────────────────────

    def test_ct04_acao_invalida_normalizada(self):
        """CT04: acao com valor inválido deve ser normalizada para MANTER."""
        json_invalido = json.dumps({
            "acao": "COMPRAR_TUDO",
            "justificativa": "Análise teste",
            "preco_alvo": None,
            "tempo_estimado": None,
        })
        resultado = _extrair_json_da_resposta(json_invalido)
        assert resultado["acao"] == "MANTER", (
            f"CT04 falhou: acao inválida 'COMPRAR_TUDO' não foi normalizada. "
            f"acao='{resultado['acao']}'"
        )

    # ── CT04.6: API key ausente retorna fallback ─────────────────────────────────

    @patch("backend.analisador_ia.os.getenv", return_value="")
    def test_ct04_sem_api_key_retorna_manter(self, mock_getenv, indicadores_validos):
        """CT04: Sem API key, deve retornar fallback MANTER com justificativa."""
        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )
        assert resultado.get("acao") == "MANTER", (
            f"CT04: sem API key, esperado acao='MANTER', obtido '{resultado.get('acao')}'"
        )
        assert resultado.get("fallback") is True
        assert resultado.get("justificativa"), (
            "CT04: justificativa deve ser não vazia mesmo em fallback"
        )

    # ── CT04.7: Timeout/falha de conexão retorna fallback estruturado ───────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake_para_teste")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct04_falha_conexao_retorna_fallback(
        self, mock_img_open, mock_genai_client, mock_getenv,
        indicadores_validos
    ):
        """CT04: Falha de conexão deve retornar fallback estruturado MANTER."""
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.side_effect = ConnectionError(
            "Timeout ao conectar com o Gemini"
        )
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        assert resultado.get("acao") == "MANTER", (
            f"CT04: falha de conexão deveria retornar acao='MANTER', "
            f"obtido '{resultado.get('acao')}'"
        )
        assert resultado.get("fallback") is True
        campos = ["acao", "justificativa", "preco_alvo", "tempo_estimado"]
        for campo in campos:
            assert campo in resultado, (
                f"CT04: campo '{campo}' ausente no fallback de erro"
            )

    # ── CT04.8: JSON em bloco Markdown é extraído corretamente ──────────────────

    def test_ct04_extrai_json_de_bloco_markdown(self):
        """CT04: JSON dentro de bloco ```json deve ser extraído corretamente."""
        texto_com_markdown = """```json
{
  "acao": "VENDER",
  "justificativa": "RSI sobrecomprado e MACD cruzando abaixo da linha de sinal.",
  "preco_alvo": "$88.000,00",
  "tempo_estimado": "3 a 7 dias"
}
```"""
        resultado = _extrair_json_da_resposta(texto_com_markdown)
        assert resultado["acao"] == "VENDER", (
            f"CT04: JSON em Markdown não foi extraído. acao='{resultado.get('acao')}'"
        )

    # ── CT04.9: _construir_prompt tem formato correto ───────────────────────────

    def test_ct04_prompt_contem_campos_obrigatorios(self, indicadores_validos):
        """CT04: O prompt gerado deve conter os campos dos indicadores."""
        prompt = _construir_prompt(indicadores_validos, par="ETH/USDT")
        assert "COMPRAR" in prompt or "VENDER" in prompt or "MANTER" in prompt, (
            "CT04: prompt deve mencionar as ações possíveis"
        )
        assert "JSON" in prompt.upper(), (
            "CT04: prompt deve instruir o modelo a responder em JSON"
        )
        assert "RSI" in prompt, "CT04: prompt deve conter RSI"
        assert "MACD" in prompt, "CT04: prompt deve conter MACD"
        assert "ETH/USDT" in prompt, "CT04: prompt deve identificar o par analisado"

    # ── CT04.10: Campo obrigatório ausente é rejeitado ─────────────────────────

    @pytest.mark.parametrize("campo", ["justificativa", "preco_alvo", "tempo_estimado"])
    def test_ct04_campo_ausente_levanta_erro(self, campo):
        """CT04: JSON sem campo obrigatório deve ser rejeitado (fallback no chamador)."""
        dados = {
            "acao": "COMPRAR",
            "justificativa": "Justificativa válida para o teste.",
            "preco_alvo": "$100.000,00",
            "tempo_estimado": "7 dias",
        }
        dados.pop(campo)
        with pytest.raises(ValueError, match=campo):
            _extrair_json_da_resposta(json.dumps(dados))

    # ── CT04.11: Justificativa vazia é rejeitada ───────────────────────────────

    def test_ct04_justificativa_vazia_levanta_erro(self):
        """CT04: justificativa vazia não é uma resposta válida."""
        dados = {
            "acao": "MANTER",
            "justificativa": "   ",
            "preco_alvo": None,
            "tempo_estimado": None,
        }
        with pytest.raises(ValueError, match="justificativa"):
            _extrair_json_da_resposta(json.dumps(dados))

    # ── CT04.12: COMPRAR/VENDER exigem alvo e prazo ────────────────────────────

    @pytest.mark.parametrize("campo", ["preco_alvo", "tempo_estimado"])
    def test_ct04_compra_venda_exige_alvo_e_prazo(self, campo):
        """CT04: COMPRAR/VENDER sem alvo ou prazo deve ser rejeitado."""
        dados = {
            "acao": "COMPRAR",
            "justificativa": "Justificativa válida para o teste.",
            "preco_alvo": "$100.000,00",
            "tempo_estimado": "7 dias",
        }
        dados[campo] = None
        with pytest.raises(ValueError, match=campo):
            _extrair_json_da_resposta(json.dumps(dados))

    # ── CT04.13: Resposta da API com campo ausente cai em MANTER ───────────────

    @patch("backend.analisador_ia.os.getenv", return_value="chave_fake_para_teste")
    @patch("backend.analisador_ia.genai.Client")
    @patch("backend.analisador_ia.PIL.Image.open")
    def test_ct04_api_campo_ausente_fallback_manter(
        self, mock_img_open, mock_genai_client, mock_getenv, indicadores_validos
    ):
        """CT04: resposta da API sem justificativa deve virar fallback MANTER."""
        mock_resposta = MagicMock()
        mock_resposta.text = json.dumps({
            "acao": "COMPRAR",
            "preco_alvo": "$100.000,00",
            "tempo_estimado": "7 dias",
        })
        mock_cliente = MagicMock()
        mock_cliente.models.generate_content.return_value = mock_resposta
        mock_genai_client.return_value = mock_cliente

        resultado = analisar_mercado_com_gemini(
            "grafico_fake.png", indicadores_validos
        )

        assert resultado["acao"] == "MANTER"
        assert resultado["fallback"] is True
        assert resultado["justificativa"]
