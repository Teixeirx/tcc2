"""
tests/unit/test_webapp.py

Testes unitários para a interface web Flask (webapp):
  - Proteção de rotas com login_required (redirecionamento para /login)
  - Autenticação com credenciais inválidas (falha)
  - Autenticação com credenciais válidas (sucesso e sessão ativa)
  - Logout (encerramento de sessão)
  - Renderização das páginas / (dashboard), /analise, /historico e /backtesting
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from webapp.app import app
import webapp.app as webapp_app


@pytest.fixture
def client():
    """Cliente de teste do Flask com sessão configurada."""
    app.config["TESTING"] = True
    app.config["CSRF_ENABLED"] = False
    webapp_app._TENTATIVAS_LOGIN.clear()
    with app.test_client() as client:
        yield client
    webapp_app._TENTATIVAS_LOGIN.clear()


def _login(client):
    return client.post("/login", data={
        "usuario": "admin",
        "senha": "admin",
    })


def _df_ohlcv(n=90):
    datas = pd.date_range("2025-01-01", periods=n, freq="D")
    precos = pd.Series([50000.0 + i for i in range(n)], index=datas)
    return pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.01,
        "Minima": precos * 0.99,
        "Fechamento": precos,
        "Volume": 1000.0,
    }, index=datas)


def _relatorio_gemini_exemplo():
    return {
        "gerado_em": "2026-11-05T21:13:39+00:00",
        "modelo": "gemini-3.5-flash-lite",
        "passo_amostragem_dias": 7,
        "resultados": [{
            "par": "BTC/USDT",
            "intervalo": "1d",
            "total_amostras": 63,
            "distribuicao_acoes": {
                "COMPRAR": 38, "VENDER": 7, "MANTER": 18,
                "COMPRAR_pct": 60.3, "VENDER_pct": 11.1, "MANTER_pct": 28.6,
            },
            "acuracia_t1_pct": 48.9,
            "acuracia_prazo_estimado_pct": 62.2,
            "taxa_atingimento_alvo_pct": 39.5,
            "retorno_acumulado_pct": -30.97,
            "buy_and_hold": {"retorno_total_pct": -31.16},
            "taxa_concordancia_proxy_pct": 42.9,
        }],
    }


def _relatorio_gemini_intraday_exemplo():
    return {
        "gerado_em": "2026-11-06T02:00:00+00:00",
        "modelo": "gemini-3.5-flash-lite",
        "passo_amostragem_dias": 3,
        "periodo": {
            "inicio": "2026-07-01T00:00:00+00:00",
            "fim": "2026-10-31T23:59:59+00:00",
        },
        "resultados": [
            {
                "par": "BTC/USDT",
                "intervalo": "1h",
                "total_amostras": 40,
                "distribuicao_acoes": {
                    "COMPRAR": 19, "VENDER": 6, "MANTER": 15,
                    "COMPRAR_pct": 47.5, "VENDER_pct": 15.0, "MANTER_pct": 37.5,
                },
                "acuracia_t1_pct": 64.0,
                "acuracia_prazo_estimado_pct": 64.0,
                "taxa_atingimento_alvo_pct": 5.3,
                "retorno_acumulado_pct": -0.71,
                "buy_and_hold": {"retorno_total_pct": 23.0},
                "taxa_concordancia_proxy_pct": 37.5,
            },
            {
                "par": "ETH/USDT",
                "intervalo": "4h",
                "total_amostras": 36,
                "distribuicao_acoes": {
                    "COMPRAR": 17, "VENDER": 4, "MANTER": 15,
                    "COMPRAR_pct": 47.2, "VENDER_pct": 11.1, "MANTER_pct": 41.7,
                },
                "acuracia_t1_pct": 61.9,
                "acuracia_prazo_estimado_pct": 57.1,
                "taxa_atingimento_alvo_pct": 41.2,
                "retorno_acumulado_pct": 11.22,
                "buy_and_hold": {"retorno_total_pct": 17.31},
                "taxa_concordancia_proxy_pct": 55.6,
            },
        ],
    }


def _resultado_backtest():
    comum = {
        "data_inicio": "2025-03-01T00:00:00+00:00",
        "data_fim": "2025-06-01T00:00:00+00:00",
        "criptoia": {
            "retorno_total_pct": 3.25,
            "acuracia_direcional": 0.56,
            "max_drawdown_pct": -2.5,
            "n_operacoes": 10,
        },
        "buy_and_hold": {
            "retorno_total_pct": -4.0,
            "max_drawdown_pct": -8.0,
        },
        "supervisionado": {
            "retorno_total_pct": 1.5,
            "acuracia_direcional": 0.52,
            "max_drawdown_pct": -3.0,
            "n_operacoes": 12,
            "n_treino": 60,
            "n_teste": 40,
            "n_refits": 1,
            "seq_len": 8,
            "acuracia_baseline_maioria": 0.51,
        },
    }
    return {
        "par": "BTC/USDT",
        "intervalo": "1d",
        "periodo_comum": comum,
        "criptoia": comum["criptoia"],
        "buy_and_hold": comum["buy_and_hold"],
        "supervisionado": comum["supervisionado"],
    }


class TestWebappAuth:

    def test_rotas_protegidas_redirecionam_sem_login(self, client):
        """Acesso a rotas restritas sem autenticação deve redirecionar para /login."""
        for rota in ["/", "/analise", "/historico", "/backtesting"]:
            resp = client.get(rota)
            assert resp.status_code == 302, f"Rota {rota} não redirecionou"
            assert "/login" in resp.headers["Location"]

    def test_login_credenciais_invalidas(self, client):
        """Tentativa de login com senha errada deve falhar."""
        resp = client.post("/login", data={
            "usuario": "admin",
            "senha": "senha_errada_123"
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert "Credenciais inválidas" in resp.data.decode("utf-8")

    def test_login_sucesso(self, client):
        """Login com credenciais corretas deve autenticar e redirecionar para /."""
        resp = client.post("/login", data={
            "usuario": "admin",
            "senha": "admin"
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert "Painel de Controle CriptoIA" in resp.data.decode("utf-8")

    def test_acesso_rotas_autenticado(self, client):
        """Após autenticação, todas as páginas devem retornar 200 OK."""
        # Faz login
        client.post("/login", data={
            "usuario": "admin",
            "senha": "admin"
        })

        rotas = ["/", "/analise", "/historico", "/backtesting"]
        for rota in rotas:
            resp = client.get(rota)
            assert resp.status_code == 200, f"Rota {rota} falhou com status {resp.status_code}"

    def test_logout_encerra_sessao(self, client):
        """Logout deve limpar a sessão e redirecionar para /login."""
        # Login
        client.post("/login", data={"usuario": "admin", "senha": "admin"})

        # Logout
        resp = client.get("/logout", follow_redirects=True)
        assert resp.status_code == 200
        assert "Sessão encerrada com sucesso" in resp.data.decode("utf-8")

        # Tenta acessar rota restrita após logout
        resp_bloqueado = client.get("/")
        assert resp_bloqueado.status_code == 302

    def test_csrf_token_presente_no_formulario(self, client):
        """A página de login deve incluir o token CSRF."""
        resp = client.get("/login")
        assert resp.status_code == 200
        assert 'name="_csrf_token"' in resp.data.decode("utf-8")

    def test_csrf_bloqueia_post_sem_token(self, client):
        """POST sem token CSRF deve ser rejeitado com 400 quando habilitado."""
        app.config["CSRF_ENABLED"] = True
        try:
            resp = client.post("/login", data={
                "usuario": "admin",
                "senha": "admin",
            })
            assert resp.status_code == 400
        finally:
            app.config["CSRF_ENABLED"] = False

    def test_cabecalhos_de_seguranca(self, client):
        """Respostas devem incluir cabeçalhos básicos de segurança."""
        resp = client.get("/login")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"


class TestWebappBacktesting:

    def test_nao_exibe_resultados_academicos_fabricados(self, client):
        """A página de backtesting não deve conter os números fixos antigos."""
        client.post("/login", data={"usuario": "admin", "senha": "admin"})
        resp = client.get("/backtesting")
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Experimentos Acadêmicos Consolidados" not in html
        assert "32.8" not in html
        # 41.2% passou a existir legitimamente no Experimento 3 (alvo do ETH 4h);
        # a verificação textual de números isolados é frágil e foi removida.
        assert "Histórico de Backtesting" in html

    def test_metodologia_menciona_lstm_real_e_proxy(self, client):
        """A interface deve declarar que o LSTM é real e que CriptoIA é proxy."""
        client.post("/login", data={"usuario": "admin", "senha": "admin"})
        html = client.get("/backtesting").data.decode("utf-8")
        assert "LSTM" in html
        assert "proxy" in html.lower()
        assert "não reproduz as respostas multimodais do Gemini" in html

    def test_grade_tcc_real_e_renderizada(self, client):
        """O relatório 4x4 persistido deve aparecer na interface."""
        _login(client)
        relatorio = {
            "gerado_em": "2026-10-22T23:49:00",
            "resultados": [{
                "par": "BTC/USDT", "intervalo": "1d",
                "periodo_comum": _resultado_backtest()["periodo_comum"],
                "supervisionado": _resultado_backtest()["supervisionado"],
            }],
        }
        with patch("webapp.app._carregar_relatorio_tcc", return_value=relatorio), \
             patch("webapp.app.listar_resultados_backtest", return_value=[]):
            resp = client.get("/backtesting")
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Grade Completa do TCC" in html
        assert "3.25%" in html
        assert "BTC/USDT" in html

    def test_sem_grade_mostra_instrucao(self, client):
        """Sem relatório consolidado, a interface deve orientar como gerá-lo."""
        _login(client)
        with patch("webapp.app._carregar_relatorio_tcc", return_value=None), \
             patch("webapp.app.listar_resultados_backtest", return_value=[]):
            html = client.get("/backtesting").data.decode("utf-8")
        assert "grade completa ainda não foi gerada" in html

    def test_post_backtesting_sucesso(self, client):
        """POST válido deve chamar a comparação, salvar e exibir métricas."""
        _login(client)
        with patch("webapp.app.comparar_estrategias", return_value=_resultado_backtest()) as mock_cmp, \
             patch("webapp.app.salvar_resultado_backtest") as mock_salvar, \
             patch("webapp.app.listar_resultados_backtest", return_value=[]), \
             patch("webapp.app._carregar_relatorio_tcc", return_value=None):
            resp = client.post("/backtesting", data={
                "par": "BTC/USDT", "intervalo": "1d",
            })
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Resultado da Simulação" in html
        assert "3.25%" in html
        mock_cmp.assert_called_once_with(
            par="BTC/USDT", intervalo="1d", **webapp_app.LSTM_WEB_CONFIG
        )
        mock_salvar.assert_called_once()

    def test_post_backtesting_par_invalido(self, client):
        """Par inválido não deve executar a comparação."""
        _login(client)
        with patch("webapp.app.comparar_estrategias") as mock_cmp:
            resp = client.post("/backtesting", data={
                "par": "BTCUSDT", "intervalo": "1d",
            })
        assert resp.status_code == 200
        assert "Par inválido" in resp.data.decode("utf-8")
        mock_cmp.assert_not_called()

    def test_post_backtesting_timeframe_invalido(self, client):
        """Timeframe inválido não deve executar a comparação."""
        _login(client)
        with patch("webapp.app.comparar_estrategias") as mock_cmp:
            resp = client.post("/backtesting", data={
                "par": "BTC/USDT", "intervalo": "2h",
            })
        assert resp.status_code == 200
        assert "Timeframe inválido" in resp.data.decode("utf-8")
        mock_cmp.assert_not_called()

    def test_post_backtesting_erro_na_execucao(self, client):
        """Exceção na comparação deve virar mensagem de erro, sem traceback."""
        _login(client)
        with patch("webapp.app.comparar_estrategias",
                   side_effect=RuntimeError("falha simulada")), \
             patch("webapp.app.salvar_resultado_backtest") as mock_salvar, \
             patch("webapp.app.listar_resultados_backtest", return_value=[]), \
             patch("webapp.app._carregar_relatorio_tcc", return_value=None):
            resp = client.post("/backtesting", data={
                "par": "BTC/USDT", "intervalo": "1d",
            })
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Erro na simulação de backtesting" in html
        assert "falha simulada" in html
        assert mock_salvar.call_args.args[0]["erro"] == "falha simulada"

    def test_experimento_gemini_e_renderizado(self, client):
        """O relatório do Gemini deve aparecer com seus números reais."""
        _login(client)
        relatorio = _relatorio_gemini_exemplo()
        with patch("webapp.app._carregar_relatorio_gemini", return_value=relatorio), \
             patch("webapp.app._carregar_relatorio_gemini_intraday", return_value=None), \
             patch("webapp.app._carregar_relatorio_tcc", return_value=None), \
             patch("webapp.app.listar_resultados_backtest", return_value=[]):
            html = client.get("/backtesting").data.decode("utf-8")
        assert "Experimento 2" in html
        assert "Gemini" in html
        assert "62.2%" in html
        assert "-30.97%" in html
        assert "Critério 1" in html

    def test_experimento_gemini_intraday_e_renderizado(self, client):
        """O complemento 1h/4h deve aparecer separado do resultado oficial."""
        _login(client)
        relatorio = _relatorio_gemini_intraday_exemplo()
        with patch("webapp.app._carregar_relatorio_gemini", return_value=None), \
             patch("webapp.app._carregar_relatorio_gemini_intraday", return_value=relatorio), \
             patch("webapp.app._carregar_relatorio_tcc", return_value=None), \
             patch("webapp.app.listar_resultados_backtest", return_value=[]):
            html = client.get("/backtesting").data.decode("utf-8")
        assert "Experimento 3" in html
        assert "Complemento Exploratório" in html
        assert "1h" in html
        assert "4h" in html
        assert "2/2" in html
        assert "1/2" in html
        assert "-0.71%" in html
        assert "+11.22%" in html
        assert "não substitui o resultado oficial" in html

    def test_carregar_relatorio_gemini(self, tmp_path):
        caminho = tmp_path / "gemini.json"
        caminho.write_text(json.dumps(_relatorio_gemini_exemplo()), encoding="utf-8")
        with patch("webapp.app.CAMINHO_RELATORIO_GEMINI", str(caminho)):
            assert webapp_app._carregar_relatorio_gemini()["modelo"] == "gemini-3.5-flash-lite"

        caminho.write_text("{invalido", encoding="utf-8")
        with patch("webapp.app.CAMINHO_RELATORIO_GEMINI", str(caminho)):
            assert webapp_app._carregar_relatorio_gemini() is None

    def test_resumo_gemini_criterios(self):
        relatorio = _relatorio_gemini_exemplo()
        resumo = webapp_app._resumo_gemini(relatorio)
        assert resumo["acuracia_ok"] == 1
        assert resumo["retorno_ok"] == 0
        assert resumo["criterio_acuracia"] is False
        assert resumo["criterio_retorno"] is False
        assert webapp_app._resumo_gemini(None) is None

    def test_carregar_relatorio_gemini_intraday(self, tmp_path):
        caminho = tmp_path / "gemini_intraday.json"
        caminho.write_text(
            json.dumps(_relatorio_gemini_intraday_exemplo()),
            encoding="utf-8",
        )
        with patch("webapp.app.CAMINHO_RELATORIO_GEMINI_INTRADAY", str(caminho)):
            carregado = webapp_app._carregar_relatorio_gemini_intraday()
            assert carregado["passo_amostragem_dias"] == 3
            assert len(carregado["resultados"]) == 2

        caminho.write_text("{invalido", encoding="utf-8")
        with patch("webapp.app.CAMINHO_RELATORIO_GEMINI_INTRADAY", str(caminho)):
            assert webapp_app._carregar_relatorio_gemini_intraday() is None

    def test_resumo_gemini_intraday(self):
        relatorio = _relatorio_gemini_intraday_exemplo()
        resumo = webapp_app._resumo_gemini_intraday(relatorio)

        assert resumo["total"] == 2
        assert resumo["acuracia_ok"] == 2
        assert resumo["retorno_ok"] == 1
        assert resumo["bh_ok"] == 0
        por_tf = {item["intervalo"]: item for item in resumo["por_timeframe"]}
        assert por_tf["1h"]["acuracia_ok"] == 1
        assert por_tf["4h"]["retorno_ok"] == 1
        assert webapp_app._resumo_gemini_intraday(None) is None


class TestWebappAnalise:

    def test_post_analise_sucesso(self, client):
        """POST válido deve executar o pipeline, salvar e exibir o resultado."""
        _login(client)
        df = _df_ohlcv()
        indicadores = {
            "preco_fechamento": 95000.0, "mm20": 94000.0, "mm50": 91000.0,
            "rsi14": 58.5, "macd": 120.0, "sinal_macd": 95.0,
            "bollinger_superior": 98000.0, "bollinger_inferior": 90000.0,
        }
        analise_ia = {
            "acao": "COMPRAR", "justificativa": "Tendência de alta",
            "preco_alvo": "$100.000", "tempo_estimado": "7 dias",
            "fallback": False,
        }
        with patch("webapp.app.buscar_dados_historicos", return_value=df) as mock_dados, \
             patch("webapp.app.buscar_preco_atual", return_value={
                 "preco": 95000.0, "variacao_24h": 1.5, "volume_24h": 1000.0,
                 "maxima_24h": 96000.0, "minima_24h": 94000.0,
             }), \
             patch("webapp.app.enriquecer_dataframe", return_value=df), \
             patch("webapp.app.resumir_indicadores", return_value=indicadores), \
             patch("webapp.app.gerar_grafico_candlestick"), \
             patch("webapp.app.analisar_mercado_com_gemini", return_value=analise_ia), \
             patch("webapp.app._gerar_grafico_plotly", return_value="<div>grafico</div>"), \
             patch("webapp.app.salvar_analise") as mock_salvar:
            resp = client.post("/analise", data={
                "par": "BTC/USDT", "intervalo": "1d", "quantidade": "90",
            })
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Tendência de alta" in html
        assert "COMPRAR" in html
        mock_dados.assert_called_once_with("BTC/USDT", "1d", 90)
        mock_salvar.assert_called_once()
        assert mock_salvar.call_args.args[0]["acao"] == "COMPRAR"

    def test_post_analise_par_invalido(self, client):
        """Par inválido deve ser rejeitado sem chamar a Binance."""
        _login(client)
        with patch("webapp.app.buscar_dados_historicos") as mock_dados:
            resp = client.post("/analise", data={"par": "BTCUSDT", "intervalo": "1d"})
        assert resp.status_code == 200
        assert "Par inválido" in resp.data.decode("utf-8")
        mock_dados.assert_not_called()

    def test_post_analise_intervalo_invalido(self, client):
        """Timeframe inválido deve ser rejeitado sem chamar a Binance."""
        _login(client)
        with patch("webapp.app.buscar_dados_historicos") as mock_dados:
            resp = client.post("/analise", data={"par": "BTC/USDT", "intervalo": "2h"})
        assert resp.status_code == 200
        assert "Timeframe inválido" in resp.data.decode("utf-8")
        mock_dados.assert_not_called()

    def test_post_analise_falha_api(self, client):
        """Falha de API deve virar erro tratado e fallback MANTER no histórico."""
        _login(client)
        with patch("webapp.app.buscar_dados_historicos",
                   side_effect=ConnectionError("sem conexão")), \
             patch("webapp.app.salvar_analise") as mock_salvar:
            resp = client.post("/analise", data={
                "par": "BTC/USDT", "intervalo": "1d", "quantidade": "90",
            })
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Erro ao executar análise" in html
        assert "sem conexão" in html
        registro = mock_salvar.call_args.args[0]
        assert registro["acao"] == "MANTER"
        assert registro["fallback"] is True

    def test_quantidade_invalida_usa_padrao(self, client):
        """Quantidade não numérica deve cair no padrão de 90."""
        _login(client)
        with patch("webapp.app.buscar_dados_historicos",
                   side_effect=RuntimeError("parar")) as mock_dados, \
             patch("webapp.app.salvar_analise"):
            client.post("/analise", data={
                "par": "BTC/USDT", "intervalo": "1d", "quantidade": "abc",
            })
        mock_dados.assert_called_once_with("BTC/USDT", "1d", 90)

    def test_falha_ao_salvar_historico_nao_invalida_analise(self, client):
        """Falha de persistência não deve desfazer uma análise concluída."""
        _login(client)
        df = _df_ohlcv()
        with patch("webapp.app.buscar_dados_historicos", return_value=df), \
             patch("webapp.app.buscar_preco_atual", return_value={
                 "preco": 95000.0, "variacao_24h": 1.0,
             }), \
             patch("webapp.app.enriquecer_dataframe", return_value=df), \
             patch("webapp.app.resumir_indicadores", return_value={}), \
             patch("webapp.app.gerar_grafico_candlestick"), \
             patch("webapp.app.analisar_mercado_com_gemini", return_value={
                 "acao": "MANTER", "justificativa": "Teste",
                 "preco_alvo": None, "tempo_estimado": None, "fallback": False,
             }), \
             patch("webapp.app._gerar_grafico_plotly", return_value="<div></div>"), \
             patch("webapp.app.salvar_analise", side_effect=OSError("disco cheio")):
            resp = client.post("/analise", data={
                "par": "BTC/USDT", "intervalo": "1d", "quantidade": "90",
            })
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "não foi possível salvá-la no histórico" in html


class TestLoginRateLimit:

    def test_bloqueia_apos_cinco_falhas(self, client):
        """Após 5 falhas no mesmo IP, novas tentativas devem retornar 429."""
        for _ in range(5):
            resp = client.post("/login", data={
                "usuario": "admin", "senha": "errada",
            })
            assert resp.status_code == 200
        resp = client.post("/login", data={
            "usuario": "admin", "senha": "admin",
        })
        assert resp.status_code == 429
        assert "Muitas tentativas" in resp.data.decode("utf-8")


class TestWebappHistoricoDashboard:

    def test_dashboard_conta_historico(self, client):
        """O dashboard deve exibir contagens reais dos históricos."""
        _login(client)
        analise = {
            "data": "22/10/2026 12:00:00", "par": "BTC/USDT",
            "intervalo": "1d", "preco_atual": 95000.0,
            "acao": "MANTER", "preco_alvo": None,
            "justificativa": "Teste",
        }
        with patch("webapp.app.listar_analises", return_value=[analise] * 3), \
             patch("webapp.app.listar_resultados_backtest", return_value=[{}] * 2):
            html = client.get("/").data.decode("utf-8")
        assert "3" in html
        assert "2" in html

    def test_historico_renderiza_analises(self, client):
        """A página de histórico deve renderizar os registros persistidos."""
        _login(client)
        analise = {
            "id": 1, "data": "22/10/2026 12:00:00", "par": "ETH/USDT",
            "intervalo": "4h", "preco_atual": 3000.0, "variacao_24h": -1.2,
            "acao": "VENDER", "preco_alvo": "$2.800", "tempo_estimado": "5 dias",
            "justificativa": "Teste", "fallback": False,
        }
        with patch("webapp.app.listar_analises", return_value=[analise]):
            html = client.get("/historico").data.decode("utf-8")
        assert "ETH/USDT" in html
        assert "VENDER" in html


class TestRelatorioTcc:

    def test_carregar_relatorio_valido(self, tmp_path, monkeypatch):
        caminho = tmp_path / "comparacao.json"
        caminho.write_text(json.dumps({"resultados": [{"par": "BTC/USDT"}]}),
                           encoding="utf-8")
        monkeypatch.setattr(webapp_app, "CAMINHO_RELATORIO_TCC", str(caminho))
        dados = webapp_app._carregar_relatorio_tcc()
        assert dados["resultados"][0]["par"] == "BTC/USDT"

    def test_carregar_relatorio_ausente(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webapp_app, "CAMINHO_RELATORIO_TCC",
                            str(tmp_path / "nao_existe.json"))
        assert webapp_app._carregar_relatorio_tcc() is None

    def test_carregar_relatorio_invalido(self, tmp_path, monkeypatch):
        caminho = tmp_path / "invalido.json"
        caminho.write_text("{invalido", encoding="utf-8")
        monkeypatch.setattr(webapp_app, "CAMINHO_RELATORIO_TCC", str(caminho))
        assert webapp_app._carregar_relatorio_tcc() is None

    def test_carregar_relatorio_sem_lista(self, tmp_path, monkeypatch):
        caminho = tmp_path / "sem_lista.json"
        caminho.write_text(json.dumps({"resultados": "nao é lista"}), encoding="utf-8")
        monkeypatch.setattr(webapp_app, "CAMINHO_RELATORIO_TCC", str(caminho))
        assert webapp_app._carregar_relatorio_tcc() is None


class TestCsrfValido:

    def test_post_com_token_valido(self, client):
        """Com token de sessão válido, o POST deve ser aceito."""
        app.config["CSRF_ENABLED"] = True
        try:
            client.get("/login")
            with client.session_transaction() as sessao:
                token = sessao["_csrf_token"]
            resp = client.post("/login", data={
                "usuario": "admin", "senha": "admin",
                "_csrf_token": token,
            })
            assert resp.status_code == 302
        finally:
            app.config["CSRF_ENABLED"] = False


def test_auth_rejeita_placeholder_em_producao():
    """Em produção, placeholder de senha não pode iniciar a aplicação."""
    raiz = Path(__file__).resolve().parents[2]
    ambiente = os.environ.copy()
    ambiente["WEB_ADMIN_PASS"] = "troque-por-uma-senha-forte"
    ambiente["CRIPTOIA_ENV"] = "production"
    ambiente["PYTHONPATH"] = str(raiz)

    resultado = subprocess.run(
        [sys.executable, "-c", "import webapp.auth"],
        cwd=raiz,
        env=ambiente,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert resultado.returncode != 0
    assert "WEB_ADMIN_PASS" in (resultado.stderr + resultado.stdout)
