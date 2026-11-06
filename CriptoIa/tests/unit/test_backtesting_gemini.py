"""
test_backtesting_gemini.py — Testes unitários para o módulo de backtesting multimodal Gemini.
"""

import os
import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from backend.backtesting_gemini import (
    converter_passo_para_candles,
    extrair_janela_dias,
    extrair_preco_alvo,
    carregar_cache_gemini,
    salvar_cache_gemini,
    calcular_buy_and_hold,
    calcular_sinal_proxy_ponto,
    executar_backtest_gemini_par,
    executar_backtest_gemini_grade,
)


def test_converter_passo_para_candles():
    assert converter_passo_para_candles("1d", 7) == 7
    assert converter_passo_para_candles("4h", 7) == 42
    assert converter_passo_para_candles("1h", 7) == 168
    assert converter_passo_para_candles("1w", 7) == 1
    with pytest.raises(ValueError):
        converter_passo_para_candles("1d", 0)
    with pytest.raises(ValueError):
        converter_passo_para_candles("5m", 7)


def test_extrair_janela_dias():
    assert extrair_janela_dias("7 a 14 dias") == (7, 14)
    assert extrair_janela_dias("5-10 dias") == (5, 10)
    assert extrair_janela_dias("2 a 4 semanas") == (14, 28)
    assert extrair_janela_dias("1 semana") == (1, 7)
    assert extrair_janela_dias("24 a 48 horas") == (1, 2)
    assert extrair_janela_dias("10 dias") == (5, 10)
    assert extrair_janela_dias(None) == (7, 14)
    assert extrair_janela_dias("indefinido") == (7, 14)


def test_extrair_preco_alvo():
    ref = 90000.0
    assert extrair_preco_alvo("$98.500,00", ref) == 98500.0
    assert extrair_preco_alvo("$98,500.00", ref) == 98500.0
    assert extrair_preco_alvo("95000", ref) == 95000.0
    assert extrair_preco_alvo("+5%", ref) == 94500.0
    assert extrair_preco_alvo(None, ref) is None
    assert extrair_preco_alvo("invalido", ref) is None
    assert extrair_preco_alvo("$1.00", ref) is None  # Fora dos limites razoáveis


def test_cache_gemini():
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho = os.path.join(tmpdir, "cache.json")
        dados = {"teste_chave": {"acao": "COMPRAR"}}
        salvar_cache_gemini(dados, caminho)
        carregado = carregar_cache_gemini(caminho)
        assert carregado == dados


def test_calcular_sinal_proxy_ponto():
    df_curto = pd.DataFrame({"Fechamento": [10.0] * 20})
    assert calcular_sinal_proxy_ponto(df_curto) == "MANTER"

    # Amostra completa simulando compra
    df_compra = pd.DataFrame({
        "Fechamento": [100.0] * 60,
        "MM20": [105.0] * 60,
        "MM50": [95.0] * 60,
        "RSI14": [50.0] * 60,
        "MACD": [2.0] * 60,
        "Sinal_MACD": [1.0] * 60,
    })
    assert calcular_sinal_proxy_ponto(df_compra) == "COMPRAR"


def test_executar_backtest_gemini_par_mock():
    # Cria DataFrame sintético com 150 candles
    datas = pd.date_range("2025-01-01", periods=150, freq="D", tz="UTC")
    precos = np.linspace(50000, 60000, 150)
    df_mock = pd.DataFrame({
        "Abertura": precos - 100,
        "Maxima": precos + 500,
        "Minima": precos - 500,
        "Fechamento": precos,
        "Volume": [1000.0] * 150,
    }, index=datas)

    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_cache = os.path.join(tmpdir, "cache.json")

        with patch("backend.backtesting_gemini.buscar_dados_historicos_paginado", return_value=df_mock), \
             patch("backend.backtesting_gemini.gerar_grafico_candlestick", return_value="fake.png"), \
             patch("backend.backtesting_gemini.analisar_mercado_com_gemini") as mock_ia, \
             patch("os.path.exists", return_value=False):

            mock_ia.return_value = {
                "acao": "COMPRAR",
                "justificativa": "Tendência de alta consistente.",
                "preco_alvo": "$58.000,00",
                "tempo_estimado": "7 a 14 dias",
            }

            resultado = executar_backtest_gemini_par(
                par="BTC/USDT",
                intervalo="1d",
                passo_amostragem=10,
                max_amostras=3,
                delay_segundos=0.0,
                caminho_cache=caminho_cache,
            )

            assert "erro" not in resultado
            assert resultado["par"] == "BTC/USDT"
            assert resultado["total_amostras"] == 3
            assert resultado["distribuicao_acoes"]["COMPRAR"] == 3
            assert resultado["acuracia_t1_pct"] is not None
            assert len(resultado["registros"]) == 3
            assert resultado["buy_and_hold"]["estrategia"] == "buy_and_hold"
            assert resultado["buy_and_hold"]["retorno_total_pct"] is not None


def test_executar_backtest_gemini_par_periodo_customizado():
    datas = pd.date_range("2025-01-01", periods=150, freq="D", tz="UTC")
    precos = np.linspace(50000, 60000, 150)
    df_mock = pd.DataFrame({
        "Abertura": precos - 100,
        "Maxima": precos + 500,
        "Minima": precos - 500,
        "Fechamento": precos,
        "Volume": [1000.0] * 150,
    }, index=datas)

    inicio = datetime(2025, 5, 1, tzinfo=timezone.utc)
    fim = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)

    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_cache = os.path.join(tmpdir, "cache.json")
        with patch("backend.backtesting_gemini.buscar_dados_historicos_paginado", return_value=df_mock) as mock_buscar, \
             patch("backend.backtesting_gemini.gerar_grafico_candlestick", return_value="fake.png"), \
             patch("backend.backtesting_gemini.analisar_mercado_com_gemini") as mock_ia, \
             patch("os.path.exists", return_value=False):
            mock_ia.return_value = {
                "acao": "COMPRAR",
                "justificativa": "Teste de período customizado.",
                "preco_alvo": "$58.000,00",
                "tempo_estimado": "7 a 14 dias",
            }
            resultado = executar_backtest_gemini_par(
                par="BTC/USDT",
                intervalo="1d",
                passo_amostragem=3,
                max_amostras=1,
                delay_segundos=0.0,
                caminho_cache=caminho_cache,
                data_inicio=inicio,
                data_fim=fim,
            )

        assert mock_buscar.call_args.args[2] == inicio
        assert mock_buscar.call_args.args[3] == fim
        assert resultado["periodo"]["inicio"] == inicio.isoformat()
        assert resultado["periodo"]["fim"] == fim.isoformat()
        assert resultado["total_amostras"] == 1


def test_executar_backtest_gemini_grade_multiplos_intervalos():
    resultado_base = {
        "par": "BTC/USDT",
        "intervalo": "1d",
        "total_amostras": 1,
        "amostras_novas": 1,
        "amostras_em_cache": 0,
        "distribuicao_acoes": {
            "COMPRAR": 1, "VENDER": 0, "MANTER": 0,
            "COMPRAR_pct": 100.0, "VENDER_pct": 0.0, "MANTER_pct": 0.0,
        },
        "acuracia_t1_pct": 100.0,
        "acuracia_prazo_estimado_pct": 100.0,
        "taxa_atingimento_alvo_pct": 100.0,
        "retorno_acumulado_pct": 5.0,
        "buy_and_hold": {"retorno_total_pct": -10.0},
        "taxa_concordancia_proxy_pct": 100.0,
        "registros": [],
    }

    def fake_par(**kwargs):
        return {
            **resultado_base,
            "par": kwargs["par"],
            "intervalo": kwargs["intervalo"],
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.backtesting_gemini.DIR_RESULTADOS", tmpdir), \
             patch("backend.backtesting_gemini.executar_backtest_gemini_par", side_effect=fake_par):
            relatorio = executar_backtest_gemini_grade(
                pares=["BTC/USDT"],
                intervalos=["1h", "4h"],
                passo_amostragem=3,
                data_inicio=datetime(2026, 7, 1, tzinfo=timezone.utc),
                data_fim=datetime(2026, 10, 31, 23, 59, 59, tzinfo=timezone.utc),
                prefixo_saida="backtesting_gemini_intraday_tcc",
            )

        assert relatorio["escopo"] == "exploratorio"
        assert relatorio["intervalos"] == ["1h", "4h"]
        assert relatorio["pares_avaliados"] == 1
        assert relatorio["combinacoes_avaliadas"] == 2
        assert relatorio["passo_amostragem_dias"] == 3
        assert os.path.exists(os.path.join(tmpdir, "backtesting_gemini_intraday_tcc.json"))
        assert os.path.exists(os.path.join(tmpdir, "backtesting_gemini_intraday_tcc.csv"))
        assert os.path.exists(os.path.join(tmpdir, "backtesting_gemini_intraday_tcc_resumo.txt"))

        with open(os.path.join(tmpdir, "backtesting_gemini_intraday_tcc.csv"), encoding="utf-8") as arquivo:
            csv = arquivo.read()
        assert "Intervalo,Par" in csv
        assert "1h,BTC/USDT" in csv
        assert "4h,BTC/USDT" in csv

        with open(os.path.join(tmpdir, "backtesting_gemini_intraday_tcc_resumo.txt"), encoding="utf-8") as arquivo:
            resumo = arquivo.read()
        assert "Timeframe(s): 1h, 4h" in resumo
        assert "Escopo: exploratorio" in resumo


def test_extrair_preco_alvo_casos_de_borda():
    assert extrair_preco_alvo("", 100.0) is None
    assert extrair_preco_alvo(",", 100.0) is None
    assert extrair_preco_alvo("98,5", 100.0) == 98.5
    assert extrair_preco_alvo("1.2.3", 100.0) is None


def test_carregar_cache_corrompido_retorna_vazio():
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho = os.path.join(tmpdir, "cache_corrompido.json")
        with open(caminho, "w", encoding="utf-8") as arquivo:
            arquivo.write("{invalido")
        assert carregar_cache_gemini(caminho) == {}


def test_sinal_proxy_vender_e_dados_ausentes():
    df_venda = pd.DataFrame({
        "Fechamento": [100.0] * 60,
        "MM20": [95.0] * 60,
        "MM50": [105.0] * 60,
        "RSI14": [50.0] * 60,
        "MACD": [1.0] * 60,
        "Sinal_MACD": [2.0] * 60,
    })
    assert calcular_sinal_proxy_ponto(df_venda) == "VENDER"

    df_nan = df_venda.copy()
    df_nan.loc[df_nan.index[-1], "MM20"] = np.nan
    assert calcular_sinal_proxy_ponto(df_nan) == "MANTER"


def test_buy_and_hold_na_janela():
    datas = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({"Fechamento": [100.0, 110.0, 90.0, 120.0]}, index=datas)
    bh = calcular_buy_and_hold(df, 0, 3)
    assert bh["retorno_total_pct"] == 20.0
    assert bh["max_drawdown_pct"] == -18.18
    assert bh["n_candles"] == 4


def test_dados_insuficientes_retorna_erro():
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_cache = os.path.join(tmpdir, "cache.json")
        with patch(
            "backend.backtesting_gemini.buscar_dados_historicos_paginado",
            return_value=pd.DataFrame(),
        ):
            resultado = executar_backtest_gemini_par(
                par="BTC/USDT",
                caminho_cache=caminho_cache,
            )
        assert "erro" in resultado


def test_executar_backtest_gemini_grade_gera_artefatos():
    resultado_par = {
        "par": "BTC/USDT",
        "intervalo": "1d",
        "total_amostras": 1,
        "amostras_novas": 1,
        "amostras_em_cache": 0,
        "distribuicao_acoes": {
            "COMPRAR": 1, "VENDER": 0, "MANTER": 0,
            "COMPRAR_pct": 100.0, "VENDER_pct": 0.0, "MANTER_pct": 0.0,
        },
        "acuracia_t1_pct": 100.0,
        "acuracia_prazo_estimado_pct": 100.0,
        "taxa_atingimento_alvo_pct": 100.0,
        "retorno_acumulado_pct": 5.0,
        "buy_and_hold": {"retorno_total_pct": -10.0},
        "taxa_concordancia_proxy_pct": 100.0,
        "registros": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.backtesting_gemini.DIR_RESULTADOS", tmpdir), \
             patch(
                 "backend.backtesting_gemini.executar_backtest_gemini_par",
                 return_value=resultado_par,
             ):
            relatorio = executar_backtest_gemini_grade(pares=["BTC/USDT"])

        assert relatorio["pares_avaliados"] == 1
        assert os.path.exists(os.path.join(tmpdir, "backtesting_gemini_tcc.json"))
        assert os.path.exists(os.path.join(tmpdir, "backtesting_gemini_tcc.csv"))
        assert os.path.exists(os.path.join(tmpdir, "backtesting_gemini_resumo.txt"))
        with open(os.path.join(tmpdir, "backtesting_gemini_tcc.csv"), encoding="utf-8") as arquivo:
            csv = arquivo.read()
        assert "Buy_and_Hold_pct" in csv
        assert "-10.0" in csv

