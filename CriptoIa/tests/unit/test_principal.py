"""
tests/unit/test_principal.py

Testes de orquestração do CriptoIA:
  - RF07/CT05: abertura do dashboard no navegador (mock de webbrowser)
  - RNF06/CT06: falha de coleta/API retorna fallback estruturado MANTER,
    sem traceback e com mensagem descritiva
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from principal import (
    _parse_data,
    _parse_intervalos,
    executar_analise,
    executar_modo_backtesting,
    executar_modo_backtesting_gemini,
    executar_modo_comparacao,
)


@pytest.fixture
def df_ohlcv():
    n = 60
    datas = pd.date_range("2025-01-01", periods=n, freq="D")
    precos = pd.Series(range(n), dtype=float) + 50000.0
    return pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.01,
        "Minima": precos * 0.99,
        "Fechamento": precos,
        "Volume": 1000.0,
    }, index=datas)


def test_executar_analise_abre_dashboard_no_navegador(df_ohlcv):
    """RF07/CT05: após sucesso, o dashboard deve ser aberto no navegador."""
    with patch("principal.buscar_dados_historicos", return_value=df_ohlcv), \
         patch("principal.buscar_preco_atual", return_value={
             "preco": 95000.0, "variacao_24h": 1.0, "volume_24h": 1000.0,
             "maxima_24h": 96000.0, "minima_24h": 94000.0,
         }), \
         patch("principal.enriquecer_dataframe", return_value=df_ohlcv), \
         patch("principal.resumir_indicadores", return_value={
             "preco_fechamento": 95000.0, "mm20": 94000.0, "mm50": 91000.0,
             "rsi14": 58.5, "macd": 120.0, "sinal_macd": 95.0,
             "bollinger_superior": 98000.0, "bollinger_inferior": 90000.0,
         }), \
         patch("principal.gerar_grafico_candlestick", return_value="grafico.png"), \
         patch("principal.analisar_mercado_com_gemini", return_value={
             "acao": "COMPRAR", "justificativa": "Teste", "preco_alvo": "$100.000",
             "tempo_estimado": "7 dias",
         }), \
         patch("principal.gerar_dashboard", return_value="dashboard.html"), \
         patch("principal.webbrowser.open") as mock_open:
        resultado = executar_analise(par="BTC/USDT", intervalo="1d", quantidade=60)

    assert resultado["status"] == "ok"
    mock_open.assert_called_once()
    assert "dashboard.html" in mock_open.call_args.args[0]


def test_executar_analise_falha_binance_retorna_fallback_manter():
    """RNF06/CT06: falha de coleta não deve propagar exceção; fallback MANTER."""
    with patch("principal.buscar_dados_historicos", side_effect=ConnectionError(
        "Sem conexão com a Binance"
    )), patch("principal._salvar_fallback", return_value="fallback.json"):
        resultado = executar_analise(par="BTC/USDT", intervalo="1d", quantidade=60)

    assert resultado["status"] == "erro"
    assert "Binance" in resultado["erro"]
    assert resultado["analise_ia"]["acao"] == "MANTER"
    assert resultado["analise_ia"]["fallback"] is True
    assert resultado["fallback_json"] == "fallback.json"


def test_executar_analise_poucos_candles_retorna_fallback(df_ohlcv):
    """RNF06/CT06: menos de 50 candles também deve gerar fallback descritivo."""
    with patch("principal.buscar_dados_historicos", return_value=df_ohlcv.iloc[:10]), \
         patch("principal.buscar_preco_atual", return_value={"preco": 1.0}), \
         patch("principal._salvar_fallback", return_value="fallback.json"):
        resultado = executar_analise(par="BTC/USDT", intervalo="1d", quantidade=10)

    assert resultado["status"] == "erro"
    assert resultado["analise_ia"]["acao"] == "MANTER"
    assert "50" in resultado["erro"]


def _args_modo(**overrides):
    base = {
        "grade": False,
        "par": "BTC/USDT",
        "intervalo": "1d",
        "inicio": "2025-01-01",
        "fim": "2025-02-01",
        "max_workers": 1,
        "epochs": 1,
        "seq_len": 4,
        "hidden_size": 4,
        "passo_retreino": 10,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_executar_modo_backtesting_salva_relatorio():
    """O modo CLI deve persistir JSON/CSV e o relatório textual."""
    with patch(
        "backend.backtesting.executar_backtesting",
        return_value={"par": "BTC/USDT", "intervalo": "1d"},
    ) as mock_exec, patch(
        "backend.backtesting.formatar_relatorio_backtesting",
        return_value="RELATORIO BT",
    ), patch(
        "backend.backtesting.salvar_resultados_backtesting",
        return_value={"json": "bt.json", "csv": "bt.csv"},
    ), patch(
        "principal._salvar_relatorio_texto", return_value="bt.txt"
    ) as mock_relatorio:
        executar_modo_backtesting(_args_modo())

    mock_exec.assert_called_once()
    mock_relatorio.assert_called_once_with(
        "backtesting_relatorio.txt", "RELATORIO BT"
    )


def test_executar_modo_backtesting_grade():
    """O modo --grade deve chamar a execução completa."""
    with patch(
        "backend.backtesting.executar_backtesting_completo",
        return_value=[{"par": "BTC/USDT", "intervalo": "1d"}],
    ) as mock_grade, patch(
        "backend.backtesting.formatar_relatorio_backtesting",
        return_value="RELATORIO BT",
    ), patch(
        "backend.backtesting.salvar_resultados_backtesting",
        return_value={"json": "bt.json", "csv": "bt.csv"},
    ), patch(
        "principal._salvar_relatorio_texto", return_value="bt.txt"
    ):
        executar_modo_backtesting(_args_modo(grade=True))

    mock_grade.assert_called_once()


def test_executar_modo_comparacao_salva_relatorio():
    """O modo de comparação deve persistir o relatório textual."""
    with patch(
        "backend.comparacao.comparar_estrategias",
        return_value={"par": "BTC/USDT", "intervalo": "1d"},
    ) as mock_cmp, patch(
        "backend.comparacao.formatar_relatorio_comparacao",
        return_value="RELATORIO CMP",
    ), patch(
        "backend.comparacao.salvar_resultados_comparacao",
        return_value={"json": "cmp.json", "csv": "cmp.csv"},
    ), patch(
        "principal._salvar_relatorio_texto", return_value="cmp.txt"
    ) as mock_relatorio:
        executar_modo_comparacao(_args_modo())

    mock_cmp.assert_called_once()
    mock_relatorio.assert_called_once_with(
        "comparacao_relatorio.txt", "RELATORIO CMP"
    )


def test_parse_data_fim_do_dia():
    """A data final do CLI deve cobrir todo o dia solicitado."""
    inicio = _parse_data("2025-01-01")
    fim = _parse_data("2026-06-30", fim_do_dia=True)

    assert inicio.hour == 0 and inicio.minute == 0 and inicio.second == 0
    assert fim.hour == 23 and fim.minute == 59 and fim.second == 59
    assert fim.tzinfo is not None


def test_parse_intervalos():
    """A lista de timeframes deve ser validada, normalizada e sem duplicatas."""
    assert _parse_intervalos(None) == ["1d"]
    assert _parse_intervalos("1h,4h") == ["1h", "4h"]
    assert _parse_intervalos("1h, 1h, 4h") == ["1h", "4h"]
    with pytest.raises(SystemExit):
        _parse_intervalos("5m")


def test_executar_modo_backtesting_gemini_grade_customizado():
    """O modo Gemini deve encaminhar período, timeframes e prefixo de saída."""
    args = SimpleNamespace(
        grade=True,
        inicio="2026-07-01",
        fim="2026-10-31",
        passo=3,
        max_amostras=2,
        delay=0.0,
        intervalos="1h,4h",
        prefixo_saida="backtesting_gemini_intraday_tcc",
    )

    with patch("backend.backtesting_gemini.executar_backtest_gemini_grade") as mock_grade:
        executar_modo_backtesting_gemini(args)

    kwargs = mock_grade.call_args.kwargs
    assert kwargs["intervalos"] == ["1h", "4h"]
    assert kwargs["passo_amostragem"] == 3
    assert kwargs["data_inicio"].isoformat() == "2026-07-01T00:00:00+00:00"
    assert kwargs["data_fim"].isoformat() == "2026-10-31T23:59:59+00:00"
    assert kwargs["prefixo_saida"] == "backtesting_gemini_intraday_tcc"
