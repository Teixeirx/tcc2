"""
tests/unit/test_backtesting_extended.py

Testes adicionais de backend/backtesting.py:
  - validação/normalização de datas e parâmetros;
  - paginação, deduplicação, filtro de período e coleta incompleta;
  - alinhamento anti look-ahead (sinal em t vs. retorno de t+1);
  - métricas, critérios de sucesso e casos de erro;
  - execução completa paralela/sequencial com isolamento de falhas;
  - persistência JSON/CSV e relatório textual.

Todos os testes usam mocks ou dados sintéticos; nenhuma chamada à Binance.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.analise_tecnica import enriquecer_dataframe
from backend import backtesting as bt


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _df_ohlcv(n: int = 120, preco_inicial: float = 50000.0) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    datas = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    precos = preco_inicial * np.cumprod(1 + rng.normal(0.002, 0.01, n))
    return pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.01,
        "Minima": precos * 0.99,
        "Fechamento": precos,
        "Volume": rng.uniform(1000, 5000, n),
    }, index=datas)


def _df_enriquecido(n: int = 120) -> pd.DataFrame:
    return enriquecer_dataframe(_df_ohlcv(n))


def _candles(inicio: datetime, quantidade: int) -> list:
    base = int(inicio.timestamp() * 1000)
    return [
        [base + i * 3600_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0]
        for i in range(quantidade)
    ]


# ── Utilidades e validação ────────────────────────────────────────────────────

class TestUtilidadesBacktesting:

    def test_normalizar_data_naive_e_aware(self):
        naive = datetime(2025, 1, 1, 12, 0)
        assert bt._normalizar_data(naive).tzinfo == timezone.utc
        aware = datetime(2025, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=-3)))
        normalizada = bt._normalizar_data(aware)
        assert normalizada.tzinfo == timezone.utc
        assert normalizada.hour == 15

    def test_timestamp_compativel_indice_naive(self):
        indice = pd.date_range("2025-01-01", periods=3, freq="D")
        ts = bt._timestamp_compativel(indice, datetime(2025, 1, 2, tzinfo=timezone.utc))
        assert ts.tzinfo is None
        assert ts == pd.Timestamp("2025-01-02")

    def test_timestamp_compativel_indice_aware(self):
        indice = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        ts = bt._timestamp_compativel(indice, datetime(2025, 1, 2))
        assert ts.tzinfo is not None

    @pytest.mark.parametrize("par", ["", "BTCUSDT", "BTC/", 123])
    def test_validar_par_invalido(self, par):
        with pytest.raises(ValueError, match="Par inválido"):
            bt._validar_parametros(
                par, "1d",
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 2, 1, tzinfo=timezone.utc),
            )

    def test_validar_timeframe_vazio(self):
        with pytest.raises(ValueError, match="Timeframe inválido"):
            bt._validar_parametros(
                "BTC/USDT", " ",
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 2, 1, tzinfo=timezone.utc),
            )

    def test_validar_ordem_das_datas(self):
        with pytest.raises(ValueError, match="data de início"):
            bt._validar_parametros(
                "BTC/USDT", "1d",
                datetime(2025, 2, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    def test_estimar_candles_e_inclusivo(self):
        inicio = datetime(2025, 1, 1, tzinfo=timezone.utc)
        fim = datetime(2025, 1, 3, tzinfo=timezone.utc)
        # 1d: 1, 2 e 3 de janeiro = 3 candles
        assert bt._estimar_candles(inicio, fim, "1d") == 3
        assert bt._estimar_candles(inicio, fim, "4h") == 13
        assert bt._estimar_candles(inicio, fim, "tf-desconhecido") == 3


# ── Paginação ─────────────────────────────────────────────────────────────────

class TestColetaPaginada:

    def test_pagina_ate_esgotar_e_filtra_periodo(self, monkeypatch):
        inicio = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
        fim = inicio + timedelta(hours=4)
        cliente = MagicMock()
        cliente.fetch_ohlcv.side_effect = [
            _candles(inicio, 3),
            _candles(inicio + timedelta(hours=3), 2),
        ]
        monkeypatch.setattr(bt, "LIMITE_CANDLES_POR_REQUISICAO", 3)
        monkeypatch.setattr(bt, "obter_cliente_binance", lambda: cliente)
        monkeypatch.setattr(bt.time, "sleep", lambda *_: None)

        df = bt.buscar_dados_historicos_paginado("BTC/USDT", "1h", inicio, fim)

        assert len(df) == 5
        assert df.attrs["coleta_incompleta"] is False
        assert df.attrs["erro_coleta"] is None
        assert cliente.fetch_ohlcv.call_count == 2
        assert df.index[0] == pd.Timestamp(inicio)
        assert df.index[-1] == pd.Timestamp(fim)

    def test_deduplica_e_exclui_fora_do_periodo(self, monkeypatch):
        inicio = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
        fim = inicio + timedelta(hours=2)
        antes = _candles(inicio - timedelta(hours=1), 1)[0]
        dentro = _candles(inicio, 2)
        depois = _candles(fim + timedelta(hours=1), 1)[0]
        cliente = MagicMock()
        cliente.fetch_ohlcv.return_value = [antes, dentro[0], dentro[0], dentro[1], depois]
        monkeypatch.setattr(bt, "obter_cliente_binance", lambda: cliente)
        monkeypatch.setattr(bt.time, "sleep", lambda *_: None)

        df = bt.buscar_dados_historicos_paginado("BTC/USDT", "1h", inicio, fim)

        assert len(df) == 2
        assert list(df["Fechamento"]) == [100.5, 101.5]

    def test_falha_no_meio_marca_coleta_incompleta(self, monkeypatch):
        inicio = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
        fim = inicio + timedelta(hours=5)
        cliente = MagicMock()
        cliente.fetch_ohlcv.side_effect = [
            _candles(inicio, 3),
            ConnectionError("timeout"),
        ]
        monkeypatch.setattr(bt, "LIMITE_CANDLES_POR_REQUISICAO", 3)
        monkeypatch.setattr(bt, "obter_cliente_binance", lambda: cliente)
        monkeypatch.setattr(bt.time, "sleep", lambda *_: None)

        df = bt.buscar_dados_historicos_paginado("BTC/USDT", "1h", inicio, fim)

        assert len(df) == 3
        assert df.attrs["coleta_incompleta"] is True
        assert "timeout" in df.attrs["erro_coleta"]

    def test_sem_candles_retorna_vazio(self, monkeypatch):
        cliente = MagicMock()
        cliente.fetch_ohlcv.return_value = []
        monkeypatch.setattr(bt, "obter_cliente_binance", lambda: cliente)
        df = bt.buscar_dados_historicos_paginado(
            "BTC/USDT", "1d",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 5, tzinfo=timezone.utc),
        )
        assert df.empty
        assert df.attrs["coleta_incompleta"] is False
        assert df.attrs["erro_coleta"] is None

    def test_erro_na_primeira_pagina_retorna_vazio_incompleto(self, monkeypatch):
        cliente = MagicMock()
        cliente.fetch_ohlcv.side_effect = ConnectionError("sem rede")
        monkeypatch.setattr(bt, "obter_cliente_binance", lambda: cliente)
        df = bt.buscar_dados_historicos_paginado(
            "BTC/USDT", "1d",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 5, tzinfo=timezone.utc),
        )
        assert df.empty
        assert df.attrs["coleta_incompleta"] is True
        assert "sem rede" in df.attrs["erro_coleta"]


# ── Sinais e métricas ─────────────────────────────────────────────────────────

class TestSinaisMetricas:

    def test_regras_de_sinal(self):
        df = pd.DataFrame({
            "MM20": [2.0, 1.0, 2.0],
            "MM50": [1.0, 2.0, 2.0],
            "RSI14": [50.0, 50.0, 80.0],
            "MACD": [2.0, 1.0, 2.0],
            "Sinal_MACD": [1.0, 2.0, 1.0],
        })
        assert list(bt.gerar_sinais(df)) == [1, -1, 0]

    def test_indicadores_ausentes_retornam_erro(self):
        df = _df_ohlcv(80)
        metricas = bt.calcular_metricas(df)
        assert "Indicadores ausentes" in metricas["erro"]

    def test_janela_sem_dados_validos_retorna_erro(self):
        df = _df_enriquecido(80)
        metricas = bt.calcular_metricas(
            df, data_inicio_avaliacao=datetime(2030, 1, 1, tzinfo=timezone.utc)
        )
        assert "Sem dados válidos" in metricas["erro"]

    def test_sem_sinais_direcionais_acuracia_none(self):
        df = _df_enriquecido(80)
        zeros = pd.Series(0, index=df.index, dtype=int)
        with patch("backend.backtesting.gerar_sinais", return_value=zeros):
            metricas = bt.calcular_metricas(df)
        assert metricas["acuracia_direcional"] is None
        assert metricas["retorno_total_pct"] == 0.0

    def test_sinal_em_t_usa_retorno_de_t_mais_1(self):
        """O sinal em t deve capturar o retorno de t+1 (sem look-ahead)."""
        n = 80
        datas = pd.date_range("2025-01-01", periods=n, freq="D")
        fechamento = np.full(n, 100.0)
        fechamento[61] = 110.0  # salto entre t=60 e t=61
        df = pd.DataFrame({
            "Fechamento": fechamento,
            "MM20": 1.0, "MM50": 1.0, "RSI14": 50.0,
            "MACD": 1.0, "Sinal_MACD": 0.0,
        }, index=datas)
        sinais = pd.Series(0, index=datas, dtype=int)
        sinais.iloc[60] = 1
        with patch("backend.backtesting.gerar_sinais", return_value=sinais):
            metricas = bt.calcular_metricas(df)

        assert metricas["retorno_total_pct"] == 10.0
        assert metricas["n_sinais_compra"] == 1
        # O último candle não possui retorno futuro e não entra na avaliação
        assert metricas["n_candles_validos"] == n - 1
        assert metricas["data_fim_avaliacao"] == datas[-2].isoformat()

    def test_janela_de_avaliacao_restringe_metricas(self):
        df = _df_enriquecido(120)
        inicio = df.index[60]
        metricas = bt.calcular_metricas(df, data_inicio_avaliacao=inicio)
        assert metricas["data_inicio_avaliacao"] == inicio.isoformat()
        assert metricas["n_candles"] == len(df)
        assert metricas["n_candles_validos"] < len(df)


# ── execução de um experimento ────────────────────────────────────────────────

class TestExecutarBacktesting:

    def test_execucao_bem_sucedida(self):
        df = _df_enriquecido(120)
        df.attrs["coleta_incompleta"] = False
        df.attrs["erro_coleta"] = None
        with patch("backend.backtesting.buscar_dados_historicos_paginado", return_value=df):
            res = bt.executar_backtesting(
                "BTC/USDT", "1d",
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 2, 1, tzinfo=timezone.utc),
            )
        assert "erro" not in res
        assert res["par"] == "BTC/USDT"
        assert res["metodologia"] == bt.METODOLOGIA_PROXY
        assert res["avalia_gemini"] is False
        assert res["n_candles_coletados"] == 120
        assert res["cobertura_solicitada_pct"] > 0

    def test_sem_dados_retorna_erro(self):
        with patch("backend.backtesting.buscar_dados_historicos_paginado", return_value=pd.DataFrame()):
            res = bt.executar_backtesting("BTC/USDT", "1d")
        assert "Nenhum dado" in res["erro"]

    def test_coleta_incompleta_retorna_erro(self):
        df = _df_enriquecido(120)
        df.attrs["coleta_incompleta"] = True
        df.attrs["erro_coleta"] = "timeout"
        with patch("backend.backtesting.buscar_dados_historicos_paginado", return_value=df):
            res = bt.executar_backtesting("BTC/USDT", "1d")
        assert "incompleta" in res["erro"]
        assert res["n_candles_coletados"] == 120

    def test_dados_insuficientes_retorna_erro(self):
        df = _df_ohlcv(40)
        df.attrs["coleta_incompleta"] = False
        with patch("backend.backtesting.buscar_dados_historicos_paginado", return_value=df):
            res = bt.executar_backtesting("BTC/USDT", "1d")
        assert "Dados insuficientes" in res["erro"]

    def test_excecao_e_capturada(self):
        with patch(
            "backend.backtesting.buscar_dados_historicos_paginado",
            side_effect=RuntimeError("falha simulada"),
        ):
            res = bt.executar_backtesting("BTC/USDT", "1d")
        assert "RuntimeError" in res["erro"]
        assert "falha simulada" in res["erro"]


# ── execução da grade e critérios ─────────────────────────────────────────────

class TestExecucaoCompleta:

    @staticmethod
    def _fake_exec(par, tf, *_):
        return {"par": par, "intervalo": tf, "acuracia_direcional": 0.55,
                "retorno_total_pct": 1.0}

    def test_ordem_sequencial(self):
        with patch("backend.backtesting.executar_backtesting", side_effect=self._fake_exec) as mock:
            res = bt.executar_backtesting_completo(
                pares=["BTC/USDT", "ETH/USDT"], timeframes=["1d", "4h"], max_workers=1
            )
        assert [(r["par"], r["intervalo"]) for r in res] == [
            ("BTC/USDT", "1d"), ("BTC/USDT", "4h"),
            ("ETH/USDT", "1d"), ("ETH/USDT", "4h"),
        ]
        assert mock.call_count == 4

    def test_ordem_paralela_e_falha_individual(self):
        def fake(par, tf, *_):
            if par == "ETH/USDT":
                raise RuntimeError("falha individual")
            return self._fake_exec(par, tf)

        with patch("backend.backtesting.executar_backtesting", side_effect=fake):
            res = bt.executar_backtesting_completo(
                pares=["BTC/USDT", "ETH/USDT"], timeframes=["1d"], max_workers=2
            )
        assert res[0]["par"] == "BTC/USDT"
        assert "erro" in res[1]
        assert "falha individual" in res[1]["erro"]

    def test_resumo_prefere_timeframe_diario(self):
        resultados = [
            {"par": "BTC/USDT", "intervalo": "1d", "acuracia_direcional": 0.60,
             "retorno_total_pct": 10.0},
            {"par": "BTC/USDT", "intervalo": "4h", "acuracia_direcional": 0.40,
             "retorno_total_pct": -5.0},
            {"par": "ETH/USDT", "intervalo": "1d", "acuracia_direcional": 0.55,
             "retorno_total_pct": 2.0},
            {"par": "SOL/USDT", "intervalo": "1d", "acuracia_direcional": 0.52,
             "retorno_total_pct": 1.0},
            {"par": "DOGE/USDT", "intervalo": "1d", "acuracia_direcional": 0.45,
             "retorno_total_pct": 0.0},
        ]
        resumo = bt.resumir_criterios_sucesso(resultados)
        assert resumo["ativos_avaliados"] == 4
        assert resumo["ativos_acuracia_acima_50"] == 3
        assert resumo["criterio_acuracia_atendido"] is True
        assert resumo["ativos_retorno_nao_negativo"] == 4
        assert resumo["criterio_retorno_atendido"] is True
        assert resumo["por_par"]["BTC/USDT"]["acuracia_media"] == 0.6

    def test_resumo_ignora_erros_e_usa_todos_quando_sem_diario(self):
        resultados = [
            {"par": "BTC/USDT", "intervalo": "4h", "acuracia_direcional": 0.60,
             "retorno_total_pct": 3.0},
            {"par": "BTC/USDT", "intervalo": "1h", "acuracia_direcional": 0.50,
             "retorno_total_pct": -1.0},
            {"par": "ETH/USDT", "intervalo": "1d", "erro": "falha"},
        ]
        resumo = bt.resumir_criterios_sucesso(resultados)
        assert resumo["ativos_avaliados"] == 1
        assert resumo["por_par"]["BTC/USDT"]["timeframes_avaliados"] == ["4h", "1h"]
        assert resumo["criterio_acuracia_atendido"] is False
        assert resumo["criterio_retorno_atendido"] is True

    def test_resumo_vazio(self):
        resumo = bt.resumir_criterios_sucesso([])
        assert resumo["ativos_avaliados"] == 0
        assert resumo["criterio_acuracia_atendido"] is False
        assert resumo["criterio_retorno_atendido"] is False


# ── Persistência e relatório ──────────────────────────────────────────────────

class TestPersistenciaRelatorio:

    def test_salva_json_e_csv(self, tmp_path):
        resultados = [{
            "par": "BTC/USDT", "intervalo": "1d", "n_candles": 100,
            "n_candles_validos": 90, "n_sinais_compra": 10,
            "n_sinais_venda": 5, "n_sinais_manter": 75,
            "acuracia_direcional": 0.55, "retorno_total_pct": 4.2,
            "max_drawdown_pct": -3.1, "n_operacoes": 12, "n_entradas": 6,
            "cobertura_solicitada_pct": 99.5,
            "primeiro_candle": "2025-01-01T00:00:00+00:00",
            "ultimo_candle": "2025-06-01T00:00:00+00:00",
        }]
        json_path = tmp_path / "sub" / "resultados.json"
        csv_path = tmp_path / "sub" / "resultados.csv"
        caminhos = bt.salvar_resultados_backtesting(
            resultados, caminho_json=str(json_path), caminho_csv=str(csv_path)
        )
        assert json_path.exists() and csv_path.exists()
        assert caminhos["json"] == str(json_path.resolve())
        assert caminhos["csv"] == str(csv_path.resolve())
        conteudo = pd.read_csv(csv_path)
        assert conteudo.loc[0, "par"] == "BTC/USDT"

    def test_salva_sem_csv(self, tmp_path):
        json_path = tmp_path / "resultados.json"
        caminhos = bt.salvar_resultados_backtesting(
            [], caminho_json=str(json_path), caminho_csv=None
        )
        assert "csv" not in caminhos
        assert json_path.exists()

    def test_relatorio_inclui_erro_individual(self):
        texto = bt.formatar_relatorio_backtesting([
            {"par": "BTC/USDT", "intervalo": "1d", "erro": "falha na coleta"},
        ])
        assert "ERRO: falha na coleta" in texto
        assert "Critério de acurácia atendido" in texto
