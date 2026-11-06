"""
tests/unit/test_comparacao_extended.py

Testes adicionais de backend/comparacao.py:
  - Buy-and-Hold e janelas de avaliação;
  - features, alvo e sequências do LSTM;
  - erros e limitações do walk-forward;
  - execução isolada/paralela da grade comparativa;
  - persistência e relatório comparativo.

Todos os testes usam dados sintéticos e mocks; nenhuma chamada real.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backend.analise_tecnica import enriquecer_dataframe
from backend import comparacao as cmp


def _df_ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    datas = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    precos = 60000.0 * np.cumprod(1 + rng.normal(0.001, 0.012, n))
    return pd.DataFrame({
        "Abertura": precos * 0.999,
        "Maxima": precos * 1.01,
        "Minima": precos * 0.99,
        "Fechamento": precos,
        "Volume": rng.uniform(1000, 5000, n),
    }, index=datas)


def _df_enriquecido(n: int = 120) -> pd.DataFrame:
    return enriquecer_dataframe(_df_ohlcv(n))


class TestBuyAndHold:

    def test_retorno_e_drawdown(self):
        df = pd.DataFrame({
            "Fechamento": [100.0, 110.0, 105.0, 120.0],
        }, index=pd.date_range("2025-01-01", periods=4, freq="D"))
        res = cmp.calcular_buy_and_hold(df)
        assert res["retorno_total_pct"] == 20.0
        assert res["max_drawdown_pct"] < 0
        assert res["n_candles"] == 4
        assert res["preco_inicial"] == 100.0
        assert res["preco_final"] == 120.0

    def test_janela_de_avaliacao(self):
        df = pd.DataFrame({
            "Fechamento": [100.0, 110.0, 105.0, 120.0],
        }, index=pd.date_range("2025-01-01", periods=4, freq="D"))
        res = cmp.calcular_buy_and_hold(df, data_inicio=datetime(2025, 1, 3))
        assert res["preco_inicial"] == 105.0
        assert res["retorno_total_pct"] == round((120 / 105 - 1) * 100, 2)
        assert res["n_candles"] == 2

    def test_dados_insuficientes(self):
        assert "erro" in cmp.calcular_buy_and_hold(pd.DataFrame())
        df = pd.DataFrame({"Fechamento": [100.0]},
                          index=pd.date_range("2025-01-01", periods=1))
        assert "erro" in cmp.calcular_buy_and_hold(df)

    def test_janela_insuficiente(self):
        df = pd.DataFrame({"Fechamento": [100.0, 101.0]},
                          index=pd.date_range("2025-01-01", periods=2))
        res = cmp.calcular_buy_and_hold(df, data_inicio=datetime(2030, 1, 1))
        assert "janela de avaliação" in res["erro"]


class TestFeaturesAlvoSequencias:

    def test_features_sem_nan_e_colunas_esperadas(self):
        feat = cmp._preparar_features(_df_enriquecido(80))
        assert list(feat.columns) == [
            "RSI_norm", "MACD_diff", "MM_diff_pct", "BB_posicao", "Retorno_1"
        ]
        assert not feat.isna().any().any()
        assert len(feat) > 0

    def test_alvo_binario_e_ultimo_nan(self):
        df = _df_ohlcv(10)
        alvo = cmp._calcular_target(df)
        assert alvo.iloc[-1] is np.nan or pd.isna(alvo.iloc[-1])
        assert set(alvo.dropna().unique()).issubset({0.0, 1.0})
        assert len(alvo.dropna()) == len(df) - 1

    def test_criar_sequencias_formato(self):
        feat = cmp._preparar_features(_df_enriquecido(80))
        alvo = cmp._calcular_target(_df_enriquecido(80))
        X, y, indices = cmp._criar_sequencias(feat, alvo, seq_len=5)
        assert X.shape[0] == y.shape[0] == len(indices)
        assert X.shape[1:] == (5, feat.shape[1])
        assert X.dtype == np.float32

    def test_criar_sequencias_vazias(self):
        feat = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        alvo = pd.Series([np.nan, np.nan, np.nan])
        X, y, indices = cmp._criar_sequencias(feat, alvo, seq_len=2)
        assert X.shape == (0, 2, 1)
        assert len(y) == 0
        assert indices == []

    def test_modelo_lstm_forward(self):
        if not cmp.TORCH_DISPONIVEL:
            pytest.skip("PyTorch indisponível")
        import torch
        modelo = cmp.ModeloLSTM(n_features=5, hidden_size=4)
        saida = modelo(torch.randn(3, 6, 5))
        assert tuple(saida.shape) == (3,)


class TestModeloSupervisionado:

    def test_erro_sem_torch(self, monkeypatch):
        monkeypatch.setattr(cmp, "TORCH_DISPONIVEL", False)
        res = cmp.executar_modelo_supervisionado(_df_enriquecido(80))
        assert "PyTorch" in res["erro"]

    def test_dados_insuficientes(self):
        res = cmp.executar_modelo_supervisionado(_df_ohlcv(10))
        assert "Dados insuficientes" in res["erro"]

    def test_janela_de_treino_insuficiente(self):
        res = cmp.executar_modelo_supervisionado(
            _df_enriquecido(80), janela_treino_pct=0.05,
            seq_len=8, epochs=1, hidden_size=4, batch_size=16,
        )
        assert "treino insuficiente" in res["erro"]
        assert res["seq_len_solicitado"] == 8

    def test_seq_len_reduzido_registrado(self):
        res = cmp.executar_modelo_supervisionado(
            _df_enriquecido(80), seq_len=500, epochs=1,
            hidden_size=4, batch_size=16,
        )
        # A validação de amostras pode falhar depois; o ajuste deve ser registrado.
        assert res["seq_len"] < 500
        assert "aviso_seq_len" in res

    def test_resultado_tem_metadados(self):
        res = cmp.executar_modelo_supervisionado(
            _df_enriquecido(100), seq_len=5, epochs=1,
            hidden_size=4, batch_size=16, passo_retreino=10,
        )
        assert "erro" not in res, res.get("erro")
        assert res["arquitetura"] == "LSTM (PyTorch)"
        assert res["n_treino"] > 0
        assert res["n_teste"] > 0
        assert 0 <= res["acuracia_direcional"] <= 1
        assert 0 <= res["acuracia_baseline_maioria"] <= 1
        assert res["n_refits"] >= 1


class TestCompararEstrategias:

    def test_erro_sem_dados(self):
        res = cmp.comparar_estrategias(
            "BTC/USDT", "1d", df_precarregado=pd.DataFrame()
        )
        assert "Nenhum dado" in res["erro"]

    def test_erro_em_enriquecimento(self):
        with patch(
            "backend.comparacao.enriquecer_dataframe",
            side_effect=ValueError("menos de 50 candles"),
        ):
            res = cmp.comparar_estrategias(
                "BTC/USDT", "1d", df_precarregado=_df_ohlcv(60)
            )
        assert "ValueError" in res["erro"]
        assert "menos de 50" in res["erro"]

    def test_periodo_comum_indisponivel_sem_previsoes(self):
        df = _df_enriquecido(100)
        with patch(
            "backend.comparacao.executar_modelo_supervisionado",
            return_value={"erro": "sem previsões", "estrategia": "LSTM_Supervisionado"},
        ):
            res = cmp.comparar_estrategias(
                "BTC/USDT", "1d", df_precarregado=df
            )
        assert "Periodo" not in res
        assert "Período comum indisponível" in res["periodo_comum"]["erro"]

    def test_periodo_comum_com_lstm(self):
        df = _df_enriquecido(110)
        with patch(
            "backend.comparacao.executar_modelo_supervisionado",
            return_value={
                "data_inicio_teste": df.index[60].isoformat(),
                "data_fim_teste": df.index[-2].isoformat(),
                "retorno_total_pct": 1.0,
                "acuracia_direcional": 0.5,
                "max_drawdown_pct": -1.0,
            },
        ):
            res = cmp.comparar_estrategias(
                "BTC/USDT", "1d", df_precarregado=df
            )
        comum = res["periodo_comum"]
        assert comum["data_inicio"] == df.index[60].isoformat()
        assert comum["criptoia"]["data_inicio_avaliacao"] == df.index[60].isoformat()
        assert comum["buy_and_hold"]["n_candles"] > 0
        assert comum["supervisionado"]["retorno_total_pct"] == 1.0


class TestGradeComparativa:

    @staticmethod
    def _fake(par, tf, *_args, **_kwargs):
        return {"par": par, "intervalo": tf, "periodo_comum": {}}

    def test_sequencial(self):
        with patch("backend.comparacao.comparar_estrategias", side_effect=self._fake) as mock:
            res = cmp.comparar_todos_pares(
                pares=["BTC/USDT", "ETH/USDT"], timeframes=["1d", "4h"],
                max_workers=1,
            )
        assert [(r["par"], r["intervalo"]) for r in res] == [
            ("BTC/USDT", "1d"), ("BTC/USDT", "4h"),
            ("ETH/USDT", "1d"), ("ETH/USDT", "4h"),
        ]
        assert mock.call_count == 4

    def test_paralelo_com_falha_individual(self):
        def fake(par, tf, *_args, **_kwargs):
            if "SOL" in par:
                raise RuntimeError("falha simulada")
            return self._fake(par, tf)

        with patch("backend.comparacao.comparar_estrategias", side_effect=fake):
            res = cmp.comparar_todos_pares(
                pares=["BTC/USDT", "SOL/USDT"], timeframes=["1d"],
                max_workers=2,
            )
        assert res[0]["par"] == "BTC/USDT"
        assert "falha simulada" in res[1]["erro"]

    def test_falha_sequencial(self):
        with patch(
            "backend.comparacao.comparar_estrategias",
            side_effect=RuntimeError("falha sequencial"),
        ):
            res = cmp.comparar_todos_pares(
                pares=["BTC/USDT"], timeframes=["1d"], max_workers=1
            )
        assert "falha sequencial" in res[0]["erro"]


class TestPersistenciaRelatorioComparacao:

    def test_salva_json_e_csv(self, tmp_path):
        resultado = [{
            "par": "BTC/USDT", "intervalo": "1d",
            "criptoia": {"retorno_total_pct": 1.0, "acuracia_direcional": 0.6},
            "buy_and_hold": {"retorno_total_pct": 2.0},
            "supervisionado": {"retorno_total_pct": 0.5,
                               "acuracia_direcional": 0.55,
                               "acuracia_baseline_maioria": 0.5,
                               "data_inicio_teste": "2025-01-01",
                               "data_fim_teste": "2025-02-01"},
            "periodo_comum": {
                "criptoia": {"retorno_total_pct": 1.0, "acuracia_direcional": 0.6},
                "buy_and_hold": {"retorno_total_pct": 2.0},
                "supervisionado": {"retorno_total_pct": 0.5, "acuracia_direcional": 0.55},
            },
        }]
        json_path = tmp_path / "cmp.json"
        csv_path = tmp_path / "cmp.csv"
        caminhos = cmp.salvar_resultados_comparacao(
            resultado, str(json_path), str(csv_path)
        )
        assert json_path.exists() and csv_path.exists()
        assert caminhos["json"] == str(json_path.resolve())
        df = pd.read_csv(csv_path)
        assert df.loc[0, "comum_criptoia_retorno_pct"] == 1.0

    def test_salva_sem_csv(self, tmp_path):
        json_path = tmp_path / "cmp.json"
        caminhos = cmp.salvar_resultados_comparacao([], str(json_path), None)
        assert "csv" not in caminhos
        assert json_path.exists()

    def test_relatorio_completo_e_erros(self):
        resultado = [{
            "par": "BTC/USDT", "intervalo": "1d",
            "data_inicio_str": "2025-01-01", "data_fim_str": "2025-06-01",
            "periodo_comum": {
                "data_inicio": "2025-03-01", "data_fim": "2025-06-01",
                "criptoia": {"retorno_total_pct": 1.0, "acuracia_direcional": 0.6,
                             "max_drawdown_pct": -2.0},
                "buy_and_hold": {"retorno_total_pct": 2.0, "acuracia_direcional": None,
                                 "max_drawdown_pct": -3.0},
                "supervisionado": {"erro": "falhou"},
            },
            "criptoia": {"erro": "falhou"},
            "buy_and_hold": {"retorno_total_pct": 2.0, "acuracia_direcional": None},
            "supervisionado": {"erro": "falhou"},
        }, {
            "par": "ETH/USDT", "intervalo": "1d",
            "data_inicio_str": "2025-01-01", "data_fim_str": "2025-06-01",
            "periodo_comum": {"erro": "indisponível"},
            "criptoia": {"retorno_total_pct": 0.0},
            "buy_and_hold": {"retorno_total_pct": 0.0},
            "supervisionado": {"erro": "sem previsões"},
        }, {
            "par": "SOL/USDT", "intervalo": "1d", "erro": "coleta falhou",
        }]
        texto = cmp.formatar_relatorio_comparacao(resultado)
        assert "RELATÓRIO COMPARATIVO" in texto
        assert "ERRO" in texto
        assert "indisponível" in texto
        assert "coleta falhou" in texto
        assert "Supervisionado (LSTM)" in texto
