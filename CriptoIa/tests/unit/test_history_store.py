"""
tests/unit/test_history_store.py

Testes do armazenamento local de histórico:
  - leitura de arquivo ausente/corrompido;
  - escrita atômica e ordenação (mais recente primeiro);
  - limites de 100 análises e 50 backtestings;
  - campos padronizados e fallback MANTER;
  - limpeza do arquivo temporário em caso de falha.
"""

import json
import os
from unittest.mock import patch

import pytest

from webapp import history_store as hs


def test_historico_ausente_retorna_lista_vazia(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "CAMINHO_HIST_ANALISES", str(tmp_path / "nao_existe.json"))
    assert hs.listar_analises() == []


def test_historico_corrompido_retorna_lista_vazia(tmp_path, monkeypatch):
    caminho = tmp_path / "corrompido.json"
    caminho.write_text("{invalido", encoding="utf-8")
    monkeypatch.setattr(hs, "CAMINHO_HIST_ANALISES", str(caminho))
    assert hs.listar_analises() == []


def test_historico_com_formato_invalido_retorna_lista_vazia(tmp_path, monkeypatch):
    caminho = tmp_path / "dicionario.json"
    caminho.write_text(json.dumps({"nao": "lista"}), encoding="utf-8")
    monkeypatch.setattr(hs, "CAMINHO_HIST_ANALISES", str(caminho))
    assert hs.listar_analises() == []


def test_salvar_e_listar_analise(tmp_path, monkeypatch):
    caminho = tmp_path / "analises.json"
    monkeypatch.setattr(hs, "CAMINHO_HIST_ANALISES", str(caminho))

    hs.salvar_analise({
        "par": "ETH/USDT",
        "intervalo": "4h",
        "quantidade": 120,
        "preco_atual": 3000.0,
        "acao": "COMPRAR",
        "justificativa": "Teste",
        "preco_alvo": "$3.200",
        "tempo_estimado": "7 dias",
        "indicadores": {"rsi14": 55.0},
        "fallback": False,
    })
    hs.salvar_analise({"par": "BTC/USDT"})

    analises = hs.listar_analises()
    assert len(analises) == 2
    assert analises[0]["par"] == "BTC/USDT"  # mais recente primeiro
    assert analises[0]["acao"] == "MANTER"
    assert analises[0]["id"] == 2
    assert analises[1]["par"] == "ETH/USDT"
    assert analises[1]["preco_alvo"] == "$3.200"
    assert analises[1]["fallback"] is False


def test_limite_de_100_analises(tmp_path, monkeypatch):
    caminho = tmp_path / "analises.json"
    monkeypatch.setattr(hs, "CAMINHO_HIST_ANALISES", str(caminho))
    for i in range(105):
        hs.salvar_analise({"par": f"PAR{i}/USDT"})
    analises = hs.listar_analises()
    assert len(analises) == 100
    assert analises[0]["par"] == "PAR104/USDT"


def test_salvar_e_listar_backtest(tmp_path, monkeypatch):
    caminho = tmp_path / "backtest.json"
    monkeypatch.setattr(hs, "CAMINHO_HIST_BACKTEST", str(caminho))

    hs.salvar_resultado_backtest({"par": "BTC/USDT", "intervalo": "1d",
                                  "erro": None, "criptoia": {"retorno_total_pct": 1.0}})
    hs.salvar_resultado_backtest({"par": "ETH/USDT", "intervalo": "4h"})

    resultados = hs.listar_resultados_backtest()
    assert len(resultados) == 2
    assert resultados[0]["par"] == "ETH/USDT"
    assert resultados[0]["id"] == 2
    assert "data_execucao" in resultados[0]
    assert resultados[1]["criptoia"]["retorno_total_pct"] == 1.0


def test_limite_de_50_backtests(tmp_path, monkeypatch):
    caminho = tmp_path / "backtest.json"
    monkeypatch.setattr(hs, "CAMINHO_HIST_BACKTEST", str(caminho))
    for i in range(55):
        hs.salvar_resultado_backtest({"par": f"PAR{i}/USDT"})
    resultados = hs.listar_resultados_backtest()
    assert len(resultados) == 50
    assert resultados[0]["par"] == "PAR54/USDT"


def test_falha_de_escrita_remove_temporario(tmp_path, monkeypatch, capsys):
    caminho = tmp_path / "falha.json"
    temporario = tmp_path / "falha.json.tmp"
    monkeypatch.setattr(hs, "CAMINHO_HIST_ANALISES", str(caminho))

    with patch("webapp.history_store.os.replace", side_effect=OSError("disco cheio")):
        with pytest.raises(OSError, match="disco cheio"):
            hs.salvar_analise({"par": "BTC/USDT"})

    assert not temporario.exists()
    assert "Erro ao salvar histórico" in capsys.readouterr().out
