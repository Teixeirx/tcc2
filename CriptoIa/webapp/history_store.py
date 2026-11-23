"""
webapp/history_store.py — Gerenciador de histórico de análises e backtesting.

Persiste e recupera registros de análises e resultados de backtesting
em arquivos JSON estruturados. As escritas são atômicas e protegidas por
lock para reduzir o risco de corrupção em requisições concorrentes.
"""

import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Any

CAMINHO_HIST_ANALISES = os.path.join(os.path.dirname(__file__), "historico_analises.json")
CAMINHO_HIST_BACKTEST = os.path.join(os.path.dirname(__file__), "historico_backtest.json")

_LOCK = threading.Lock()


def _carregar_json(caminho: str) -> List[Dict[str, Any]]:
    if not os.path.exists(caminho):
        return []
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
            return dados if isinstance(dados, list) else []
    except Exception:
        return []


def _salvar_json(caminho: str, dados: List[Dict[str, Any]]) -> None:
    """Salva o JSON de forma atômica (arquivo temporário + replace)."""
    temporario = f"{caminho}.tmp"
    try:
        with open(temporario, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=2, ensure_ascii=False, default=str)
        os.replace(temporario, caminho)
    except Exception as e:
        print(f"[HistoryStore] Erro ao salvar histórico em {caminho}: {e}")
        if os.path.exists(temporario):
            try:
                os.remove(temporario)
            except OSError:
                pass
        # Propaga para que a camada web/CLI informe a falha de persistência
        # em vez de exibir uma mensagem de sucesso enganosa.
        raise


def salvar_analise(registro: Dict[str, Any]) -> None:
    """Adiciona uma análise ao histórico."""
    with _LOCK:
        historico = _carregar_json(CAMINHO_HIST_ANALISES)
        registro_formatado = {
            "id": len(historico) + 1,
            "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "par": registro.get("par", "BTC/USDT"),
            "intervalo": registro.get("intervalo", "1d"),
            "quantidade": registro.get("quantidade", 90),
            "preco_atual": registro.get("preco_atual"),
            "variacao_24h": registro.get("variacao_24h"),
            "acao": registro.get("acao", "MANTER"),
            "justificativa": registro.get("justificativa", ""),
            "preco_alvo": registro.get("preco_alvo"),
            "tempo_estimado": registro.get("tempo_estimado"),
            "indicadores": registro.get("indicadores", {}),
            "fallback": bool(registro.get("fallback", False)),
        }
        historico.insert(0, registro_formatado)  # Mais recente primeiro
        _salvar_json(CAMINHO_HIST_ANALISES, historico[:100])  # Mantém até 100 registros


def listar_analises() -> List[Dict[str, Any]]:
    """Retorna todas as análises salvas no histórico."""
    return _carregar_json(CAMINHO_HIST_ANALISES)


def salvar_resultado_backtest(resultado: Dict[str, Any]) -> None:
    """Salva um resultado de backtesting no histórico."""
    with _LOCK:
        historico = _carregar_json(CAMINHO_HIST_BACKTEST)
        registro = dict(resultado)
        registro["id"] = len(historico) + 1
        registro["data_execucao"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        historico.insert(0, registro)
        _salvar_json(CAMINHO_HIST_BACKTEST, historico[:50])


def listar_resultados_backtest() -> List[Dict[str, Any]]:
    """Retorna os resultados de backtesting salvos."""
    return _carregar_json(CAMINHO_HIST_BACKTEST)
