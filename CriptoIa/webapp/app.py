"""
webapp/app.py — Aplicação Web Flask do Sistema CriptoIA (TCC II).

Rotas e Funcionalidades:
  - /login       : Autenticação com sessão protegida e CSRF
  - /logout      : Encerramento de sessão
  - /            : Dashboard de visão geral com métricas e histórico
  - /analise     : Execução de análise técnica multimodal com Gemini
  - /historico   : Consulta ao histórico persistido de análises
  - /backtesting : Avaliação histórica e comparação de estratégias

A interface usa apenas resultados persistidos/executados. Não há números
acadêmicos fixos ou simulados apresentados como se fossem reais.
"""

import os
import re
import sys
import hmac
import json
import time
import secrets
from datetime import datetime, timedelta
from flask import (
    Flask, render_template, request, redirect, url_for, flash, session, abort
)

# Garante importação dos módulos do backend e frontend
DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIR_RAIZ not in sys.path:
    sys.path.insert(0, DIR_RAIZ)

from backend.coleta_dados import buscar_dados_historicos, buscar_preco_atual
from backend.analise_tecnica import enriquecer_dataframe, resumir_indicadores
from backend.gerador_grafico import gerar_grafico_candlestick
from backend.analisador_ia import analisar_mercado_com_gemini, MODELO_GEMINI
from frontend.gerador_dashboard import _gerar_grafico_plotly
from backend.backtesting import PARES_TCC, TIMEFRAMES_TCC
from backend.comparacao import comparar_estrategias
from webapp.auth import verificar_credenciais, login_required, USANDO_SENHA_PADRAO
from webapp.history_store import (
    salvar_analise,
    listar_analises,
    salvar_resultado_backtest,
    listar_resultados_backtest,
)

app = Flask(__name__)

# ── Configuração de segurança ─────────────────────────────────────────────────

_SECRET_ENV = os.getenv("FLASK_SECRET_KEY")
if not _SECRET_ENV:
    _SECRET_ENV = secrets.token_hex(32)
    print(
        "[AVISO] FLASK_SECRET_KEY não definida; usando segredo aleatório "
        "por processo. Defina a variável para manter sessões entre reinícios."
    )

app.secret_key = _SECRET_ENV
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    CSRF_ENABLED=os.getenv("CSRF_ENABLED", "1") != "0",
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
)

# Parâmetros enxutos do LSTM para uso interativo na web
LSTM_WEB_CONFIG = {
    "epochs": int(os.getenv("LSTM_WEB_EPOCHS", "8")),
    "seq_len": int(os.getenv("LSTM_WEB_SEQ_LEN", "16")),
    "hidden_size": int(os.getenv("LSTM_WEB_HIDDEN", "16")),
    "passo_retreino": int(os.getenv("LSTM_WEB_PASSO_RETREINO", "2000")),
}

PAR_VALIDO = re.compile(r"^[A-Z0-9]{2,20}/[A-Z0-9]{2,20}$")
TIMEFRAMES_VALIDOS = {"1h", "4h", "1d", "1w"}

# Rate limit simples em memória para tentativas de login por IP.
_MAX_TENTATIVAS_LOGIN = 5
_JANELA_TENTATIVAS_SEGUNDOS = 300
_TENTATIVAS_LOGIN = {}


def _login_bloqueado(ip: str) -> bool:
    agora = time.monotonic()
    tentativas = [
        t for t in _TENTATIVAS_LOGIN.get(ip, [])
        if agora - t < _JANELA_TENTATIVAS_SEGUNDOS
    ]
    _TENTATIVAS_LOGIN[ip] = tentativas
    return len(tentativas) >= _MAX_TENTATIVAS_LOGIN


def _registrar_falha_login(ip: str) -> None:
    _TENTATIVAS_LOGIN.setdefault(ip, []).append(time.monotonic())


def _limpar_falhas_login(ip: str) -> None:
    _TENTATIVAS_LOGIN.pop(ip, None)

CAMINHO_RELATORIO_TCC = os.path.join(DIR_RAIZ, "resultados", "comparacao_tcc.json")
CAMINHO_RELATORIO_GEMINI = os.path.join(
    DIR_RAIZ, "resultados", "backtesting_gemini_tcc.json"
)
CAMINHO_RELATORIO_GEMINI_INTRADAY = os.path.join(
    DIR_RAIZ, "resultados", "backtesting_gemini_intraday_tcc.json"
)


def _carregar_relatorio_json(caminho, rotulo):
    """Carrega um relatório JSON consolidado com lista de resultados."""
    if not os.path.exists(caminho):
        return None
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        if isinstance(dados, dict) and isinstance(dados.get("resultados"), list):
            return dados
    except Exception as erro:
        print(f"[Webapp] Não foi possível ler {rotulo}: {erro}")
    return None


def _carregar_relatorio_tcc():
    """Carrega o relatório consolidado 4x4 salvo pelo CLI, se existir."""
    return _carregar_relatorio_json(CAMINHO_RELATORIO_TCC, "o relatório TCC")


def _carregar_relatorio_gemini():
    """Carrega o relatório oficial do Gemini (1d), se existir."""
    return _carregar_relatorio_json(CAMINHO_RELATORIO_GEMINI, "o relatório Gemini")


def _carregar_relatorio_gemini_intraday():
    """Carrega o relatório exploratório do Gemini (1h/4h), se existir."""
    return _carregar_relatorio_json(
        CAMINHO_RELATORIO_GEMINI_INTRADAY,
        "o relatório Gemini intraday",
    )


def _resumo_gemini(relatorio):
    """Resume os critérios do TCC para o experimento multimodal real."""
    if not relatorio:
        return None
    resultados = [
        r for r in relatorio.get("resultados", []) if isinstance(r, dict) and "erro" not in r
    ]
    if not resultados:
        return None

    def _valor(item, campo):
        valor = item.get(campo)
        return float(valor) if isinstance(valor, (int, float)) else None

    acuracia_ok = sum(
        1 for r in resultados
        if (_valor(r, "acuracia_prazo_estimado_pct") or 0) > 50
    )
    retorno_ok = sum(
        1 for r in resultados
        if (_valor(r, "retorno_acumulado_pct") or 0) >= 0
    )
    return {
        "total_ativos": len(resultados),
        "acuracia_ok": acuracia_ok,
        "retorno_ok": retorno_ok,
        "criterio_acuracia": acuracia_ok >= 3,
        "criterio_retorno": retorno_ok == len(resultados),
    }


def _resumo_gemini_intraday(relatorio):
    """Resume o experimento exploratório 1h/4h por combinação e timeframe."""
    if not relatorio:
        return None
    resultados = [
        r for r in relatorio.get("resultados", []) if isinstance(r, dict) and "erro" not in r
    ]
    if not resultados:
        return None

    def _valor(item, campo):
        valor = item.get(campo)
        return float(valor) if isinstance(valor, (int, float)) else None

    def _bh(item):
        buy_and_hold = item.get("buy_and_hold") or {}
        return _valor(buy_and_hold, "retorno_total_pct")

    def _resumir(subresultados):
        acuracia_ok = sum(
            1 for r in subresultados
            if _valor(r, "acuracia_prazo_estimado_pct") is not None
            and _valor(r, "acuracia_prazo_estimado_pct") > 50
        )
        retorno_ok = sum(
            1 for r in subresultados
            if _valor(r, "retorno_acumulado_pct") is not None
            and _valor(r, "retorno_acumulado_pct") >= 0
        )
        bh_ok = sum(
            1 for r in subresultados
            if (_valor(r, "retorno_acumulado_pct") is not None
                and _bh(r) is not None
                and _valor(r, "retorno_acumulado_pct") > _bh(r))
        )
        return {
            "total": len(subresultados),
            "acuracia_ok": acuracia_ok,
            "retorno_ok": retorno_ok,
            "bh_ok": bh_ok,
        }

    resumo_geral = _resumir(resultados)
    timeframes = sorted({r.get("intervalo") for r in resultados if r.get("intervalo")})
    por_timeframe = []
    for timeframe in timeframes:
        sub = [r for r in resultados if r.get("intervalo") == timeframe]
        item = {"intervalo": timeframe}
        item.update(_resumir(sub))
        por_timeframe.append(item)

    return {
        **resumo_geral,
        "por_timeframe": por_timeframe,
        "periodo": relatorio.get("periodo"),
    }


def _obter_csrf_token() -> str:
    """Gera/retorna o token CSRF da sessão."""
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = _obter_csrf_token


@app.before_request
def _proteger_csrf():
    """Valida CSRF em todos os POSTs quando habilitado."""
    if request.method == "POST" and app.config.get("CSRF_ENABLED", True):
        token_sessao = session.get("_csrf_token", "")
        token_form = request.form.get("_csrf_token", "")
        if not token_sessao or not token_form or not hmac.compare_digest(token_sessao, token_form):
            abort(400, description="Token CSRF inválido ou ausente.")


@app.after_request
def _cabecalhos_seguranca(resposta):
    """Adiciona cabeçalhos básicos de segurança."""
    resposta.headers.setdefault("X-Content-Type-Options", "nosniff")
    resposta.headers.setdefault("X-Frame-Options", "DENY")
    resposta.headers.setdefault("Referrer-Policy", "same-origin")
    return resposta


# ── Rotas de Autenticação ──────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip = request.remote_addr or "desconhecido"
        if _login_bloqueado(ip):
            flash(
                "Muitas tentativas de login. Aguarde alguns minutos e tente novamente.",
                "danger",
            )
            return render_template(
                "login.html", usando_senha_padrao=USANDO_SENHA_PADRAO
            ), 429

        usuario = request.form.get("usuario", "").strip()
        senha = request.form.get("senha", "").strip()

        if verificar_credenciais(usuario, senha):
            _limpar_falhas_login(ip)
            session.clear()
            session["usuario_autenticado"] = usuario
            session["_csrf_token"] = secrets.token_urlsafe(32)
            session.permanent = True
            if USANDO_SENHA_PADRAO:
                flash(
                    "Login efetuado com a senha padrão de desenvolvimento. "
                    "Defina WEB_ADMIN_PASS no .env para uso seguro.",
                    "warning",
                )
            else:
                flash(f"Bem-vindo, {usuario}! Sessão iniciada com sucesso.", "success")
            return redirect(url_for("index"))

        _registrar_falha_login(ip)
        flash("Credenciais inválidas. Verifique o usuário e senha.", "danger")

    return render_template("login.html", usando_senha_padrao=USANDO_SENHA_PADRAO)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    flash("Sessão encerrada com sucesso.", "success")
    return redirect(url_for("login"))


# ── Rota Principal / Dashboard ─────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    todas_analises = listar_analises()
    todos_backtests = listar_resultados_backtest()
    return render_template(
        "dashboard.html",
        total_analises=len(todas_analises),
        total_backtests=len(todos_backtests),
        ultimas_analises=todas_analises[:5],
        modelo_gemini=MODELO_GEMINI,
    )


# ── Rota de Nova Análise ───────────────────────────────────────────────────────

@app.route("/analise", methods=["GET", "POST"])
@login_required
def analise():
    resultado = None
    par = "BTC/USDT"
    intervalo = "1d"
    quantidade = 90

    if request.method == "POST":
        par = request.form.get("par", "BTC/USDT").strip().upper()
        intervalo = request.form.get("intervalo", "1d").strip().lower()
        try:
            quantidade = int(request.form.get("quantidade", 90))
        except ValueError:
            quantidade = 90

        if not PAR_VALIDO.match(par):
            flash("Par inválido. Use o formato BASE/QUOTE, ex.: BTC/USDT.", "danger")
            return render_template(
                "analise.html", resultado=None, par_selecionado=par,
                intervalo_selecionado=intervalo, quantidade_selecionada=quantidade,
            )

        if intervalo not in TIMEFRAMES_VALIDOS:
            flash("Timeframe inválido. Use 1h, 4h, 1d ou 1w.", "danger")
            return render_template(
                "analise.html", resultado=None, par_selecionado=par,
                intervalo_selecionado=intervalo, quantidade_selecionada=quantidade,
            )

        quantidade = max(50, min(quantidade, 1000))

        try:
            # 1. Coleta dados
            df_historico = buscar_dados_historicos(par, intervalo, quantidade)
            ticker = buscar_preco_atual(par)

            # 2. Indicadores técnicos
            df_enriquecido = enriquecer_dataframe(df_historico)
            indicadores = resumir_indicadores(df_enriquecido)

            # 3. Gráfico para o Gemini
            caminho_grafico = os.path.join(DIR_RAIZ, "resultados", "grafico_bitcoin.png")
            gerar_grafico_candlestick(
                df_enriquecido,
                nome_arquivo=caminho_grafico,
                titulo=f"{par} — {intervalo}"
            )

            # 4. Análise com Gemini (fallback MANTER em caso de erro)
            analise_ia = analisar_mercado_com_gemini(
                caminho_grafico, indicadores, par=par
            )

            # 5. Gráfico Plotly interativo para a interface web
            grafico_html = _gerar_grafico_plotly(df_enriquecido, include_plotlyjs="cdn")

            resultado = {
                "par": par,
                "intervalo": intervalo,
                "quantidade": quantidade,
                "data": datetime.now().strftime("%d/%m/%Y às %H:%M:%S"),
                "modelo": MODELO_GEMINI,
                "ticker": ticker,
                "indicadores": indicadores,
                "analise_ia": analise_ia,
                "grafico_html": grafico_html,
            }

            # Persiste no histórico (uma falha de persistência não invalida
            # a análise já concluída)
            try:
                salvar_analise({
                    "par": par,
                    "intervalo": intervalo,
                    "quantidade": quantidade,
                    "preco_atual": ticker.get("preco"),
                    "variacao_24h": ticker.get("variacao_24h"),
                    "acao": analise_ia.get("acao", "MANTER"),
                    "justificativa": analise_ia.get("justificativa", ""),
                    "preco_alvo": analise_ia.get("preco_alvo"),
                    "tempo_estimado": analise_ia.get("tempo_estimado"),
                    "indicadores": indicadores,
                    "fallback": analise_ia.get("fallback", False),
                })
            except Exception as erro_hist:
                print(f"[Webapp] Erro ao salvar análise no histórico: {erro_hist}")
                flash(
                    "A análise foi concluída, mas não foi possível salvá-la no histórico.",
                    "warning",
                )

            if analise_ia.get("fallback"):
                flash(
                    "A análise foi concluída com fallback MANTER: "
                    f"{analise_ia.get('motivo_fallback', 'erro na API')}",
                    "warning",
                )
            else:
                flash(f"Análise de {par} ({intervalo}) concluída com sucesso!", "success")

        except Exception as e:
            motivo = f"{type(e).__name__}: {e}"
            try:
                salvar_analise({
                    "par": par,
                    "intervalo": intervalo,
                    "quantidade": quantidade,
                    "acao": "MANTER",
                    "justificativa": (
                        f"Fallback MANTER: falha ao executar a análise: {motivo}"
                    ),
                    "preco_alvo": None,
                    "tempo_estimado": None,
                    "indicadores": {},
                    "fallback": True,
                })
            except Exception as erro_hist:
                print(f"[Webapp] Erro ao salvar fallback no histórico: {erro_hist}")
            flash(
                f"Erro ao executar análise para {par}: {motivo}. "
                "Fallback MANTER registrado no histórico.",
                "danger",
            )

    return render_template(
        "analise.html",
        resultado=resultado,
        par_selecionado=par,
        intervalo_selecionado=intervalo,
        quantidade_selecionada=quantidade,
    )


# ── Rota de Histórico ──────────────────────────────────────────────────────────

@app.route("/historico")
@login_required
def historico():
    analises = listar_analises()
    return render_template("historico.html", analises=analises)


# ── Rota de Backtesting ────────────────────────────────────────────────────────

@app.route("/backtesting", methods=["GET", "POST"])
@login_required
def backtesting():
    resultado_simulacao = None
    par = "BTC/USDT"
    intervalo = "1d"

    if request.method == "POST":
        par = request.form.get("par", "BTC/USDT").strip().upper()
        intervalo = request.form.get("intervalo", "1d").strip().lower()

        if not PAR_VALIDO.match(par):
            flash("Par inválido. Use o formato BASE/QUOTE, ex.: BTC/USDT.", "danger")
        elif intervalo not in TIMEFRAMES_VALIDOS:
            flash("Timeframe inválido. Use 1h, 4h, 1d ou 1w.", "danger")
        else:
            try:
                resultado_simulacao = comparar_estrategias(
                    par=par,
                    intervalo=intervalo,
                    **LSTM_WEB_CONFIG,
                )

                try:
                    salvar_resultado_backtest(resultado_simulacao)
                except Exception as erro_hist:
                    print(f"[Webapp] Erro ao salvar backtesting: {erro_hist}")
                    flash(
                        "A simulação foi executada, mas não foi possível "
                        "salvá-la no histórico.",
                        "warning",
                    )

                if resultado_simulacao.get("erro"):
                    flash(
                        f"Backtesting de {par} ({intervalo}) retornou erro: "
                        f"{resultado_simulacao['erro']}",
                        "danger",
                    )
                else:
                    flash(
                        f"Simulação de {par} ({intervalo}) executada com sucesso!",
                        "success",
                    )
            except Exception as e:
                flash(f"Erro na simulação de backtesting: {e}", "danger")
                resultado_simulacao = {
                    "erro": str(e), "par": par, "intervalo": intervalo
                }
                try:
                    salvar_resultado_backtest(resultado_simulacao)
                except Exception as erro_hist:
                    print(f"[Webapp] Erro ao salvar erro de backtesting: {erro_hist}")

    relatorio_gemini = _carregar_relatorio_gemini()
    relatorio_gemini_intraday = _carregar_relatorio_gemini_intraday()
    return render_template(
        "backtesting.html",
        resultado_simulacao=resultado_simulacao,
        historico_backtest=listar_resultados_backtest(),
        relatorio_tcc=_carregar_relatorio_tcc(),
        relatorio_gemini=relatorio_gemini,
        resumo_gemini=_resumo_gemini(relatorio_gemini),
        relatorio_gemini_intraday=relatorio_gemini_intraday,
        resumo_gemini_intraday=_resumo_gemini_intraday(relatorio_gemini_intraday),
        pares_tcc=PARES_TCC,
        timeframes_tcc=TIMEFRAMES_TCC,
        par_selecionado=par,
        intervalo_selecionado=intervalo,
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Iniciando Servidor Web CriptoIA (Flask) — TCC II")
    print("  Acesse no navegador: http://127.0.0.1:5000")
    print("  Configure WEB_ADMIN_USER, WEB_ADMIN_PASS e FLASK_SECRET_KEY no .env")
    print("=" * 60 + "\n")
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
