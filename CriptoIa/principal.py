"""
principal.py -- Ponto de entrada do sistema CriptoIA.

Orquestra o fluxo completo de análise:
  1. Coleta dados históricos da Binance (OHLCV via API pública)
  2. Calcula indicadores técnicos (RSI, MACD, Médias Móveis, Bollinger)
  3. Gera gráfico de candlestick (PNG para o Gemini)
  4. Envia imagem + indicadores ao Google Gemini para análise
  5. Gera e abre o dashboard web interativo no navegador

Também oferece modos de linha de comando para backtesting, comparação
com Buy-and-Hold/LSTM e análise paralela de múltiplos ativos.

Uso:
    python principal.py
    python principal.py --par ETH/USDT --intervalo 4h --quantidade 60
    python principal.py --backtesting --par BTC/USDT --intervalo 1d
    python principal.py --backtesting --grade
    python principal.py --comparacao --par BTC/USDT --intervalo 1d
    python principal.py --multiplos --pares BTC/USDT,ETH/USDT --timeframes 1d,4h
"""

import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime, timezone

# Garante saida UTF-8 no terminal do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Adiciona a pasta raiz ao path para importar backend/ e frontend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.coleta_dados import buscar_dados_historicos, buscar_preco_atual
from backend.analise_tecnica import enriquecer_dataframe, resumir_indicadores
from backend.gerador_grafico import gerar_grafico_candlestick
from backend.analisador_ia import analisar_mercado_com_gemini
from frontend.gerador_dashboard import gerar_dashboard


def _fallback_manter(motivo: str) -> dict:
    """Retorna o fallback JSON estruturado exigido pelo RNF06/CT06."""
    return {
        "acao": "MANTER",
        "justificativa": f"Falha ao executar a análise: {motivo}",
        "preco_alvo": None,
        "tempo_estimado": None,
        "fallback": True,
        "motivo_fallback": motivo,
    }


def _salvar_fallback(resultado: dict) -> str:
    """Persiste um fallback estruturado para auditoria."""
    diretorio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, "fallback_analise.json")
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2, default=str)
    return caminho


def executar_analise(par: str = "BTC/USDT", intervalo: str = "1d", quantidade: int = 90):
    """
    Executa o fluxo completo de análise de criptomoedas.

    Em qualquer falha de coleta, indicadores, IA ou dashboard, imprime uma
    mensagem descritiva, salva um fallback JSON com acao="MANTER" e retorna
    um dicionário estruturado, sem traceback não tratado (RNF06/CT06).

    Retorna:
        Dicionário com status, parâmetros, análise, caminhos e erro (se houver).
    """
    print("\n" + "=" * 55)
    print("  CriptoIA -- Sistema de Analise com Gemini")
    print("=" * 55)
    print(f"  Par:       {par}")
    print(f"  Intervalo: {intervalo}")
    print(f"  Candles:   {quantidade}")
    print("=" * 55 + "\n")

    horario_analise = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
    resultado = {
        "status": "ok",
        "par": par,
        "intervalo": intervalo,
        "quantidade": quantidade,
        "horario_analise": horario_analise,
    }

    try:
        # ── ETAPA 1: Coleta de Dados ─────────────────────────────────────────
        print("[1/5] Coletando dados históricos da Binance...")
        df_historico = buscar_dados_historicos(par, intervalo, quantidade)

        print("[1/5] Buscando preço em tempo real...")
        dados_ticker = buscar_preco_atual(par)

        # ── ETAPA 2: Análise Técnica ─────────────────────────────────────────
        print("\n[2/5] Calculando indicadores técnicos...")
        df_enriquecido = enriquecer_dataframe(df_historico)
        indicadores = resumir_indicadores(df_enriquecido)

        print(f"       RSI(14):  {indicadores.get('rsi14', 'N/A')}")
        print(f"       MACD:     {indicadores.get('macd', 'N/A')}")
        print(f"       MM20:     ${indicadores.get('mm20') or 0:,.2f}")
        print(f"       MM50:     ${indicadores.get('mm50') or 0:,.2f}")

        # ── ETAPA 3: Geração do Gráfico ──────────────────────────────────────
        print("\n[3/5] Gerando gráfico técnico para análise...")
        caminho_grafico = gerar_grafico_candlestick(
            df_enriquecido,
            titulo=f"{par} — Análise Técnica ({intervalo})"
        )

        # ── ETAPA 4: Análise com IA ──────────────────────────────────────────
        print("\n[4/5] Enviando para análise com Google Gemini...")
        analise_ia = analisar_mercado_com_gemini(
            caminho_grafico, indicadores, par=par
        )

        print("\n" + "-" * 55)
        print("  RESULTADO DA ANALISE (Gemini):")
        print("-" * 55)
        print(f"  Acao:            {analise_ia.get('acao', 'N/A')}")
        print(f"  Preco Alvo:      {analise_ia.get('preco_alvo', 'N/A')}")
        print(f"  Tempo p/ Lucro:  {analise_ia.get('tempo_estimado', 'N/A')}")
        justif = str(analise_ia.get('justificativa', '')).encode('ascii', 'replace').decode('ascii')
        print(f"  Justificativa:   {justif[:120]}...")
        print("-" * 55)

        # ── ETAPA 5: Geração do Dashboard ────────────────────────────────────
        print("\n[5/5] Gerando dashboard web interativo...")
        caminho_dashboard = gerar_dashboard(
            df=df_enriquecido,
            dados_ticker=dados_ticker,
            analise_ia=analise_ia,
            indicadores=indicadores,
            par=par,
            horario_analise=horario_analise,
        )

        caminho_absoluto = os.path.abspath(caminho_dashboard).replace(os.sep, "/")
        url_dashboard = f"file:///{caminho_absoluto}"

        print(f"\nDashboard gerado: {caminho_absoluto}")
        print("Abrindo no navegador...")
        webbrowser.open(url_dashboard)

        resultado.update({
            "ticker": dados_ticker,
            "indicadores": indicadores,
            "analise_ia": analise_ia,
            "caminho_grafico": caminho_grafico,
            "caminho_dashboard": caminho_dashboard,
        })

        print("\n" + "=" * 55)
        print("  Analise concluida com sucesso!")
        print("=" * 55 + "\n")
        return resultado

    except Exception as erro:
        motivo = f"{type(erro).__name__}: {erro}"
        print("\n" + "!" * 55)
        print("  ERRO NA ANALISE")
        print(f"  {motivo}")
        print("  Retornando fallback JSON com acao='MANTER'.")
        print("!" * 55 + "\n")

        resultado.update({
            "status": "erro",
            "erro": motivo,
            "analise_ia": _fallback_manter(motivo),
        })
        try:
            resultado["fallback_json"] = _salvar_fallback(resultado)
        except Exception as erro_fallback:
            print(f"  [AVISO] Não foi possível salvar o fallback: {erro_fallback}")
        return resultado


def _parse_data(valor: str, fim_do_dia: bool = False) -> datetime:
    """Converte YYYY-MM-DD em datetime UTC.

    Quando ``fim_do_dia`` é True, a data representa o último instante do dia,
    garantindo que o período final (ex.: 30/06/2026) inclua todos os candles
    daquele dia.
    """
    data = datetime.strptime(valor, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if fim_do_dia:
        data = data.replace(hour=23, minute=59, second=59)
    return data


def _parse_intervalos(valor: str) -> list:
    """Valida e normaliza uma lista de timeframes separada por vírgulas."""
    if not valor:
        return ["1d"]

    validos = {"1h", "4h", "1d", "1w"}
    intervalos = [item.strip() for item in valor.split(",") if item.strip()]
    invalidos = [item for item in intervalos if item not in validos]
    if invalidos:
        raise SystemExit(
            f"Timeframe(s) inválido(s): {', '.join(invalidos)}. "
            f"Use: {', '.join(sorted(validos))}."
        )
    return list(dict.fromkeys(intervalos))


def _salvar_relatorio_texto(nome_arquivo: str, conteudo: str) -> str:
    """Salva um relatório textual em resultados/ com codificação UTF-8."""
    diretorio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, nome_arquivo)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)
    return caminho


def executar_modo_backtesting(args) -> None:
    """Executa backtesting único ou a grade completa do TCC."""
    from backend.backtesting import (
        executar_backtesting,
        executar_backtesting_completo,
        formatar_relatorio_backtesting,
        salvar_resultados_backtesting,
        PARES_TCC,
        TIMEFRAMES_TCC,
    )

    inicio = _parse_data(args.inicio)
    fim = _parse_data(args.fim, fim_do_dia=True)

    if args.grade:
        resultados = executar_backtesting_completo(
            pares=PARES_TCC,
            timeframes=TIMEFRAMES_TCC,
            data_inicio=inicio,
            data_fim=fim,
            max_workers=args.max_workers,
        )
    else:
        resultados = [executar_backtesting(args.par, args.intervalo, inicio, fim)]

    relatorio = formatar_relatorio_backtesting(resultados)
    print(relatorio)
    caminhos = salvar_resultados_backtesting(
        resultados,
        periodo={
            "inicio": inicio.isoformat(),
            "fim": fim.isoformat(),
        },
    )
    caminhos["relatorio"] = _salvar_relatorio_texto(
        "backtesting_relatorio.txt", relatorio
    )
    print(f"\nResultados salvos em: {caminhos}")


def executar_modo_comparacao(args) -> None:
    """Executa comparação única ou a grade completa do TCC."""
    from backend.comparacao import (
        comparar_estrategias,
        comparar_todos_pares,
        formatar_relatorio_comparacao,
        salvar_resultados_comparacao,
        PARES_TCC,
        TIMEFRAMES_TCC,
    )

    inicio = _parse_data(args.inicio)
    fim = _parse_data(args.fim, fim_do_dia=True)
    kwargs = {
        "epochs": args.epochs,
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
        "passo_retreino": args.passo_retreino,
    }

    if args.grade:
        resultados = comparar_todos_pares(
            pares=PARES_TCC,
            timeframes=TIMEFRAMES_TCC,
            data_inicio=inicio,
            data_fim=fim,
            max_workers=1,
            **kwargs,
        )
    else:
        resultados = [
            comparar_estrategias(
                args.par, args.intervalo, inicio, fim, **kwargs
            )
        ]

    relatorio = formatar_relatorio_comparacao(resultados)
    print(relatorio)
    caminhos = salvar_resultados_comparacao(
        resultados,
        periodo={
            "inicio": inicio.isoformat(),
            "fim": fim.isoformat(),
        },
    )
    caminhos["relatorio"] = _salvar_relatorio_texto(
        "comparacao_relatorio.txt", relatorio
    )
    print(f"\nResultados salvos em: {caminhos}")


def executar_modo_multiplos(args) -> None:
    """Executa análises paralelas de múltiplos pares/timeframes."""
    from backend.multiplos_ativos import (
        analisar_multiplos,
        gerar_relatorio_consolidado,
        salvar_relatorio_csv,
    )

    pares = [p.strip().upper() for p in args.pares.split(",") if p.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    combinacoes = [
        {"par": par, "intervalo": tf, "quantidade": args.quantidade}
        for par in pares
        for tf in timeframes
    ]

    resultados = analisar_multiplos(
        combinacoes,
        max_workers=args.max_workers,
        gerar_graficos=args.usar_ia,
        usar_ia=args.usar_ia,
    )
    relatorio = gerar_relatorio_consolidado(resultados)
    print(relatorio["resumo_texto"])
    if args.salvar_csv:
        salvar_relatorio_csv(resultados)


def executar_modo_backtesting_gemini(args) -> None:
    """Executa o backtesting multimodal do Gemini."""
    from backend.backtesting_gemini import (
        executar_backtest_gemini_par,
        executar_backtest_gemini_grade,
        PARES_TCC,
    )

    inicio = _parse_data(args.inicio)
    fim = _parse_data(args.fim, fim_do_dia=True)

    if args.grade:
        intervalos = _parse_intervalos(args.intervalos)
        executar_backtest_gemini_grade(
            pares=PARES_TCC,
            passo_amostragem=args.passo,
            max_amostras_por_par=args.max_amostras,
            delay_segundos=args.delay,
            intervalos=intervalos,
            data_inicio=inicio,
            data_fim=fim,
            prefixo_saida=args.prefixo_saida,
        )
    else:
        if args.intervalos:
            raise SystemExit(
                "Use --grade com --intervalos; para um único par, use --intervalo."
            )
        executar_backtest_gemini_par(
            par=args.par,
            intervalo=args.intervalo,
            passo_amostragem=args.passo,
            max_amostras=args.max_amostras,
            delay_segundos=args.delay,
            data_inicio=inicio,
            data_fim=fim,
        )


def configurar_argumentos() -> argparse.Namespace:
    """Configura e retorna os argumentos da linha de comando."""
    analisador = argparse.ArgumentParser(
        description="CriptoIA — Análise de criptomoedas com Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python principal.py
  python principal.py --par BTC/USDT --intervalo 1d --quantidade 90
  python principal.py --par ETH/USDT --intervalo 4h --quantidade 60
  python principal.py --backtesting --par BTC/USDT --intervalo 1d
  python principal.py --backtesting --grade
  python principal.py --backtesting-gemini --grade --passo 7 --delay 2.0
  python principal.py --backtesting-gemini --grade --intervalos 1h,4h --inicio 2026-07-01 --fim 2026-10-31 --passo 3 --prefixo-saida backtesting_gemini_intraday_tcc
  python principal.py --comparacao --par BTC/USDT --intervalo 1d
  python principal.py --multiplos --pares BTC/USDT,ETH/USDT --timeframes 1d,4h
        """
    )
    analisador.add_argument(
        "--par",
        type=str,
        default="BTC/USDT",
        help="Par de negociação (padrão: BTC/USDT)"
    )
    analisador.add_argument(
        "--intervalo",
        type=str,
        default="1d",
        choices=["1h", "4h", "1d", "1w"],
        help="Timeframe dos candles (padrão: 1d)"
    )
    analisador.add_argument(
        "--intervalos", type=str, default=None,
        help="Com --backtesting-gemini --grade, timeframes separados por vírgula (ex.: 1h,4h). Padrão: 1d."
    )
    analisador.add_argument(
        "--prefixo-saida", type=str, default=None,
        help="Nome base dos arquivos JSON/CSV/TXT do backtesting Gemini (sem extensão)."
    )
    analisador.add_argument(
        "--quantidade",
        type=int,
        default=90,
        help="Número de candles a buscar (padrão: 90; mínimo 50 para indicadores)"
    )
    analisador.add_argument(
        "--backtesting", action="store_true",
        help="Executa backtesting em vez da análise multimodal"
    )
    analisador.add_argument(
        "--backtesting-gemini", action="store_true",
        help="Executa backtesting multimodal real com a API do Google Gemini"
    )
    analisador.add_argument(
        "--passo", type=int, default=7,
        help="Intervalo em dias para amostragem do backtesting Gemini (padrão: 7)"
    )
    analisador.add_argument(
        "--max-amostras", type=int, default=None,
        help="Limite máximo de amostras por par no backtesting Gemini"
    )
    analisador.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay em segundos entre chamadas da API Gemini (padrão: 2.0)"
    )
    analisador.add_argument(
        "--comparacao", action="store_true",
        help="Executa comparação CriptoIA x Buy-and-Hold x LSTM"
    )
    analisador.add_argument(
        "--multiplos", action="store_true",
        help="Executa análise paralela de múltiplos pares/timeframes"
    )
    analisador.add_argument(
        "--grade", action="store_true",
        help="Com --backtesting/--comparacao, executa os 4 pares x 4 timeframes do TCC"
    )
    analisador.add_argument(
        "--pares", type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT",
        help="Lista separada por vírgula para --multiplos"
    )
    analisador.add_argument(
        "--timeframes", type=str,
        default="1h,4h,1d,1w",
        help="Lista separada por vírgula para --multiplos"
    )
    analisador.add_argument("--inicio", type=str, default="2025-01-01",
                            help="Data inicial do backtesting (YYYY-MM-DD)")
    analisador.add_argument("--fim", type=str, default="2026-06-30",
                            help="Data final do backtesting (YYYY-MM-DD)")
    analisador.add_argument("--max-workers", type=int, default=2,
                            help="Threads para coleta/análises paralelas (padrão: 2)")
    analisador.add_argument("--usar-ia", action="store_true",
                            help="Com --multiplos, gera gráficos e chama o Gemini")
    analisador.add_argument("--salvar-csv", action="store_true",
                            help="Com --multiplos, salva relatório CSV")
    analisador.add_argument("--epochs", type=int, default=20,
                            help="Épocas de treino do LSTM (padrão: 20)")
    analisador.add_argument("--seq-len", type=int, default=24,
                            help="Tamanho da sequência do LSTM (padrão: 24)")
    analisador.add_argument("--hidden-size", type=int, default=32,
                            help="Tamanho oculto do LSTM (padrão: 32)")
    analisador.add_argument("--passo-retreino", type=int, default=1000,
                            help="Re-treino do LSTM a cada N candles (padrão: 1000)")
    return analisador.parse_args()


if __name__ == "__main__":
    args = configurar_argumentos()

    if args.backtesting_gemini:
        executar_modo_backtesting_gemini(args)
    elif args.backtesting:
        executar_modo_backtesting(args)
    elif args.comparacao:
        executar_modo_comparacao(args)
    elif args.multiplos:
        executar_modo_multiplos(args)
    else:
        executar_analise(
            par=args.par,
            intervalo=args.intervalo,
            quantidade=args.quantidade
        )
