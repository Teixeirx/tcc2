"""
analisador_ia.py — Módulo de análise de mercado com Google Gemini.

Envia o gráfico de candlestick + indicadores técnicos numéricos ao modelo
Gemini e recebe uma recomendação estruturada em JSON:
  - acao           : 'COMPRAR', 'VENDER' ou 'MANTER'
  - justificativa  : Explicação detalhada da recomendação
  - preco_alvo     : Estimativa de preço-alvo da operação
  - tempo_estimado : Estimativa de tempo para geração de lucro

RNF06/CT06: qualquer falha (JSON inválido, campo ausente, timeout, chave
inválida) retorna um fallback estruturado com acao="MANTER", sem propagar
exceção para o chamador.
"""

import os
import json
import re
import time
import PIL.Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Modelo Gemini a ser utilizado (padrão gemini-3.5-flash-lite)
MODELO_GEMINI = os.getenv("MODELO_GEMINI", "gemini-3.5-flash-lite").strip()

ACOES_VALIDAS = {"COMPRAR", "VENDER", "MANTER"}


def _fallback_manter(motivo: str) -> dict:
    """Fallback estruturado exigido pelo RNF06/CT06."""
    return {
        "acao": "MANTER",
        "justificativa": f"Fallback aplicado: {motivo}",
        "preco_alvo": None,
        "tempo_estimado": None,
        "fallback": True,
        "motivo_fallback": motivo,
    }


def _construir_prompt(indicadores: dict, par: str = "BTC/USDT") -> str:
    """
    Constrói o prompt textual enviado ao Gemini com os dados numéricos
    dos indicadores técnicos.

    Parâmetros:
        indicadores : Dicionário retornado por analise_tecnica.resumir_indicadores()
        par         : Par analisado (ex.: BTC/USDT)

    Retorna:
        String com o prompt completo
    """
    rsi = indicadores.get("rsi14")
    mm20 = indicadores.get("mm20")
    mm50 = indicadores.get("mm50")
    macd = indicadores.get("macd")
    sinal = indicadores.get("sinal_macd")
    boll_sup = indicadores.get("bollinger_superior")
    boll_inf = indicadores.get("bollinger_inferior")
    fechamento = indicadores.get("preco_fechamento")

    # Avaliações qualitativas dos indicadores
    situacao_rsi = (
        "sobrecomprado (possível reversão de baixa)" if rsi and rsi > 70
        else "sobrevendido (possível reversão de alta)" if rsi and rsi < 30
        else "neutro"
    )

    tendencia_macd = (
        "sinal de compra (MACD acima da linha de sinal)" if macd and sinal and macd > sinal
        else "sinal de venda (MACD abaixo da linha de sinal)" if macd and sinal and macd < sinal
        else "indefinida"
    )

    posicao_medias = (
        "tendência de alta (MM20 acima da MM50)" if mm20 and mm50 and mm20 > mm50
        else "tendência de baixa (MM20 abaixo da MM50)" if mm20 and mm50 and mm20 < mm50
        else "indefinida"
    )

    def _num(valor, formato: str) -> str:
        return format(valor, formato) if isinstance(valor, (int, float)) else "N/A"

    prompt = f"""Você é um trader especialista em análise técnica de criptomoedas.
Analise o gráfico de candlestick do par {par} anexo junto com os dados numéricos abaixo:

=== INDICADORES TÉCNICOS (último candle) ===
- Preço de Fechamento: ${_num(fechamento, ',.2f')} USDT
- RSI (14): {_num(rsi, '.2f')} → {situacao_rsi}
- MACD: {_num(macd, '.4f')} | Linha de Sinal: {_num(sinal, '.4f')} → {tendencia_macd}
- Média Móvel 20: ${_num(mm20, ',.2f')} | Média Móvel 50: ${_num(mm50, ',.2f')} → {posicao_medias}
- Banda de Bollinger Superior: ${_num(boll_sup, ',.2f')}
- Banda de Bollinger Inferior: ${_num(boll_inf, ',.2f')}
============================================

Com base na imagem do gráfico E nos indicadores acima, determine:

1. A ação recomendada: COMPRAR, VENDER ou MANTER
2. Uma justificativa técnica clara e detalhada (mínimo 2 parágrafos)
3. Se a ação for COMPRAR ou VENDER: um preço-alvo estimado para encerrar a operação com lucro
4. Uma estimativa de tempo para que a operação gere lucro (ex: "3 a 7 dias", "2 a 4 semanas")

IMPORTANTE: Responda EXCLUSIVAMENTE em formato JSON válido, sem texto adicional antes ou depois.
Use exatamente este formato:

{{
  "acao": "COMPRAR",
  "justificativa": "Justificativa técnica detalhada aqui.",
  "preco_alvo": "$98.500,00",
  "tempo_estimado": "7 a 14 dias"
}}

Se a ação for MANTER, use null para preco_alvo e tempo_estimado.
A chave 'acao' deve conter APENAS uma das palavras: COMPRAR, VENDER ou MANTER.
"""
    return prompt


def _extrair_json_da_resposta(texto: str) -> dict:
    """
    Extrai e valida o JSON da resposta do Gemini.

    Lida com blocos Markdown e valida os campos obrigatórios. Em caso de
    JSON inválido, tipo incorreto ou campo obrigatório ausente, levanta
    ValueError para que o chamador aplique o fallback MANTER.

    Retorna:
        Dicionário com acao, justificativa, preco_alvo e tempo_estimado.
    """
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("Resposta vazia da API Gemini.")

    texto = texto.strip()

    # Remove marcações Markdown de código (```json ... ``` ou ``` ... ```)
    texto = re.sub(r"^```(?:json)?\s*", "", texto)
    texto = re.sub(r"\s*```$", "", texto)
    texto = texto.strip()

    dados = json.loads(texto)
    if not isinstance(dados, dict):
        raise ValueError("A resposta JSON não é um objeto.")

    if "acao" not in dados:
        raise ValueError("Campo obrigatório ausente: acao.")
    if not isinstance(dados["acao"], str):
        raise ValueError("Campo 'acao' deve ser uma string.")

    acao = dados["acao"].upper().strip()
    aviso = None
    if acao not in ACOES_VALIDAS:
        aviso = f"acao inválida '{dados['acao']}'; normalizada para MANTER"
        acao = "MANTER"

    justificativa = dados.get("justificativa")
    if not isinstance(justificativa, str) or not justificativa.strip():
        raise ValueError("Campo obrigatório ausente ou vazio: justificativa.")

    if "preco_alvo" not in dados:
        raise ValueError("Campo obrigatório ausente: preco_alvo.")
    if "tempo_estimado" not in dados:
        raise ValueError("Campo obrigatório ausente: tempo_estimado.")

    preco_alvo = dados.get("preco_alvo")
    tempo_estimado = dados.get("tempo_estimado")

    if acao in {"COMPRAR", "VENDER"}:
        if preco_alvo in (None, ""):
            raise ValueError("preco_alvo é obrigatório para COMPRAR/VENDER.")
        if tempo_estimado in (None, ""):
            raise ValueError("tempo_estimado é obrigatório para COMPRAR/VENDER.")

    resultado = {
        "acao": acao,
        "justificativa": justificativa.strip(),
        "preco_alvo": preco_alvo,
        "tempo_estimado": tempo_estimado,
    }
    if aviso:
        resultado["aviso"] = aviso
        resultado["fallback"] = True
    return resultado


def analisar_mercado_com_gemini(
    caminho_grafico: str,
    indicadores: dict,
    par: str = "BTC/USDT",
) -> dict:
    """
    Envia o gráfico + indicadores ao Google Gemini e retorna a análise.

    Parâmetros:
        caminho_grafico : Caminho para o arquivo PNG do gráfico
        indicadores     : Dicionário de indicadores de analise_tecnica.resumir_indicadores()
        par             : Par analisado (padrão: BTC/USDT)

    Retorna:
        Dicionário com: acao, justificativa, preco_alvo e tempo_estimado.
        Em qualquer falha, retorna fallback estruturado com acao="MANTER".
    """
    chave_api = os.getenv("GEMINI_API_KEY", "").strip()
    if "AIza" in chave_api:
        chave_api = chave_api[chave_api.index("AIza"):].strip()

    if not chave_api or chave_api == "sua_chave_gemini_aqui" or "adicione aqui" in chave_api.lower():
        return _fallback_manter(
            "a variável GEMINI_API_KEY não está configurada no arquivo .env."
        )

    try:
        cliente = genai.Client(api_key=chave_api)

        imagem = PIL.Image.open(caminho_grafico)
        prompt = _construir_prompt(indicadores, par=par)

        modelos_para_tentar = [MODELO_GEMINI]
        if "lite" in MODELO_GEMINI:
            modelos_para_tentar.append(MODELO_GEMINI.replace("-lite", ""))
        elif "flash" in MODELO_GEMINI:
            modelos_para_tentar.append(f"{MODELO_GEMINI}-lite")

        ultimo_erro = None
        for modelo_atual in modelos_para_tentar:
            try:
                print(f"[Gemini] Enviando gráfico e indicadores para análise ({modelo_atual})...")
                resposta = cliente.models.generate_content(
                    model=modelo_atual,
                    contents=[prompt, imagem]
                )
                print("[Gemini] Análise recebida. Processando resultado...")
                return _extrair_json_da_resposta(getattr(resposta, "text", ""))
            except (json.JSONDecodeError, ValueError, TypeError) as erro_validacao:
                print(f"[Gemini] Resposta inválida: {erro_validacao}")
                return _fallback_manter(f"Resposta inválida da API: {erro_validacao}")
            except Exception as erro:
                ultimo_erro = erro
                msg_erro = str(erro).lower()
                if "503" in msg_erro or "unavailable" in msg_erro or "demand" in msg_erro:
                    print(f"[Gemini] Modelo {modelo_atual} temporariamente ocupado (503). Tentando alternativa...")
                    time.sleep(1.5)
                    continue
                else:
                    break

        print(f"[Gemini] Erro na chamada à API: {ultimo_erro}")
        return _fallback_manter(f"falha ao conectar com o Google Gemini: {ultimo_erro}")

    except Exception as erro_geral:
        print(f"[Gemini] Erro geral: {erro_geral}")
        return _fallback_manter(f"falha ao conectar com o Google Gemini: {erro_geral}")
