"""
coleta_dados.py — Módulo de coleta de dados históricos e em tempo real da Binance.

Utiliza a biblioteca ccxt para se conectar à API pública da Binance
e retorna os dados como DataFrame do Pandas.
"""

import os
import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def obter_cliente_binance(forcar_publico: bool = False) -> ccxt.binance:
    """
    Cria e retorna um cliente autenticado (ou público) da Binance via ccxt.
    Se as chaves de API não estiverem configuradas ou se forcar_publico=True,
    usa modo público (permite leitura de dados de mercado sem autenticação).
    """
    if forcar_publico:
        return ccxt.binance({
            "apiKey": None,
            "secret": None,
            "enableRateLimit": True,
        })

    chave_api = os.getenv("BINANCE_API_KEY", "")
    chave_secreta = os.getenv("BINANCE_SECRET_KEY", "")

    # Usa modo público se as chaves forem placeholders ou estiverem ausentes
    if (
        not chave_api
        or "sua_chave" in chave_api.lower()
        or "adicione aqui" in chave_api.lower()
        or len(chave_api.strip()) < 10
    ):
        chave_api = None
        chave_secreta = None

    cliente = ccxt.binance({
        "apiKey": chave_api,
        "secret": chave_secreta,
        "enableRateLimit": True,
    })
    return cliente


def buscar_dados_historicos(
    par: str = "BTC/USDT",
    intervalo: str = "1d",
    quantidade: int = 90
) -> pd.DataFrame:
    """
    Busca dados históricos (candles OHLCV) da Binance.

    Parâmetros:
        par        : Par de negociação, ex: 'BTC/USDT'
        intervalo  : Timeframe dos candles, ex: '1d', '4h', '1h'
        quantidade : Número de candles a buscar

    Retorna:
        DataFrame com colunas: timestamp, Abertura, Maxima, Minima, Fechamento, Volume
    """
    cliente = obter_cliente_binance()
    print(f"[Binance] Buscando {quantidade} candles de {par} ({intervalo})...")

    try:
        dados_brutos = cliente.fetch_ohlcv(par, intervalo, limit=quantidade)
    except Exception as erro:
        msg = str(erro).lower()
        if "signature" in msg or "key" in msg or "1022" in msg:
            cliente = obter_cliente_binance(forcar_publico=True)
            dados_brutos = cliente.fetch_ohlcv(par, intervalo, limit=quantidade)
        else:
            raise

    df = pd.DataFrame(
        dados_brutos,
        columns=["timestamp", "Abertura", "Maxima", "Minima", "Fechamento", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    print(f"[Binance] {len(df)} candles carregados com sucesso.")
    return df


def buscar_preco_atual(par: str = "BTC/USDT") -> dict:
    """
    Busca o preço atual do ativo em tempo real via ticker da Binance.

    Parâmetros:
        par : Par de negociação, ex: 'BTC/USDT'

    Retorna:
        Dicionário com 'preco', 'variacao_24h', 'volume_24h', 'maxima_24h', 'minima_24h'
    """
    cliente = obter_cliente_binance()
    print(f"[Binance] Buscando preço atual de {par}...")

    try:
        ticker = cliente.fetch_ticker(par)
    except Exception as erro:
        msg = str(erro).lower()
        if "signature" in msg or "key" in msg or "1022" in msg:
            cliente = obter_cliente_binance(forcar_publico=True)
            ticker = cliente.fetch_ticker(par)
        else:
            raise

    resultado = {
        "preco": ticker.get("last", 0.0),
        "variacao_24h": ticker.get("percentage", 0.0),
        "volume_24h": ticker.get("quoteVolume", 0.0),
        "maxima_24h": ticker.get("high", 0.0),
        "minima_24h": ticker.get("low", 0.0),
    }

    print(f"[Binance] Preço atual: ${resultado['preco']:,.2f} USDT")
    return resultado
