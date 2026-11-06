"""
analise_tecnica.py — Módulo de cálculo de indicadores técnicos.

Calcula RSI, MACD, Médias Móveis e Bandas de Bollinger sobre
um DataFrame de dados históricos de criptomoedas.
Utiliza Pandas para todas as operações.
"""

import pandas as pd


def calcular_media_movel(df: pd.DataFrame, janela: int) -> pd.Series:
    """
    Calcula a Média Móvel Simples (SMA) para uma janela de períodos.

    Parâmetros:
        df     : DataFrame com coluna 'Fechamento'
        janela : Número de períodos da média móvel

    Retorna:
        Series com os valores da média móvel
    """
    return df["Fechamento"].rolling(window=janela).mean()


def calcular_rsi(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """
    Calcula o Índice de Força Relativa (RSI - Relative Strength Index).
    Valores acima de 70 indicam sobrecompra; abaixo de 30, sobrevenda.

    Parâmetros:
        df      : DataFrame com coluna 'Fechamento'
        periodo : Período para o cálculo (padrão: 14)

    Retorna:
        Series com os valores do RSI (0 a 100)
    """
    variacao = df["Fechamento"].diff()
    ganho = variacao.clip(lower=0)
    perda = -variacao.clip(upper=0)

    media_ganho = ganho.rolling(window=periodo).mean()
    media_perda = perda.rolling(window=periodo).mean()

    rs = media_ganho / media_perda
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calcular_macd(
    df: pd.DataFrame,
    periodo_rapido: int = 12,
    periodo_lento: int = 26,
    periodo_sinal: int = 9
) -> pd.DataFrame:
    """
    Calcula o MACD (Moving Average Convergence Divergence).

    Parâmetros:
        df              : DataFrame com coluna 'Fechamento'
        periodo_rapido  : EMA rápida (padrão: 12)
        periodo_lento   : EMA lenta (padrão: 26)
        periodo_sinal   : Linha de sinal (padrão: 9)

    Retorna:
        DataFrame com colunas: 'macd', 'sinal_macd', 'histograma_macd'
    """
    ema_rapida = df["Fechamento"].ewm(span=periodo_rapido, adjust=False).mean()
    ema_lenta = df["Fechamento"].ewm(span=periodo_lento, adjust=False).mean()

    linha_macd = ema_rapida - ema_lenta
    linha_sinal = linha_macd.ewm(span=periodo_sinal, adjust=False).mean()
    histograma = linha_macd - linha_sinal

    return pd.DataFrame({
        "macd": linha_macd,
        "sinal_macd": linha_sinal,
        "histograma_macd": histograma,
    })


def calcular_bandas_bollinger(
    df: pd.DataFrame,
    janela: int = 20,
    desvios: float = 2.0
) -> pd.DataFrame:
    """
    Calcula as Bandas de Bollinger.

    Parâmetros:
        df      : DataFrame com coluna 'Fechamento'
        janela  : Período da média móvel central (padrão: 20)
        desvios : Número de desvios padrão para as bandas (padrão: 2.0)

    Retorna:
        DataFrame com colunas: 'banda_superior', 'banda_media', 'banda_inferior'
    """
    banda_media = df["Fechamento"].rolling(window=janela).mean()
    desvio_padrao = df["Fechamento"].rolling(window=janela).std()

    return pd.DataFrame({
        "banda_superior": banda_media + (desvios * desvio_padrao),
        "banda_media": banda_media,
        "banda_inferior": banda_media - (desvios * desvio_padrao),
    })


MINIMO_CANDLES_INDICADORES = 50


def enriquecer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona todos os indicadores técnicos ao DataFrame de dados históricos.

    Calcula e anexa ao DataFrame:
        - MM20, MM50     : Médias Móveis de 20 e 50 períodos
        - RSI14          : RSI de 14 períodos
        - MACD, Sinal    : Linhas do MACD
        - Bollinger      : Bandas de Bollinger (20 períodos)

    Parâmetros:
        df : DataFrame com colunas OHLCV (Abertura, Maxima, Minima, Fechamento, Volume)

    Retorna:
        DataFrame original com colunas de indicadores adicionadas

    Levanta:
        ValueError : se houver menos de 50 candles (RF03/CT02), pois a MM50
                     não teria uma janela completa e os indicadores não seriam
                     confiáveis para análise.
    """
    if df is None or len(df) < MINIMO_CANDLES_INDICADORES:
        quantidade = 0 if df is None else len(df)
        raise ValueError(
            f"São necessários pelo menos {MINIMO_CANDLES_INDICADORES} candles "
            f"para calcular os indicadores; recebidos {quantidade}."
        )

    df = df.copy()

    # Médias Móveis Simples
    df["MM20"] = calcular_media_movel(df, 20)
    df["MM50"] = calcular_media_movel(df, 50)

    # RSI
    df["RSI14"] = calcular_rsi(df, 14)

    # MACD
    macd_df = calcular_macd(df)
    df["MACD"] = macd_df["macd"]
    df["Sinal_MACD"] = macd_df["sinal_macd"]
    df["Histograma_MACD"] = macd_df["histograma_macd"]

    # Bandas de Bollinger
    bollinger_df = calcular_bandas_bollinger(df, 20)
    df["Bollinger_Superior"] = bollinger_df["banda_superior"]
    df["Bollinger_Media"] = bollinger_df["banda_media"]
    df["Bollinger_Inferior"] = bollinger_df["banda_inferior"]

    return df


def resumir_indicadores(df: pd.DataFrame) -> dict:
    """
    Retorna um resumo dos indicadores técnicos mais recentes em formato de dicionário.
    Útil para incluir no prompt do Gemini.

    Parâmetros:
        df : DataFrame já enriquecido com indicadores

    Retorna:
        Dicionário com os últimos valores de cada indicador
    """
    ultima_linha = df.iloc[-1]

    return {
        "preco_fechamento": round(ultima_linha["Fechamento"], 2),
        "mm20": round(ultima_linha["MM20"], 2) if pd.notna(ultima_linha["MM20"]) else None,
        "mm50": round(ultima_linha["MM50"], 2) if pd.notna(ultima_linha["MM50"]) else None,
        "rsi14": round(ultima_linha["RSI14"], 2) if pd.notna(ultima_linha["RSI14"]) else None,
        "macd": round(ultima_linha["MACD"], 4) if pd.notna(ultima_linha["MACD"]) else None,
        "sinal_macd": round(ultima_linha["Sinal_MACD"], 4) if pd.notna(ultima_linha["Sinal_MACD"]) else None,
        "bollinger_superior": round(ultima_linha["Bollinger_Superior"], 2) if pd.notna(ultima_linha["Bollinger_Superior"]) else None,
        "bollinger_inferior": round(ultima_linha["Bollinger_Inferior"], 2) if pd.notna(ultima_linha["Bollinger_Inferior"]) else None,
    }
