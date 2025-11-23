import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import logging
from datetime import datetime

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

logging.basicConfig(level=logging.WARNING)

st.set_page_config(layout="wide", page_title="Trend Scanner Pro - Opções")

# --- CONSTANTES DA ESTRATÉGIA ---
MA_SHORT = 20
MA_MEDIUM = 50
MA_LONG = 200
DONCHIAN_LEN = 20

RSI_LOW = 40
RSI_HIGH = 60

PULLBACK_TOL = 0.02
SPREAD_CALL_PCT = 0.04
SPREAD_PUT_PCT = 0.04

CACHE_TTL = 900

TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO", "VOO", "XLF",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "XLE", "XLU", "XLI", "XLB", "XLP", "XLY", "XLV", "XBI", "VNQ", "EEM",
    "AMD", "TSLA", "CRM", "INTC", "JPM", "BAC", "V", "GS", "UNH", "JNJ",
    "PFE", "HD", "MCD", "NKE", "WMT", "COST", "PG", "CAT", "BA", "XOM"
]

PERIOD_OPTIONS = {
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y"
}

@st.cache_data(ttl=CACHE_TTL)
def get_data(tickers, period="1y", interval="1d"):
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True
    )
    return data


def get_ticker_df(raw_data, ticker):
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        tickers_level = raw_data.columns.get_level_values(0)
        if ticker in tickers_level:
            return raw_data[ticker].dropna()
        else:
            return pd.DataFrame()

    return raw_data.dropna()


def analyze_ticker(ticker, df):

    try:
        if df is None or df.empty or len(df) < MA_LONG + 5:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        ma20 = ta.sma(close, length=MA_SHORT)
        ma50 = ta.sma(close, length=MA_MEDIUM)
        ma200 = ta.sma(close, length=MA_LONG)
        rsi = ta.rsi(close, length=14)

        donchian_high = high.rolling(window=DONCHIAN_LEN).max()
        donchian_low = low.rolling(window=DONCHIAN_LEN).min()

        curr_price = close.iloc[-1]
        curr_ma20 = ma20.iloc[-1]
        curr_ma50 = ma50.iloc[-1]
        curr_ma200 = ma200.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        prev_high_20 = donchian_high.iloc[-2]
        prev_low_20 = donchian_low.iloc[-2]

        sugestao = "Aguardar"
        motivo = "-"
        vencimento = "-"
        strike_alvo = "-"
        cor_fundo = "white"
        cor_texto = "black"

        if curr_price > curr_ma200 and curr_ma50 > curr_ma200:

            if curr_price > prev_high_20:
                sugestao = "COMPRA CALL (Seco)"
                motivo = "Rompimento Explosivo"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#b6d7a8"

            elif (curr_price <= curr_ma20 * (1 + PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE ALTA (Call Spread)"
                motivo = "Pullback (Correção)"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 + SPREAD_CALL_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#38761d"
                cor_texto = "white"

        elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:

            if curr_price < prev_low_20:
                sugestao = "COMPRA PUT (Seco)"
                motivo = "Perda de Suporte"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#ea9999"

            elif (curr_price >= curr_ma20 * (1 - PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE BAIXA (Put Spread)"
                motivo = "Repique p/ Cair"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 - SPREAD_PUT_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#990000"
                cor_texto = "white"

        return {
            "Ticker": ticker,
            "Preço": f"${curr_price:.2f}",
            "Estratégia": sugestao,
            "Strikes (Ref)": strike_alvo,
            "Vencimento": vencimento,
            "Motivo": motivo,
            "_cor_fundo": cor_fundo,
            "_cor_texto": cor_texto
        }

    except:
        return None


st.title("🎯 Trend Scanner - Opções (Strikes & Prazos)")
st.caption(f"Dados de mercado obtidos via yfinance (cache de até {CACHE_TTL // 60} minutos).")

st.sidebar.header("Configurações")

period_label = st.sidebar.selectbox(
    "Período do histórico:",
    list(PERIOD_OPTIONS.keys()),
    index=1
)
period = PERIOD_OPTIONS[period_label]

if st.button("🔄 Atualizar Scanner"):
    get_data.clear()

with st.spinner(f"Calculando estratégias para {len(TICKERS)} ativos..."):
    raw_data = get_data(TICKERS, period=period, interval="1d")

results = []

if raw_data is not None and not raw_data.empty:
    for ticker in TICKERS:
        df_t = get_ticker_df(raw_data, ticker)
        if df_t.empty:
            continue

        res = analyze_ticker(ticker, df_t)
        if res:
            results.append(res)

df_results = pd.DataFrame(results)

if not df_results.empty:

    st.subheader("Resumo das Sinalizações")

    estrategias_prioritarias = [
        "COMPRA CALL (Seco)",
        "TRAVA DE ALTA (Call Spread)",
        "COMPRA PUT (Seco)",
        "TRAVA DE BAIXA (Put Spread)"
    ]

    cols = st.columns(len(estrategias_prioritarias))

    for col, est in zip(cols, estrategias_prioritarias):
        qtd = (df_results["Estratégia"] == est).sum()
        col.metric(est, qtd)

    opcoes = df_results["Estratégia"].unique()

    filtro = st.sidebar.multiselect(
        "Filtrar por Operação:",
        options=opcoes,
        default=[x for x in opcoes if x != "Aguardar"]
    )

    if filtro:
        df_final = df_results[df_results["Estratégia"].isin(filtro)].copy()
    else:
        df_final = df_results.copy()

    df_final.reset_index(drop=True, inplace=True)
    df_final.index = df_final.index + 1

    df_show = df_final.drop(columns=["_cor_fundo", "_cor_texto"]).copy()

    def apply_style(row):
        bg = row["_cor_fundo"]
        txt = row["_cor_texto"]
        return [f"background-color: {bg}; color: {txt}" for _ in row]

    st.subheader("Oportunidades Identificadas")

    st.dataframe(
        df_show.style.apply(apply_style, axis=1),
        use_container_width=True,
        height=650
    )

else:
    st.warning("Nenhum dado ou erro ao calcular.")

st.divider()
st.subheader("Análise Gráfica")

sel = st.selectbox("Ticker:", TICKERS)

if sel:

    try:
        df_chart = get_ticker_df(raw_data, sel)

        df_chart["MA20"] = ta.sma(df_chart["Close"], length=MA_SHORT)
        df_chart["MA50"] = ta.sma(df_chart["Close"], length=MA_MEDIUM)

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"], high=df_chart["High"],
            low=df_chart["Low"], close=df_chart["Close"],
            name="Preço"
        ))

        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=df_chart["MA20"],
            line=dict(color="orange"), name="MA20 (Curto)"
        ))

        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=df_chart["MA50"],
            line=dict(color="blue"), name="MA50 (Tendência)"
        ))

        fig.update_layout(xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.error("Erro ao exibir gráfico")
