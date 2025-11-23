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

PULLBACK_TOL = 0.02  # 2% em relação à MA20
SPREAD_CALL_PCT = 0.04  # 4% acima no call spread
SPREAD_PUT_PCT = 0.04   # 4% abaixo no put spread

CACHE_TTL = 900  # segundos (15 minutos)

# --- UNIVERSO DE ATIVOS ---
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO", "VOO", "XLF",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "XLE", "XLU", "XLI", "XLB", "XLP", "XLY", "XLV", "XBI", "VNQ", "EEM",
    "AMD", "TSLA", "CRM", "INTC", "JPM", "BAC", "V", "GS", "UNH", "JNJ",
    "PFE", "HD", "MCD", "NKE", "WMT", "COST", "PG", "CAT", "BA", "XOM"
]

# Mapeamento de rótulos de período para o yfinance
PERIOD_OPTIONS = {
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y"
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_data(tickers, period="1y", interval="1d"):
    """
    Baixa dados de mercado do yfinance com cache.
    """
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
    """
    Extrai o DataFrame de um único ticker a partir do retorno do yfinance.
    Lida com MultiIndex automaticamente.
    """
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        # Estrutura típica quando se baixa vários tickers
        tickers_level = raw_data.columns.get_level_values(0)
        if ticker in tickers_level:
            df_t = raw_data[ticker].dropna()
            return df_t
        else:
            return pd.DataFrame()
    else:
        # Caso improvável aqui, mas deixado por segurança
        return raw_data.dropna()


def analyze_ticker(ticker, df):
    """
    Aplica a lógica de análise técnica e define a estratégia de opções.
    Retorna um dicionário com os campos da tabela ou None se não houver sinal.
    """
    try:
        # Garante quantidade mínima de candles para MA longa
        if df is None or df.empty or len(df) < MA_LONG + 5:
            return None

        # Proteção caso venha MultiIndex de alguma forma
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        if not {"Close", "High", "Low"}.issubset(df.columns):
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        ma20 = ta.sma(close, length=MA_SHORT)
        ma50 = ta.sma(close, length=MA_MEDIUM)
        ma200 = ta.sma(close, length=MA_LONG)
        rsi = ta.rsi(close, length=14)

        donchian_high = high.rolling(window=DONCHIAN_LEN).max()
        donchian_low = low.rolling(window=DONCHIAN_LEN).min()

        # Pega os valores atuais (última barra) com proteção
        curr_price = close.iloc[-1]
        curr_ma20 = ma20.iloc[-1]
        curr_ma50 = ma50.iloc[-1]
        curr_ma200 = ma200.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        prev_high_20 = donchian_high.iloc[-2]
        prev_low_20 = donchian_low.iloc[-2]

        # Defaults
        sugestao = "Aguardar"
        motivo = "-"
        vencimento = "-"
        strike_alvo = "-"
        cor_fundo = "white"
        cor_texto = "black"

        # ====================================================
        # 1. TENDÊNCIA DE ALTA
        # ====================================================
        if curr_price > curr_ma200 and curr_ma50 > curr_ma200:

            # A) Rompimento (Call Seco)
            if curr_price > prev_high_20:
                sugestao = "COMPRA CALL (Seco)"
                motivo = "Rompimento Explosivo"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#b6d7a8"  # Verde Claro

            # B) Pullback (Trava de Alta)
            elif (curr_price <= curr_ma20 * (1 + PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE ALTA (Call Spread)"
                motivo = "Pullback (Correção)"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 + SPREAD_CALL_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#38761d"  # Verde Escuro
                cor_texto = "white"

        # ====================================================
        # 2. TENDÊNCIA DE BAIXA
        # ====================================================
        elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:

            # C) Perda de Fundo (Put Seco)
            if curr_price < prev_low_20:
                sugestao = "COMPRA PUT (Seco)"
                motivo = "Perda de Suporte"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#ea9999"  # Vermelho Claro

            # D) Pullback de Baixa (Trava de Baixa)
            elif (curr_price >= curr_ma20 * (1 - PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE BAIXA (Put Spread)"
                motivo = "Repique p/ Cair"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 - SPREAD_PUT_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#990000"  # Vermelho Escuro
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

    except Exception as e:
        logging.exception(f"Erro ao analisar {ticker}: {e}")
        return None


# ============================================================
# INTERFACE
# ============================================================

st.title("🎯 Trend Scanner - Opções (Strikes & Prazos)")
st.caption(f"Dados de mercado obtidos via yfinance (cache de até {CACHE_TTL // 60} minutos).")

# Seleção de período na sidebar
st.sidebar.header("Configurações")
period_label = st.sidebar.selectbox(
    "Período do histórico:",
    list(PERIOD_OPTIONS.keys()),
    index=1  # default: 1 ano
)
period = PERIOD_OPTIONS[period_label]

# Botão para limpar cache apenas deste get_data
if st.button("🔄 Atualizar Scanner"):
    get_data.clear()

with st.spinner(f"Calculando estratégias para {len(TICKERS)} ativos..."):
    raw_data = get_data(TICKERS, period=period, interval="1d")

# ============================================================
# CÁLCULO DAS ESTRATÉGIAS
# ============================================================

results = []

if raw_data is not None and not raw_data.empty:
    for ticker in TICKERS:
        df_t = get_ticker_df(raw_data, ticker)
        if df_t.empty:
            continue

        res = analyze_ticker(ticker, df_t)
        if res is not None:
            results.append(res)

df_results = pd.DataFrame(results)

# ============================================================
# RESUMO + TABELA DE OPORTUNIDADES
# ============================================================

if not df_results.empty and "Estratégia" in df_results.columns:
    # Resumo das sinalizações
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

    # Filtros na sidebar
    opcoes = df_results["Estratégia"].unique()
    default_filtro = [x for x in opcoes if x != "Aguardar"]

    filtro = st.sidebar.multiselect(
        "Filtrar por Operação:",
        options=opcoes,
        default=default_filtro
    )

    # Aplica filtro
    if filtro:
        df_final = df_results[df_results["Estratégia"].isin(filtro)].copy()
    else:
        df_final = df_results.copy()

    # Ajusta índice para começar em 1
    df_final.reset_index(drop=True, inplace=True)
    df_final.index = df_final.index + 1

    # Função de estilização por linha
    def apply_style(row):
        bg = row["_cor_fundo"]
        txt = row["_cor_texto"]
        return [f"background-color: {bg}; color: {txt}" for _ in row]

    st.subheader("Oportunidades Identificadas")

    # DataFrame estilizado, ocultando colunas de cor
    styled = (
        df_final
        .style
        .apply(apply_style, axis=1)
        .hide(axis="columns", subset=["_cor_fundo", "_cor_texto"])
    )

    st.dataframe(
        styled,
        width="stretch",
        height=600
    )

    # Botão para exportar sinais em CSV (mantém as colunas internas no arquivo)
    st.download_button(
        "📥 Baixar sinais em CSV",
        df_final.to_csv(index=True).encode("utf-8"),
        file_name=f"trend_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    # Explicação opcional
    with st.expander("Como interpretar as estratégias?"):
        st.markdown(
            """
- **COMPRA CALL (Seco)**: operação direcional apostando em alta forte de curto prazo.
- **TRAVA DE ALTA (Call Spread)**: operação direcional de alta com risco e ganho máximos limitados.
- **COMPRA PUT (Seco)**: operação direcional apostando em queda forte de curto prazo.
- **TRAVA DE BAIXA (Put Spread)**: operação direcional de baixa com risco e ganho máximos limitados.
- **Aguardar**: nenhum setup claro de acordo com os critérios definidos.

> Este painel é apenas um scanner técnico e **não constitui recomendação de investimento.**
            """
        )

else:
    st.warning("Nenhum dado ou erro na conexão. Tente atualizar novamente.")

# ============================================================
# ÁREA DE GRÁFICO
# ============================================================

st.divider()
st.subheader("Análise Gráfica")

sel = st.selectbox("Analisar Gráfico:", TICKERS)

if sel:
    try:
        df_chart = get_ticker_df(raw_data, sel)
        if df_chart is None or df_chart.empty:
            raise ValueError("Sem dados suficientes para este ticker.")

        df_chart = df_chart.copy()
        df_chart["MA20"] = ta.sma(df_chart["Close"], length=MA_SHORT)
        df_chart["MA50"] = ta.sma(df_chart["Close"], length=MA_MEDIUM)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name="Preço"
        ))
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=df_chart["MA20"],
            line=dict(color="orange"),
            name=f"MA{MA_SHORT} (Curto)"
        ))
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=df_chart["MA50"],
            line=dict(color="blue"),
            name=f"MA{MA_MEDIUM} (Tendência)"
        ))

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            title=f"{sel} - Gráfico Diário"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Mostra detalhes da estratégia do ticker selecionado, se existir
        if not df_results.empty and "Ticker" in df_results.columns:
            sel_info = df_results[df_results["Ticker"] == sel].head(1)
            if not sel_info.empty:
                st.markdown("**Detalhes da Estratégia para o ticker selecionado:**")
                st.table(sel_info[["Estratégia", "Strikes (Ref)", "Vencimento", "Motivo"]])

    except Exception as e:
        st.error(f"Gráfico indisponível para {sel}: {e}")
