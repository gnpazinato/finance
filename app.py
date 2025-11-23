import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- CONFIGURAÇÃO ---
st.set_page_config(layout="wide", page_title="Trend Scanner Pro - Opções")

# --- LISTA DE ATIVOS ---
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO", "VOO", "XLF", 
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "XLE", "XLU", "XLI", "XLB", "XLP", "XLY", "XLV", "XBI", "VNQ", "EEM",
    "AMD", "TSLA", "CRM", "INTC", "JPM", "BAC", "V", "GS", "UNH", "JNJ", 
    "PFE", "HD", "MCD", "NKE", "WMT", "COST", "PG", "CAT", "BA", "XOM"
]

# --- DADOS ---
@st.cache_data(ttl=900) 
def get_data(tickers):
    data = yf.download(tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    return data

# --- ANÁLISE TÉCNICA & ESTRATÉGIA ---
def analyze_ticker(ticker, df):
    if df.empty or len(df) < 200: return None
    
    # Limpeza de MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    if not {'Close', 'High', 'Low'}.issubset(df.columns): return None

    # Indicadores
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    ma20 = ta.sma(close, length=20)
    ma50 = ta.sma(close, length=50)
    ma200 = ta.sma(close, length=200)
    rsi = ta.rsi(close, length=14)
    
    donchian_high = high.rolling(window=20).max()
    donchian_low = low.rolling(window=20).min()
    
    try:
        curr_price = close.iloc[-1]
        curr_ma20 = ma20.iloc[-1]
        curr_ma50 = ma50.iloc[-1]
        curr_ma200 = ma200.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        prev_high_20 = donchian_high.iloc[-2]
        prev_low_20 = donchian_low.iloc[-2]
    except:
        return None
    
    # --- LÓGICA DE DECISÃO ---
    sugestao = "Aguardar"
    motivo = "-"
    vencimento = "-"
    strike_alvo = "-"
    cor_fundo = "white"
    cor_texto = "black"

    # 1. TENDÊNCIA DE ALTA
    if curr_price > curr_ma200 and curr_ma50 > curr_ma200:
        
        # A) Rompimento (Call Seco)
        if curr_price > prev_high_20:
            sugestao = "COMPRA CALL (Seco)"
            motivo = "Rompimento Explosivo"
            vencimento = "Curto (15-30d)"
            strike_alvo = f"${curr_price:.0f} (ATM)" 
            cor_fundo = "#b6d7a8" # Verde Claro
            
        # B) Pullback (Trava de Alta)
        elif (curr_price <= curr_ma20 * 1.02) and (curr_rsi < 60) and (curr_rsi > 40):
            sugestao = "TRAVA DE ALTA (Call Spread)"
            motivo = "Pullback (Correção)"
            vencimento = "Médio (30-45d)"
            strike_long = curr_price
            strike_short = curr_price * 1.04
            strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
            cor_fundo = "#38761d" # Verde Escuro
            cor_texto = "white"

    # 2. TENDÊNCIA DE BAIXA
    elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:
        
        # C) Perda de Fundo (Put Seco)
        if curr_price < prev_low_20:
            sugestao = "COMPRA PUT (Seco)"
            motivo = "Perda de Suporte"
            vencimento = "Curto (15-30d)"
            strike_alvo = f"${curr_price:.0f} (ATM)"
            cor_fundo = "#ea9999" # Vermelho Claro
            
        # D) Pullback de Baixa (Trava de Baixa)
        elif (curr_price >= curr_ma20 * 0.98) and (curr_rsi > 40) and (curr_rsi < 60):
            sugestao = "TRAVA DE BAIXA (Put Spread)"
            motivo = "Repique p/ Cair"
            vencimento = "Médio (30-45d)"
            strike_long = curr_price
            strike_short = curr_price * 0.96
            strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
            cor_fundo = "#990000" # Vermelho Escuro
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

# --- INTERFACE ---
st.title("🎯 Trend Scanner - Opções (Strikes & Prazos)")

if st.button("🔄 Atualizar Scanner"):
    st.cache_data.clear()

with st.spinner('Calculando estratégias para 45 ativos...'):
    raw_data = get_data(TICKERS)

results = []
for ticker in TICKERS:
    try:
        if isinstance(raw_data.columns, pd.MultiIndex) and ticker in raw_data.columns.get_level_values(0):
            df_t = raw_data[ticker].dropna()
        else:
            df_t = raw_data.dropna()
        
        res = analyze_ticker(ticker, df_t)
        if res: results.append(res)
    except:
        continue

df_results = pd.DataFrame(results)

if not df_results.empty and "Estratégia" in df_results.columns:
    # Filtros
    opcoes = df_results["Estratégia"].unique()
    filtro = st.sidebar.multiselect("Filtrar por Operação:", options=opcoes, default=[x for x in opcoes if x != "Aguardar"])
    
    # Cria uma cópia para não afetar o original
    if filtro:
        df_final = df_results[df_results["Estratégia"].isin(filtro)].copy()
    else:
        df_final = df_results.copy()

    # 1. Reseta o índice para ficar 0, 1, 2, 3...
    df_final.reset_index(drop=True, inplace=True)
    # Opcional: Ajustar índice para começar em 1 visualmente
    df_final.index = df_final.index + 1

    # Estilização Segura
    def apply_style(row):
        bg = row["_cor_fundo"]
        txt = row["_cor_texto"]
        # Aplica estilo em TUDO, depois esconderemos as colunas técnicas
        return [f'background-color: {bg}; color: {txt}' for _ in row]

    st.subheader("Oportunidades Identificadas")
    
    # 2. Aplica estilo no DF inteiro e USA .hide() para esconder as colunas de controle
    st.dataframe(
        df_final.style.apply(apply_style, axis=1).hide(axis="columns", subset=["_cor_fundo", "_cor_texto"]),
        use_container_width=True,
        height=600
    )
    
else:
    st.warning("Nenhum dado ou erro na conexão. Tente atualizar novamente.")

# --- ÁREA DE GRÁFICO ---
st.divider()
sel = st.selectbox("Analisar Gráfico:", TICKERS)
if sel:
    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            df_chart = raw_data[sel].dropna()
        else:
            df_chart = raw_data.dropna()

        df_chart['MA20'] = ta.sma(df_chart['Close'], length=20)
        df_chart['MA50'] = ta.sma(df_chart['Close'], length=50)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Preço'))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange'), name='MA20 (Curto)'))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], line=dict(color='blue'), name='MA50 (Tendência)'))
        fig.update_layout(xaxis_rangeslider_visible=False, title=f"{sel} - Gráfico Diário")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Gráfico indisponível.")
