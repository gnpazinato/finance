import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Trend Scanner Pro - Opções")

# --- LISTA DE ATIVOS ---
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO", "VOO", "XLF", 
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "XLE", "XLU", "XLI", "XLB", "XLP", "XLY", "XLV", "XBI", "VNQ", "EEM",
    "AMD", "TSLA", "CRM", "INTC", "JPM", "BAC", "V", "GS", "UNH", "JNJ", 
    "PFE", "HD", "MCD", "NKE", "WMT", "COST", "PG", "CAT", "BA", "XOM"
]

# --- FUNÇÃO PARA PEGAR DADOS ---
@st.cache_data(ttl=900) 
def get_data(tickers):
    # threads=True acelera o download
    data = yf.download(tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    return data

# --- FUNÇÃO DE ANÁLISE TÉCNICA ---
def analyze_ticker(ticker, df):
    # Validações iniciais
    if df.empty or len(df) < 200:
        return None
    
    # Remove MultiIndex nas colunas se houver
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
        
    # Garante colunas necessárias
    req_cols = ['Close', 'High', 'Low']
    if not all(c in df.columns for c in req_cols):
        return None

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
    
    # Tenta pegar o último valor (trata erro se dataframe estiver quebrado no final)
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
    
    # --- CÉREBRO DA ESTRATÉGIA ---
    tendencia = "Indefinida"
    sugestao_opcao = "Aguardar"
    vencimento_ideal = "-"
    motivo = "-"
    cor_fundo = "white"
    cor_texto = "black"

    # 1. TENDÊNCIA DE ALTA
    if curr_price > curr_ma200 and curr_ma50 > curr_ma200:
        tendencia = "Alta"
        
        if curr_price > prev_high_20:
            sugestao_opcao = "COMPRA SECO DE CALL"
            motivo = "Rompimento Explosivo"
            vencimento_ideal = "Curto (15-30d)"
            cor_fundo = "#b6d7a8" # Verde Claro
            
        elif (curr_price <= curr_ma20 * 1.02) and (curr_rsi < 60) and (curr_rsi > 40):
            sugestao_opcao = "TRAVA DE ALTA (Call Spread)"
            motivo = "Pullback (Correção)"
            vencimento_ideal = "Médio (30-45d)"
            cor_fundo = "#38761d" # Verde Escuro
            cor_texto = "white"

    # 2. TENDÊNCIA DE BAIXA
    elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:
        tendencia = "Baixa"
        
        if curr_price < prev_low_20:
            sugestao_opcao = "COMPRA SECO DE PUT"
            motivo = "Perda de Fundo"
            vencimento_ideal = "Curto (15-30d)"
            cor_fundo = "#ea9999" # Vermelho Claro
            
        elif (curr_price >= curr_ma20 * 0.98) and (curr_rsi > 40) and (curr_rsi < 60):
            sugestao_opcao = "TRAVA DE BAIXA (Put Spread)"
            motivo = "Pullback (Repique)"
            vencimento_ideal = "Médio (30-45d)"
            cor_fundo = "#990000" # Vermelho Escuro
            cor_texto = "white"

    return {
        "Ticker": ticker,
        "Preço": f"{curr_price:.2f}",
        "Tendência": tendencia,
        "Estratégia": sugestao_opcao,
        "Motivo": motivo,
        "Vencimento": vencimento_ideal,
        "_cor_fundo": cor_fundo,
        "_cor_texto": cor_texto
    }

# --- INTERFACE ---
st.title("🎯 Trend Scanner - Opções")

if st.button("🔄 Atualizar"):
    st.cache_data.clear()

with st.spinner('Analisando ativos...'):
    raw_data = get_data(TICKERS)

results = []
for ticker in TICKERS:
    try:
        # Tenta acessar de forma segura considerando a estrutura do yfinance
        if isinstance(raw_data.columns, pd.MultiIndex) and ticker in raw_data.columns.get_level_values(0):
            df_t = raw_data[ticker].dropna()
        else:
            df_t = raw_data
            
        res = analyze_ticker(ticker, df_t)
        if res: results.append(res)
    except:
        continue

df_results = pd.DataFrame(results)

if df_results.empty:
    st.warning("Nenhum dado encontrado. O Yahoo Finance pode estar instável.")
else:
    # Filtros
    if "Estratégia" in df_results.columns:
        opcoes = df_results["Estratégia"].unique()
        filtro = st.sidebar.multiselect("Filtrar:", options=opcoes, default=[x for x in opcoes if x != "Aguardar"])
        
        if filtro:
            df_final = df_results[df_results["Estratégia"].isin(filtro)]
        else:
            df_final = df_results

        # Função de Estilo
        def apply_style(row):
            # Pinta a linha inteira com base nas colunas ocultas
            return [f'background-color: {row["_cor_fundo"]}; color: {row["_cor_texto"]}' for _ in row]

        st.subheader("Scanner")
        
        # Aplica o estilo E DEPOIS esconde as colunas técnicas
        st.dataframe(
            df_final.style.apply(apply_style, axis=1).hide(axis="columns", subset=["_cor_fundo", "_cor_texto"]),
            use_container_width=True,
            height=600
        )
    else:
        st.error("Erro ao processar colunas.")

# --- GRÁFICO ---
st.divider()
sel = st.selectbox("Ver Gráfico:", TICKERS)
if sel:
    try:
        if isinstance(raw_data.columns, pd.MultiIndex) and sel in raw_data.columns.get_level_values(0):
            df_chart = raw_data[sel].dropna()
        else:
            df_chart = raw_data.dropna() # Fallback
            
        df_chart['MA20'] = ta.sma(df_chart['Close'], length=20)
        df_chart['MA50'] = ta.sma(df_chart['Close'], length=50)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Preço'))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange'), name='MA20'))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], line=dict(color='blue'), name='MA50'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Dados insuficientes para o gráfico.")
