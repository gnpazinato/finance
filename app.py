import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Trend Scanner Pro")

# --- LISTA DE ATIVOS (SEUS 45 TICKERS) ---
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO", "VOO", "XLF", 
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "XLE", "XLU", "XLI", "XLB", "XLP", "XLY", "XLV", "XBI", "VNQ", "EEM",
    "AMD", "TSLA", "CRM", "INTC", "JPM", "BAC", "V", "GS", "UNH", "JNJ", 
    "PFE", "HD", "MCD", "NKE", "WMT", "COST", "PG", "CAT", "BA", "XOM"
]

# --- FUNÇÃO PARA PEGAR DADOS (COM CACHE PARA NÃO FICAR LENTO) ---
@st.cache_data(ttl=900) # Atualiza a cada 15 minutos
def get_data(tickers):
    data = yf.download(tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True)
    return data

# --- FUNÇÃO DE ANÁLISE TÉCNICA ---
def analyze_ticker(ticker, df):
    # Verifica se há dados suficientes
    if df.empty or len(df) < 200:
        return None
    
    # Cálculos de Indicadores
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Médias Móveis
    ma20 = ta.sma(close, length=20)
    ma50 = ta.sma(close, length=50)
    ma200 = ta.sma(close, length=200)
    
    # RSI (Índice de Força Relativa) - Bom para Pullbacks
    rsi = ta.rsi(close, length=14)
    
    # Donchian Channels (Para Rompimentos) - Máxima dos últimos 20 dias
    donchian_high = high.rolling(window=20).max()
    donchian_low = low.rolling(window=20).min()
    
    # Pega o último valor válido
    curr_price = close.iloc[-1]
    curr_ma20 = ma20.iloc[-1]
    curr_ma50 = ma50.iloc[-1]
    curr_ma200 = ma200.iloc[-1]
    curr_rsi = rsi.iloc[-1]
    prev_high_20 = donchian_high.iloc[-2] # Máxima de ontem (para ver se rompeu hoje)
    
    # --- LÓGICA DE CLASSIFICAÇÃO (CÉREBRO DO SCRIPT) ---
    setup = "Neutro"
    prazo = "Indefinido"
    cor = "white"

    # 1. TENDÊNCIA DE ALTA (Filtro Base: Preço > MA50 > MA200)
    if curr_ma50 > curr_ma200 and curr_price > curr_ma200:
        
        # Cenário A: ROMPIMENTO (Preço rompeu a máxima de 20 dias e está forte)
        if curr_price > prev_high_20:
            setup = "🚀 Rompimento de Alta"
            prazo = "Curto Prazo (Explosão)"
            cor = "#90ee90" # Light Green
            
        # Cenário B: PULLBACK (Preço está acima da MA50, mas recuou perto da MA20 ou RSI < 50)
        elif (curr_price < curr_ma20 * 1.02) and (curr_rsi < 55) and (curr_rsi > 40):
            setup = "🛒 Pullback de Alta"
            prazo = "Médio Prazo (Entrada Segura)"
            cor = "#006400" # Dark Green (Texto Branco idealmente)

    # 2. TENDÊNCIA DE BAIXA (Filtro Base: Preço < MA50 < MA200)
    elif curr_ma50 < curr_ma200 and curr_price < curr_ma200:
        
        # Cenário C: ROMPIMENTO BAIXA (Perdeu fundo)
        if curr_price < donchian_low.iloc[-2]:
            setup = "🔻 Rompimento de Baixa"
            prazo = "Curto Prazo (Queda Rápida)"
            cor = "#ffcccb" # Light Red
            
        # Cenário D: PULLBACK DE BAIXA (Repique até a média para cair mais)
        elif (curr_price > curr_ma20 * 0.98) and (curr_rsi > 45):
            setup = "🐻 Pullback de Baixa"
            prazo = "Médio Prazo (Venda/Put)"
            cor = "#8b0000" # Dark Red

    return {
        "Ticker": ticker,
        "Preço": round(curr_price, 2),
        "RSI": round(curr_rsi, 0),
        "Tendência (MA50/200)": "Alta" if curr_ma50 > curr_ma200 else "Baixa",
        "Setup Identificado": setup,
        "Horizonte Sugerido": prazo
    }

# --- INTERFACE PRINCIPAL ---
st.title("📈 Trend Scanner Pro - Opções & Ações")
st.markdown("Monitor de Rompimentos e Pullbacks em tempo real (delay de 15min).")

if st.button("🔄 Atualizar Dados do Mercado"):
    st.cache_data.clear()

# Loading
with st.spinner('Baixando dados do Yahoo Finance...'):
    raw_data = get_data(TICKERS)

# Processamento
results = []
for ticker in TICKERS:
    try:
        # yfinance retorna MultiIndex, precisamos isolar o ticker
        df_ticker = raw_data[ticker].dropna()
        res = analyze_ticker(ticker, df_ticker)
        if res:
            results.append(res)
    except Exception as e:
        continue

# Cria DataFrame
df_results = pd.DataFrame(results)

# --- FILTROS LATERAIS ---
st.sidebar.header("Filtros")
filtro_setup = st.sidebar.multiselect(
    "Filtrar por Tipo de Setup:",
    options=df_results["Setup Identificado"].unique(),
    default=df_results["Setup Identificado"].unique()
)

# Filtrar Tabela
df_final = df_results[df_results["Setup Identificado"].isin(filtro_setup)]

# Exibir Tabela (Scanner)
st.subheader("🔭 Scanner de Oportunidades")

def color_setup(val):
    color = 'white'
    if 'Rompimento de Alta' in val: color = '#cfdac8' # Verde claro
    elif 'Pullback de Alta' in val: color = '#90ee90' # Verde forte
    elif 'Rompimento de Baixa' in val: color = '#f4cccc' # Vermelho claro
    elif 'Pullback de Baixa' in val: color = '#ea9999' # Vermelho forte
    return f'background-color: {color}; color: black'

st.dataframe(
    df_final.style.applymap(color_setup, subset=['Setup Identificado']),
    use_container_width=True,
    height=500
)

# --- ANÁLISE INDIVIDUAL (GRÁFICO) ---
st.divider()
st.subheader("🔍 Análise Individual")

selected_ticker = st.selectbox("Selecione um ativo para ver o gráfico:", TICKERS)

if selected_ticker:
    df_chart = raw_data[selected_ticker].dropna()
    
    # Médias para o gráfico
    df_chart['MA20'] = ta.sma(df_chart['Close'], length=20)
    df_chart['MA50'] = ta.sma(df_chart['Close'], length=50)
    
    # Criação do Gráfico Plotly
    fig = go.Figure()
    
    # Candles
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'], high=df_chart['High'],
        low=df_chart['Low'], close=df_chart['Close'],
        name='Preço'
    ))
    
    # Médias
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1), name='MA20 (Curto)'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], line=dict(color='blue', width=2), name='MA50 (Médio)'))
    
    fig.update_layout(title=f"Gráfico Diário - {selected_ticker}", xaxis_rangeslider_visible=False, height=600)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostra dados recentes
    last_close = df_chart['Close'].iloc[-1]
    last_ma20 = df_chart['MA20'].iloc[-1]
    dist_ma20 = ((last_close / last_ma20) - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Preço Atual", f"${last_close:.2f}")
    col2.metric("Distância da MA20", f"{dist_ma20:.2f}%", help="Se estiver muito longe (>5%), cuidado com rompimentos, pode estar esticado.")
    col3.metric("RSI (14)", f"{ta.rsi(df_chart['Close']).iloc[-1]:.0f}", help="RSI < 30 (Sobrevendido), RSI > 70 (Sobrecomprado)")
