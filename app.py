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
    # Ajuste para garantir formato correto mesmo se baixar apenas 1 ticker
    data = yf.download(tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    return data

# --- FUNÇÃO DE ANÁLISE TÉCNICA (LÓGICA DE OPÇÕES) ---
def analyze_ticker(ticker, df):
    # Garante que as colunas existem
    if df.empty or len(df) < 200:
        return None
    
    # Normaliza nomes das colunas (remove MultiIndex se existir)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Verifica se as colunas essenciais estão presentes
    required_cols = ['Close', 'High', 'Low']
    if not all(col in df.columns for col in required_cols):
        return None

    # Indicadores
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    ma20 = ta.sma(close, length=20)
    ma50 = ta.sma(close, length=50)
    ma200 = ta.sma(close, length=200)
    rsi = ta.rsi(close, length=14)
    
    # Donchian (Canais de Preço) para Rompimentos
    donchian_high = high.rolling(window=20).max()
    donchian_low = low.rolling(window=20).min()
    
    # Valores Atuais (trata erros de índice)
    try:
        curr_price = close.iloc[-1]
        curr_ma20 = ma20.iloc[-1]
        curr_ma50 = ma50.iloc[-1]
        curr_ma200 = ma200.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        prev_high_20 = donchian_high.iloc[-2]
        prev_low_20 = donchian_low.iloc[-2]
    except IndexError:
        return None
    
    # --- CÉREBRO DA ESTRATÉGIA ---
    tendencia = "Lateral/Indefinida"
    sugestao_opcao = "Aguardar"
    vencimento_ideal = "-"
    motivo = "-"
    cor_fundo = "white"
    cor_texto = "black"

    # 1. TENDÊNCIA DE ALTA (Preço acima da MA200)
    if curr_price > curr_ma200 and curr_ma50 > curr_ma200:
        tendencia = "Alta (Bullish)"
        
        # A) ROMPIMENTO DE ALTA (MOMENTUM)
        if curr_price > prev_high_20:
            sugestao_opcao = "COMPRA SECO DE CALL"
            motivo = "Rompimento Explosivo"
            vencimento_ideal = "15-30 Dias"
            cor_fundo = "#b6d7a8" # Verde Claro
            
        # B) PULLBACK DE ALTA (OPORTUNIDADE)
        elif (curr_price <= curr_ma20 * 1.02) and (curr_rsi < 60) and (curr_rsi > 40):
            sugestao_opcao = "TRAVA DE ALTA (Call Spread)"
            motivo = "Correção na Tendência"
            vencimento_ideal = "30-45 Dias"
            cor_fundo = "#6aa84f" # Verde Escuro
            cor_texto = "white"

    # 2. TENDÊNCIA DE BAIXA (Preço abaixo da MA200)
    elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:
        tendencia = "Baixa (Bearish)"
        
        # C) ROMPIMENTO DE BAIXA (CRASH)
        if curr_price < prev_low_20:
            sugestao_opcao = "COMPRA SECO DE PUT"
            motivo = "Perda de Suporte"
            vencimento_ideal = "15-30 Dias"
            cor_fundo = "#ea9999" # Vermelho Claro
            
        # D) PULLBACK DE BAIXA (RESPIRO)
        elif (curr_price >= curr_ma20 * 0.98) and (curr_rsi > 40) and (curr_rsi < 60):
            sugestao_opcao = "TRAVA DE BAIXA (Put Spread)"
            motivo = "Repique na Tendência"
            vencimento_ideal = "30-45 Dias"
            cor_fundo = "#990000" # Vermelho Escuro
            cor_texto = "white"

    return {
        "Ticker": ticker,
        "Preço": round(curr_price, 2),
        "Tendência Macro": tendencia,
        "Estratégia Opções": sugestao_opcao,
        "Setup (Motivo)": motivo,
        "Vencimento Sugerido": vencimento_ideal,
        "_cor_fundo": cor_fundo,
        "_cor_texto": cor_texto
    }

# --- INTERFACE PRINCIPAL ---
st.title("🎯 Opções Trend Scanner")
st.markdown("Focado em identificar setups para **Compra a Seco** (Explosão) ou **Travas** (Correção).")

if st.button("🔄 Atualizar Mercado"):
    st.cache_data.clear()

with st.spinner('Analisando 45 ativos...'):
    raw_data = get_data(TICKERS)

results = []
debug_errors = []

for ticker in TICKERS:
    try:
        # Acesso seguro aos dados do ticker
        if ticker in raw_data.columns.get_level_values(0):
            df_ticker = raw_data[ticker].dropna()
        else:
            # Tenta acessar diretamente caso a estrutura varie
            df_ticker = raw_data
        
        res = analyze_ticker(ticker, df_ticker)
        if res:
            results.append(res)
            
    except Exception as e:
        debug_errors.append(f"{ticker}: {str(e)}")
        continue

df_results = pd.DataFrame(results)

# --- VERIFICAÇÃO SE HÁ DADOS ---
if df_results.empty:
    st.warning("⚠️ Nenhum dado encontrado ou todos os ativos deram erro na análise.")
    if debug_errors:
        with st.expander("Ver erros técnicos"):
            for err in debug_errors:
                st.write(err)
else:
    # --- FILTROS ---
    st.sidebar.header("Filtros de Estratégia")
    
    # Verifica se a coluna existe antes de acessar
    if "Estratégia Opções" in df_results.columns:
        tipos_estrategia = df_results["Estratégia Opções"].unique()
        filtro = st.sidebar.multiselect("Mostrar apenas:", tipos_estrategia, default=[x for x in tipos_estrategia if x != "Aguardar"])

        if not filtro:
            df_final = df_results
        else:
            df_final = df_results[df_results["Estratégia Opções"].isin(filtro)]

        # --- TABELA COLORIDA ---
        def style_dataframe(row):
            return [f'background-color: {row["_cor_fundo"]}; color: {row["_cor_texto"]}' for _ in row]

        st.subheader("Mesa de Operações")
        # Remove colunas de cor antes de mostrar
        display_cols = [c for c in df_final.columns if not c.startswith("_")]

        st.dataframe(
            df_final[display_cols].style.apply(style_dataframe, axis=1),
            use_container_width=True,
            height=600
        )
    else:
        st.error("Erro na estrutura dos dados: Coluna de estratégia não gerada.")

    # --- GRÁFICO ---
    st.divider()
    st.subheader("🔍 Validação Visual")
    selected = st.selectbox("Selecione para ver o gráfico:", TICKERS)

    if selected:
        try:
            df_chart = raw_data[selected].dropna()
            
            # Normaliza colunas para o gráfico
            if isinstance(df_chart.columns, pd.MultiIndex):
                 df_chart.columns = df_chart.columns.get_level_values(-1)

            df_chart['MA20'] = ta.sma(df_chart['Close'], length=20)
            df_chart['MA50'] = ta.sma(df_chart['Close'], length=50)
            df_chart['MA200'] = ta.sma(df_chart['Close'], length=200)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Preço'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1), name='MA20'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], line=dict(color='blue', width=2), name='MA50'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA200'], line=dict(color='black', width=2, dash='dot'), name='MA200 (Tendência Macro)'))
            
            fig.update_layout(title=f"{selected} - Gráfico Diário", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao carregar gráfico para {selected}: {e}")
