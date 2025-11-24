import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import logging
from datetime import datetime, date

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
    "1 ano": "1y",
    "2 anos": "2y"
}

# ============================================================
# MÓDULO DE RISCO MACRO (NEWS)
# ============================================================

# Dicionário com a interpretação do que fazer em cada evento
EVENT_GUIDE = {
    "Payroll": "O Payroll mede a criação de empregos nos EUA. \n- **Expectativa:** Dados muito fortes podem fazer o Fed manter juros altos (ruim para Bolsa/Bonds). Dados fracos podem sinalizar recessão.\n- **Ação:** Alta volatilidade garantida às 08:30 AM (ET). Evite abrir novas travas direcionais 24h antes.",
    "CPI": "Índice de Inflação ao Consumidor. \n- **Expectativa:** Inflação alta = Juros altos = Bolsa cai. Inflação baixa = Bolsa sobe.\n- **Ação:** Movimentos violentos. Se estiver comprado em Call, proteja com Stop Loss.",
    "Decisão de Juros (FOMC)": "O evento mais importante do mundo. \n- **Expectativa:** O mercado foca na fala do Powell e no gráfico de pontos (dot plot).\n- **Ação:** NÃO opere durante o anúncio (14:00 ET). Espere a tendência se definir após as 15:30.",
    "PCE": "A medida de inflação preferida do Fed. \n- **Expectativa:** Confirma ou diverge do CPI. Impacto similar, mas às vezes menor.\n- **Ação:** Monitorar yields dos títulos de 10 anos (TNX).",
}

# Calendário Estimado (Baseado em padrões do FED/BLS para o período solicitado)
MACRO_EVENTS = [
    # DEZEMBRO 2025
    {"name": "Payroll", "date": "2025-12-05", "impact": -2},
    {"name": "CPI", "date": "2025-12-10", "impact": -2},
    {"name": "Decisão de Juros (FOMC)", "date": "2025-12-17", "impact": -3}, 
    
    # JANEIRO 2026
    {"name": "Payroll", "date": "2026-01-09", "impact": -2},
    {"name": "CPI", "date": "2026-01-13", "impact": -2},
    {"name": "Decisão de Juros (FOMC)", "date": "2026-01-28", "impact": -3}, 
    
    # FEVEREIRO 2026
    {"name": "Payroll", "date": "2026-02-06", "impact": -2},
    {"name": "CPI", "date": "2026-02-12", "impact": -2},
    {"name": "PCE", "date": "2026-02-27", "impact": -2},
]

NEWS_WINDOW_DAYS = 3  # Dias de alerta antes do evento

def get_macro_alerts(current_date: date):
    """Retorna alertas ativos e suas explicações."""
    if isinstance(current_date, pd.Timestamp):
        current_date = current_date.date()
    
    # Se não tiver data (ex: fim de semana), assume hoje
    if not current_date:
        current_date = datetime.now().date()

    alerts = []
    for ev in MACRO_EVENTS:
        try:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
            days_until = (ev_date - current_date).days
            
            if 0 <= days_until <= NEWS_WINDOW_DAYS:
                # Busca a explicação no dicionário
                explanation = EVENT_GUIDE.get(ev["name"], "Alta volatilidade esperada.")
                alerts.append({
                    "event": f"{ev['name']} ({ev['date']})",
                    "days": days_until,
                    "guide": explanation
                })
        except:
            continue
    return alerts

# ============================================================
# FUNÇÕES TÉCNICAS (ANALISE)
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_data(tickers, period="1y", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True, threads=True)
    return data

def get_ticker_df(raw_data, ticker):
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            return raw_data.xs(ticker, level=0, axis=1).dropna()
        except KeyError:
            return pd.DataFrame()
    else:
        return raw_data.dropna()

def anti_po_filter(direction, df, ma20, ma50, ma200, rsi_series, atr_series):
    try:
        price = df["Close"].iloc[-1]
        curr_rsi = rsi_series.iloc[-1]
        curr_atr = atr_series.iloc[-1]
        curr_atr_pct = curr_atr / price if price > 0 else 0.0
        
        reasons = []
        ok = True
        
        if curr_atr_pct > 0.06:
            ok = False
            reasons.append("Volatilidade extrema")

        if direction == "bull" and curr_rsi > 75:
            ok = False
            reasons.append("RSI Sobrecomprado (>75)")
        if direction == "bear" and curr_rsi < 25:
            ok = False
            reasons.append("RSI Sobrevendido (<25)")

        if not reasons: return True, "-"
        return ok, "; ".join(reasons)
    except:
        return True, "Erro Filtro"

def analyze_ticker(ticker, df):
    try:
        if df is None or df.empty or len(df) < MA_LONG + 5: return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        ma20 = ta.sma(close, length=MA_SHORT)
        ma50 = ta.sma(close, length=MA_MEDIUM)
        ma200 = ta.sma(close, length=MA_LONG)
        rsi = ta.rsi(close, length=14)
        atr = ta.atr(high, low, close, length=14)

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
        cor_fundo = "#ffffff" 
        cor_texto = "#000000"
        direction = "none"
        score = 0 # Para o termômetro

        # 1. ALTA
        if curr_price > curr_ma200 and curr_ma50 > curr_ma200:
            if curr_price > prev_high_20:
                sugestao = "COMPRA CALL (Seco)"
                motivo = "Rompimento Explosivo"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#b6d7a8"
                direction = "bull"
                score = 2 # Alta forte
            elif (curr_price <= curr_ma20 * (1 + PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE ALTA (Call Spread)"
                motivo = "Pullback (Correção)"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 + SPREAD_CALL_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#38761d"
                cor_texto = "#ffffff"
                direction = "bull"
                score = 1 # Alta moderada

        # 2. BAIXA
        elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:
            if curr_price < prev_low_20:
                sugestao = "COMPRA PUT (Seco)"
                motivo = "Perda de Suporte"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#ea9999"
                direction = "bear"
                score = -2 # Baixa forte
            elif (curr_price >= curr_ma20 * (1 - PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE BAIXA (Put Spread)"
                motivo = "Repique p/ Cair"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 - SPREAD_PUT_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#990000"
                cor_texto = "#ffffff"
                direction = "bear"
                score = -1 # Baixa moderada

        if direction == "none":
            filtro_ok = True
            motivo_filtro = "-"
        else:
            filtro_ok, motivo_filtro = anti_po_filter(direction, df, ma20, ma50, ma200, rsi, atr)

        return {
            "Ticker": ticker,
            "Preço": f"${curr_price:.2f}",
            "Estratégia": sugestao,
            "Strikes (Ref)": strike_alvo,
            "Vencimento": vencimento,
            "Motivo": motivo,
            "Filtro_OK": filtro_ok,
            "Score": score, # Para o termômetro
            "_cor_fundo": cor_fundo,
            "_cor_texto": cor_texto
        }
    except Exception:
        return None

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

st.title("🎯 Trend Scanner Pro - Opções")

# Sidebar
st.sidebar.header("Configurações")
period_label = st.sidebar.selectbox("Histórico:", list(PERIOD_OPTIONS.keys()), index=0)
period = PERIOD_OPTIONS[period_label]

if st.button("🔄 Atualizar Scanner"):
    get_data.clear()

with st.spinner(f"Analisando {len(TICKERS)} ativos..."):
    raw_data = get_data(TICKERS, period=period, interval="1d")

# Processamento
results = []
alerts_to_show = []

if raw_data is not None and not raw_data.empty:
    # Verifica alertas macro
    current_date = raw_data.index[-1]
    alerts_to_show = get_macro_alerts(current_date)
    
    for ticker in TICKERS:
        df_t = get_ticker_df(raw_data, ticker)
        if df_t.empty: continue
        res = analyze_ticker(ticker, df_t)
        if res: results.append(res)

df_results = pd.DataFrame(results)

# ------------------------------------------------------------
# 1. ÁREA DE ALERTAS MACRO
# ------------------------------------------------------------
if alerts_to_show:
    st.error("🚨 **ALERTA DE RISCO MACROECONÔMICO**")
    for alert in alerts_to_show:
        with st.container():
            st.markdown(f"### 📅 {alert['event']}")
            if alert['days'] == 0:
                st.markdown("**HOJE! Cuidado redobrado.**")
            else:
                st.markdown(f"Faltam **{alert['days']} dias**.")
            st.info(f"💡 **O que fazer:** {alert['guide']}")
    st.divider()
else:
    st.success("✅ Cenário Macro livre de eventos críticos (FOMC/CPI/Payroll) nos próximos 3 dias.")
    with st.expander("📅 Ver Próximos Eventos Relevantes"):
        st.table(pd.DataFrame(MACRO_EVENTS))

# ------------------------------------------------------------
# 2. TERMÔMETRO DE VIÉS E PROTEÇÃO
# ------------------------------------------------------------
if not df_results.empty:
    # Filtrar apenas sinais válidos para o termômetro
    df_valid = df_results[df_results["Filtro_OK"] == True].copy()
    
    if not df_valid.empty:
        # Calcula a média dos scores (-2 a +2)
        avg_score = df_valid["Score"].mean()
        
        st.divider()
        st.subheader("🌡️ Termômetro de Sentimento do Mercado")
        
        col_term, col_prot = st.columns([1, 2])
        
        with col_term:
            # Exibe o valor do score
            label = "NEUTRO"
            delta_color = "off"
            if avg_score > 0.5: 
                label = "VIÉS DE ALTA"
                delta_color = "normal" # Verde
            elif avg_score < -0.5: 
                label = "VIÉS DE BAIXA"
                delta_color = "inverse" # Vermelho
            
            st.metric("Sentimento Agregado", f"{label} ({avg_score:.2f})", delta=avg_score, delta_color=delta_color)
        
        with col_prot:
            # Lógica de Proteção (Hedge)
            if avg_score > 1.0:
                st.warning("⚠️ **ALERTA DE EUFORIA (Mercado Esticado):** Risco de correção.")
                st.markdown("""
                **🛡️ Como se Proteger (Hedge):**
                1. **Não aumente a exposição:** Evite abrir muitas novas Calls agora.
                2. **Proteção (Hedge):** Considere comprar **Puts de índice (SPY/QQQ) curtas (15-30 dias)**. Se o mercado corrigir, elas valorizam e compensam a queda das Calls.
                3. **Travas:** Prefira Travas de Alta (risco limitado) a compras secas.
                """)
            elif avg_score < -1.0:
                st.warning("⚠️ **ALERTA DE PÂNICO (Tendência de Baixa):** Cuidado com repiques.")
                st.markdown("""
                **🛡️ Como se Proteger (Hedge):**
                1. **Não tente adivinhar o fundo:** Não compre Calls "porque caiu muito".
                2. **Proteção:** Se tiver carteira de ações, mantenha **Puts longas** ou venda Calls cobertas (OTM) para gerar caixa.
                3. **Espere:** Aguarde o score voltar para > -0.5 para pensar em compras.
                """)
            else:
                st.info("ℹ️ **Mercado Equilibrado:** O viés não está extremo.")
                st.markdown("Siga os sinais individuais da tabela abaixo com a gestão de risco padrão (1-2% por trade).")

# ------------------------------------------------------------
# 3. TABELA
# ------------------------------------------------------------
    st.divider()
    if df_valid.empty:
        st.warning("Nenhum ativo passou nos filtros de segurança hoje.")
    else:
        st.subheader(f"Oportunidades ({len(df_valid)})")
        
        opcoes = df_valid["Estratégia"].unique()
        filtro = st.sidebar.multiselect("Filtrar Estratégia:", opcoes, default=[x for x in opcoes if x != "Aguardar"])
        
        if filtro:
            df_show = df_valid[df_valid["Estratégia"].isin(filtro)].copy()
        else:
            df_show = df_valid.copy()

        # Reset Index para 1, 2, 3...
        df_show.reset_index(drop=True, inplace=True)
        df_show.index = df_show.index + 1
        
        # Colunas para exibir (apenas as colunas finais visíveis)
        cols_to_show = ["Ticker", "Preço", "Estratégia", "Strikes (Ref)", "Vencimento", "Motivo"]
        
        # Função de estilo que usa o índice para acessar o DF original (df_show)
        def apply_row_colors(row):
            idx = row.name 
            bg_color = df_show.loc[idx, "_cor_fundo"]
            text_color = df_show.loc[idx, "_cor_texto"]
            return [f'background-color: {bg_color}; color: {text_color}' for _ in row]

        # Aplica o estilo
        st.dataframe(
            df_show[cols_to_show].style.apply(apply_row_colors, axis=1),
            use_container_width=True,
            height=600
        )

else:
    st.error("Erro ao carregar dados.")

# ------------------------------------------------------------
# 4. GRÁFICO
# ------------------------------------------------------------
st.divider()
st.subheader("Análise Gráfica")
sel = st.selectbox("Ver Gráfico:", TICKERS)

if sel and raw_data is not None:
    try:
        df_chart = get_ticker_df(raw_data, sel)
        if not df_chart.empty:
            df_chart["MA20"] = ta.sma(df_chart["Close"], length=MA_SHORT)
            df_chart["MA50"] = ta.sma(df_chart["Close"], length=MA_MEDIUM)
            donchian = df_chart["High"].rolling(window=DONCHIAN_LEN).max()

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart["Open"], high=df_chart["High"], low=df_chart["Low"], close=df_chart["Close"], name="Preço"))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["MA20"], line=dict(color='orange', width=1), name="MA20"))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["MA50"], line=dict(color='blue', width=2), name="MA50"))
            fig.add_trace(go.Scatter(x=df_chart.index, y=donchian, line=dict(color='green', width=1, dash='dot'), name="Topo 20d"))
            
            fig.update_layout(xaxis_rangeslider_visible=False, title=f"{sel} - Diário", height=600)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro no gráfico: {e}")
