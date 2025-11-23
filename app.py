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

# Calendário Estimado (Baseado em padrões do FED/BLS para o período solicitado)
# ATENÇÃO: Datas futuras são estimativas baseadas no calendário padrão.
MACRO_EVENTS = [
    # DEZEMBRO 2025
    {"name": "Payroll (Relatório de Emprego)", "date": "2025-12-05", "impact": -2},
    {"name": "CPI (Inflação Consumidor)", "date": "2025-12-10", "impact": -2},
    {"name": "Decisão de Juros (FOMC)", "date": "2025-12-17", "impact": -3}, # Data crítica
    {"name": "PIB Trimestral (Final)", "date": "2025-12-21", "impact": -1},
    
    # JANEIRO 2026
    {"name": "Ata do FOMC", "date": "2026-01-07", "impact": -1},
    {"name": "Payroll (Relatório de Emprego)", "date": "2026-01-09", "impact": -2},
    {"name": "CPI (Inflação Consumidor)", "date": "2026-01-13", "impact": -2},
    {"name": "Início Temporada de Balanços (Bancos)", "date": "2026-01-16", "impact": -1},
    {"name": "Decisão de Juros (FOMC)", "date": "2026-01-28", "impact": -3}, # Data crítica
    
    # FEVEREIRO 2026
    {"name": "Payroll (Relatório de Emprego)", "date": "2026-02-06", "impact": -2},
    {"name": "CPI (Inflação Consumidor)", "date": "2026-02-12", "impact": -2},
    {"name": "Ata do FOMC", "date": "2026-02-18", "impact": -1},
    {"name": "PCE (Inflação Preferida do Fed)", "date": "2026-02-27", "impact": -2},
]

NEWS_WINDOW_DAYS = 3  # Dias de alerta antes do evento

def get_macro_risk_score(current_date: date):
    """Calcula risco macro baseado na proximidade de eventos."""
    if isinstance(current_date, pd.Timestamp):
        current_date = current_date.date()

    active_events = []
    
    # Se não tiver data atual (ex: fim de semana), usa hoje
    if not current_date:
        current_date = datetime.now().date()

    for ev in MACRO_EVENTS:
        try:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
            # Verifica eventos nos próximos dias (Janela de alerta)
            days_until = (ev_date - current_date).days
            
            if 0 <= days_until <= NEWS_WINDOW_DAYS:
                active_events.append(f"⚠️ {ev['name']} em {days_until} dia(s) ({ev['date']})")
        except:
            continue

    return active_events

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

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
        try:
            # Tenta acessar o nível superior do índice
            return raw_data.xs(ticker, level=0, axis=1).dropna()
        except KeyError:
            return pd.DataFrame()
    else:
        return raw_data.dropna()


def anti_po_filter(direction, df, ma20, ma50, ma200, rsi_series, atr_series):
    """
    Filtro técnico de segurança.
    """
    try:
        price = df["Close"].iloc[-1]
        curr_rsi = rsi_series.iloc[-1]
        curr_atr = atr_series.iloc[-1]
        
        # Volatilidade relativa
        curr_atr_pct = curr_atr / price if price > 0 else 0.0
        
        reasons = []
        ok = True

        # 1) Volatilidade Explosiva (> 6%)
        if curr_atr_pct > 0.06:
            ok = False
            reasons.append("Volatilidade extrema")

        # 2) RSI Extremo (Evitar comprar topo/vender fundo)
        if direction == "bull" and curr_rsi > 75:
            ok = False
            reasons.append("RSI Sobrecomprado (>75)")
        if direction == "bear" and curr_rsi < 25:
            ok = False
            reasons.append("RSI Sobrevendido (<25)")

        if not reasons:
            return True, "-"

        return ok, "; ".join(reasons)

    except:
        return True, "Erro Filtro"


def analyze_ticker(ticker, df):
    try:
        if df is None or df.empty or len(df) < MA_LONG + 5:
            return None

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
        # Cores HEX diretas
        cor_fundo = "#ffffff" 
        cor_texto = "#000000"
        direction = "none"

        # 1. TENDÊNCIA DE ALTA
        if curr_price > curr_ma200 and curr_ma50 > curr_ma200:
            # A) Rompimento
            if curr_price > prev_high_20:
                sugestao = "COMPRA CALL (Seco)"
                motivo = "Rompimento Explosivo"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#b6d7a8"  # Verde Claro
                direction = "bull"

            # B) Pullback
            elif (curr_price <= curr_ma20 * (1 + PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE ALTA (Call Spread)"
                motivo = "Pullback (Correção)"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 + SPREAD_CALL_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#38761d"  # Verde Escuro
                cor_texto = "#ffffff"
                direction = "bull"

        # 2. TENDÊNCIA DE BAIXA
        elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:
            # C) Perda de Fundo
            if curr_price < prev_low_20:
                sugestao = "COMPRA PUT (Seco)"
                motivo = "Perda de Suporte"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#ea9999"  # Vermelho Claro
                direction = "bear"

            # D) Pullback de Baixa
            elif (curr_price >= curr_ma20 * (1 - PULLBACK_TOL)) and (RSI_LOW < curr_rsi < RSI_HIGH):
                sugestao = "TRAVA DE BAIXA (Put Spread)"
                motivo = "Repique p/ Cair"
                vencimento = "Médio (30-45d)"
                strike_long = curr_price
                strike_short = curr_price * (1 - SPREAD_PUT_PCT)
                strike_alvo = f"C:${strike_long:.0f} / V:${strike_short:.0f}"
                cor_fundo = "#990000"  # Vermelho Escuro
                cor_texto = "#ffffff"
                direction = "bear"

        # Filtro Anti-Pó
        if direction == "none":
            filtro_ok = True
            motivo_filtro = "-"
        else:
            filtro_ok, motivo_filtro = anti_po_filter(
                direction, df, ma20, ma50, ma200, rsi, atr
            )

        return {
            "Ticker": ticker,
            "Preço": f"${curr_price:.2f}",
            "Estratégia": sugestao,
            "Strikes (Ref)": strike_alvo,
            "Vencimento": vencimento,
            "Motivo": motivo,
            "Filtro_OK": filtro_ok,
            "Alerta Filtro": motivo_filtro, # Renomeado para ficar claro
            "_cor_fundo": cor_fundo,
            "_cor_texto": cor_texto
        }

    except Exception:
        return None


# ============================================================
# INTERFACE
# ============================================================

st.title("🎯 Trend Scanner Pro - Opções")

# Sidebar Configs
st.sidebar.header("Configurações")
period_label = st.sidebar.selectbox(
    "Período do histórico:",
    list(PERIOD_OPTIONS.keys()),
    index=0 
)
period = PERIOD_OPTIONS[period_label]

if st.button("🔄 Atualizar Scanner"):
    get_data.clear()

with st.spinner(f"Analisando {len(TICKERS)} ativos..."):
    raw_data = get_data(TICKERS, period=period, interval="1d")

# Cálculo
results = []
if raw_data is not None and not raw_data.empty:
    current_date = raw_data.index[-1] # Data do último dado
    
    # Check de Eventos Macro
    macro_alerts = get_macro_risk_score(current_date)
    
    for ticker in TICKERS:
        df_t = get_ticker_df(raw_data, ticker)
        if df_t.empty: continue
        res = analyze_ticker(ticker, df_t)
        if res: results.append(res)

df_results = pd.DataFrame(results)

# ============================================================
# DASHBOARD
# ============================================================

if not df_results.empty:
    
    # 1. ALERTA MACROECONÔMICO
    if macro_alerts:
        st.error("🚨 **ALERTA DE EVENTOS MACRO PRÓXIMOS (3 DIAS):**")
        for alert in macro_alerts:
            st.write(alert)
        st.info("Em dias de eventos críticos (como FOMC ou CPI), a volatilidade pode invalidar setups técnicos. Cuidado.")
    else:
        st.success("✅ Nenhum evento macro crítico (FOMC/CPI/Payroll) previsto para os próximos 3 dias.")
        with st.expander("Ver Calendário Futuro Próximo"):
            st.table(pd.DataFrame(MACRO_EVENTS))

    st.divider()

    # 2. FILTRO DE SINAIS VÁLIDOS
    # Só mostramos o que passou no filtro de segurança
    df_valid = df_results[df_results["Filtro_OK"] == True].copy()
    
    if df_valid.empty:
        st.warning("Nenhum ativo passou nos filtros de segurança hoje.")
    else:
        st.subheader(f"Oportunidades Identificadas ({len(df_valid)})")
        
        # Filtro de Estratégia
        opcoes = df_valid["Estratégia"].unique()
        filtro = st.sidebar.multiselect("Filtrar Estratégia:", opcoes, default=[x for x in opcoes if x != "Aguardar"])
        
        if filtro:
            df_show = df_valid[df_valid["Estratégia"].isin(filtro)].copy()
        else:
            df_show = df_valid.copy()

        # Limpeza Final da Tabela
        df_show.reset_index(drop=True, inplace=True)
        df_show.index = df_show.index + 1
        
        # Colunas para exibir (Removemos as de controle interno)
        cols_to_show = ["Ticker", "Preço", "Estratégia", "Strikes (Ref)", "Vencimento", "Motivo"]
        
        # Função de Estilo SIMPLIFICADA (Evita KeyError)
        def color_rows(row):
            bg = row["_cor_fundo"]
            txt = row["_cor_texto"]
            return [f'background-color: {bg}; color: {txt}' for _ in row.index]

        # Aplica estilo e mostra apenas colunas relevantes
        st.dataframe(
            df_show.style.apply(color_rows, axis=1),
            column_order=cols_to_show,
            use_container_width=True,
            height=600
        )

else:
    st.error("Falha ao carregar dados. Tente novamente em instantes.")

# ============================================================
# GRÁFICO
# ============================================================
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
            
            # Mostra motivo do filtro se foi rejeitado
            if not df_results.empty:
                info = df_results[df_results["Ticker"] == sel].iloc[0]
                if not info["Filtro_OK"]:
                    st.warning(f"⚠️ Este ativo foi filtrado e não aparece na tabela principal. Motivo: {info['Alerta Filtro']}")

    except Exception as e:
        st.error(f"Erro no gráfico: {e}")
