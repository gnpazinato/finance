import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import logging
from datetime import datetime, date, timedelta
import calendar

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
# FUNÇÕES DE CALENDÁRIO PARA EVENTOS MACRO
# ============================================================

def first_weekday(year: int, month: int, weekday: int) -> date:
    """
    Retorna a 1ª ocorrência de um dia da semana (0=Seg, 4=Sex, etc) no mês.
    """
    for day in range(1, 8):
        d = date(year, month, day)
        if d.weekday() == weekday:
            return d
    # fallback (não deve acontecer)
    return date(year, month, 1)


def last_weekday(year: int, month: int, weekday: int) -> date:
    """
    Retorna a última ocorrência de um dia da semana no mês.
    """
    _, last_day = calendar.monthrange(year, month)
    for day in range(last_day, last_day - 7, -1):
        d = date(year, month, day)
        if d.weekday() == weekday:
            return d
    return date(year, month, last_day)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """
    Retorna a n-ésima ocorrência de um dia da semana no mês.
    Ex: 3ª quarta -> weekday=2 (quarta), n=3
    """
    first = first_weekday(year, month, weekday)
    return first + timedelta(days=7 * (n - 1))


def generate_macro_events(months_ahead: int = 6):
    """
    Gera automaticamente um calendário estimado de eventos macroeconômicos
    para os próximos 'months_ahead' meses, a partir da data atual.

    Regras:
    - Payroll: 1ª sexta-feira do mês
    - CPI: 2ª quarta-feira do mês
    - PCE: última sexta-feira do mês
    - FOMC (Decisão de Juros): 3ª quarta-feira a cada 2 meses (aproximação)
    """
    today = date.today()
    events = []
    current_year = today.year
    current_month = today.month

    for i in range(months_ahead):
        # Calcula o mês/ano alvo
        month = current_month + i
        year = current_year + (month - 1) // 12
        month = (month - 1) % 12 + 1

        # Payroll: 1ª sexta-feira (weekday=4)
        payroll_date = first_weekday(year, month, 4)
        events.append({
            "name": "Payroll",
            "date": payroll_date.strftime("%Y-%m-%d"),
            "impact": -2
        })

        # CPI: 2ª quarta-feira (weekday=2)
        first_wed = first_weekday(year, month, 2)
        cpi_date = first_wed + timedelta(days=7)
        events.append({
            "name": "CPI",
            "date": cpi_date.strftime("%Y-%m-%d"),
            "impact": -2
        })

        # PCE: última sexta-feira (weekday=4)
        pce_date = last_weekday(year, month, 4)
        events.append({
            "name": "PCE",
            "date": pce_date.strftime("%Y-%m-%d"),
            "impact": -2
        })

        # FOMC: 3ª quarta-feira a cada 2 meses (aproximado, calendário estimado)
        if i % 2 == 0:  # a cada dois meses, começando pelo mês atual
            fomc_date = nth_weekday(year, month, 2, 3)  # 3ª quarta
            events.append({
                "name": "Decisão de Juros (FOMC)",
                "date": fomc_date.strftime("%Y-%m-%d"),
                "impact": -3
            })

    return events

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

# Calendário estimado automaticamente para os próximos 6 meses
MACRO_EVENTS = generate_macro_events(months_ahead=6)

NEWS_WINDOW_DAYS = 3  # Dias de alerta antes do evento


def get_macro_alerts(current_date: date):
    """Retorna alertas ativos e suas explicações."""
    if isinstance(current_date, pd.Timestamp):
        current_date = current_date.date()

    if not current_date:
        current_date = datetime.now().date()

    alerts = []
    for ev in MACRO_EVENTS:
        try:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
            days_until = (ev_date - current_date).days

            if 0 <= days_until <= NEWS_WINDOW_DAYS:
                explanation = EVENT_GUIDE.get(ev["name"], "Alta volatilidade esperada.")
                alerts.append({
                    "event": f"{ev['name']} ({ev['date']})",
                    "days": days_until,
                    "guide": explanation
                })
        except Exception:
            continue
    return alerts

# ============================================================
# FUNÇÕES TÉCNICAS (ANÁLISE)
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
            return raw_data.xs(ticker, level=0, axis=1).dropna()
        except KeyError:
            return pd.DataFrame()
    else:
        return raw_data.dropna()


def anti_po_filter(direction, df, ma20, ma50, ma200, rsi_series, atr_series):
    """
    Filtro simples anti-pó para evitar operações em condições extremas.
    """
    try:
        price = df["Close"].iloc[-1]
        curr_rsi = rsi_series.iloc[-1]
        curr_atr = atr_series.iloc[-1]
        curr_atr_pct = curr_atr / price if price > 0 else 0.0

        reasons = []
        ok = True

        # Volatilidade muito alta
        if curr_atr_pct > 0.06:
            ok = False
            reasons.append("Volatilidade extrema (ATR% > 6%)")

        # RSI extremo contra a direção
        if direction == "bull" and curr_rsi > 75:
            ok = False
            reasons.append("RSI Sobrecomprado (> 75)")
        if direction == "bear" and curr_rsi < 25:
            ok = False
            reasons.append("RSI Sobrevendido (< 25)")

        if not reasons:
            return True, "-"
        return ok, "; ".join(reasons)
    except Exception:
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
        cor_fundo = "#ffffff"
        cor_texto = "#000000"
        direction = "none"
        score = 0  # para o termômetro

        # 1. ALTA
        if curr_price > curr_ma200 and curr_ma50 > curr_ma200:
            if curr_price > prev_high_20:
                sugestao = "COMPRA CALL (Seco)"
                motivo = "Rompimento Explosivo"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#b6d7a8"
                direction = "bull"
                score = 2  # alta forte
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
                score = 1  # alta moderada

        # 2. BAIXA
        elif curr_price < curr_ma200 and curr_ma50 < curr_ma200:
            if curr_price < prev_low_20:
                sugestao = "COMPRA PUT (Seco)"
                motivo = "Perda de Suporte"
                vencimento = "Curto (15-30d)"
                strike_alvo = f"${curr_price:.0f} (ATM)"
                cor_fundo = "#ea9999"
                direction = "bear"
                score = -2  # baixa forte
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
                score = -1  # baixa moderada

        # Filtro anti-pó
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
            "Score": score,  # para o termômetro
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
        if df_t.empty:
            continue
        res = analyze_ticker(ticker, df_t)
        if res:
            results.append(res)

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
    with st.expander("📅 Ver Próximos Eventos Relevantes (estimados)"):
        st.table(pd.DataFrame(MACRO_EVENTS))

# ------------------------------------------------------------
# 2. TERMÔMETRO DE VIÉS E PROTEÇÃO
# ------------------------------------------------------------
if not df_results.empty:
    # Filtrar apenas sinais válidos para o termômetro
    df_valid = df_results[df_results["Filtro_OK"] == True].copy()

    if not df_valid.empty:
        st.divider()
        st.subheader("🌡️ Termômetro Institucional de Sentimento do Mercado (0–100)")

        # ---------- COMPONENTES DO SCORE ----------
        total_sinais = len(df_valid)
        bull_mask = df_valid["Score"] > 0
        bear_mask = df_valid["Score"] < 0

        bull_count = bull_mask.sum()
        bear_count = bear_mask.sum()
        flat_count = total_sinais - bull_count - bear_count

        pct_bull = bull_count / total_sinais if total_sinais > 0 else 0
        pct_bear = bear_count / total_sinais if total_sinais > 0 else 0

        # força média dos sinais (-2 a +2) → normalizado para [-1, 1]
        avg_strength_raw = df_valid["Score"].mean()  # [-2, 2]
        avg_strength_norm = avg_strength_raw / 2.0   # [-1, 1]

        # balance direcional pela quantidade de sinais → [-1, 1]
        dir_balance = (bull_count - bear_count) / total_sinais if total_sinais > 0 else 0

        # combinação ponderada (dê mais peso à quantidade de sinais)
        # m fica em torno de [-1, 1]
        m = 0.6 * dir_balance + 0.4 * avg_strength_norm

        # converte para escala 0–100
        tm_score = (m + 1) * 50
        tm_score = max(0, min(100, tm_score))  # clamp

        # ---------- CLASSIFICAÇÃO ----------
        if tm_score >= 80:
            label = "EUFORIA (ALTA FORTE)"
            delta_color = "normal"
            comentario_curto = "Mercado com forte predominância de sinais de alta e intensidade relevante."
        elif tm_score >= 60:
            label = "VIÉS DE ALTA"
            delta_color = "normal"
            comentario_curto = "Maioria dos sinais aponta para alta, mas ainda sem euforia extrema."
        elif tm_score <= 20:
            label = "PÂNICO / BAIXA FORTE"
            delta_color = "inverse"
            comentario_curto = "Predomínio forte de sinais de baixa e ambiente de aversão a risco."
        elif tm_score <= 40:
            label = "VIÉS DE BAIXA"
            delta_color = "inverse"
            comentario_curto = "Mais sinais de queda do que de alta, tendência de baixa moderada."
        else:
            label = "NEUTRO / EQUILIBRADO"
            delta_color = "off"
            comentario_curto = "Forças de alta e baixa relativamente equilibradas; atenção ao contexto macro."

        # Percentuais formatados
        pct_bull_str = f"{pct_bull*100:.0f}%"
        pct_bear_str = f"{pct_bear*100:.0f}%"

        col_term, col_prot = st.columns([1, 2])

        with col_term:
            st.metric(
                "Score Direcional Agregado",
                f"{tm_score:.0f}/100",
                delta=f"{label}",
                delta_color=delta_color
            )
            st.markdown(
                f"""
**Composição do Termômetro:**
- 🔼 **Bullish:** {bull_count} ativos ({pct_bull_str})
- 🔻 **Bearish:** {bear_count} ativos ({pct_bear_str})
- ⚪ **Neutros / Aguardar:** {flat_count} ativos
- 🔎 **Força média dos sinais:** {avg_strength_raw:+.2f} (escala -2 a +2)
                """
            )

        with col_prot:
            st.markdown(f"**Leitura atual:** `{label}`")
            st.markdown(comentario_curto)

            # Orientações gerais de proteção baseadas no score
            if tm_score >= 80:
                st.warning(
                    "⚠️ **Zona de EUFORIA:** mercado esticado para cima.\n\n"
                    "- Evite aumentar muito exposição direcional em Calls.\n"
                    "- Prefira **travas de alta** ao invés de Calls secas.\n"
                    "- Considere montar **hedges baratos (Puts em índices / Calls em VIX)**.\n"
                )
            elif tm_score >= 60:
                st.info(
                    "ℹ️ **Viés de Alta:** predominância de sinais bullish.\n\n"
                    "- Favorável para operações compradas em Calls/Travas de Alta.\n"
                    "- Já comece a pensar em hedges se a exposição ficar grande.\n"
                )
            elif tm_score <= 20:
                st.warning(
                    "⚠️ **Zona de Pânico / Baixa Forte:** mercado pressionado para baixo.\n\n"
                    "- Evite tentar adivinhar fundo comprando Calls muito cedo.\n"
                    "- Proteja carteiras de ações com **Puts longas** ou **Calls cobertas**.\n"
                    "- Para quem está vendido, cuidado com short squeeze em repiques fortes.\n"
                )
            elif tm_score <= 40:
                st.info(
                    "ℹ️ **Viés de Baixa:** mais sinais de queda do que de alta.\n\n"
                    "- Contexto mais favorável para **Puts** e **Travas de Baixa**.\n"
                    "- Se estiver muito comprado em ações, pense em hedge parcial.\n"
                )
            else:
                st.info(
                    "ℹ️ **Mercado Neutro / Equilibrado:**\n\n"
                    "- Evite posições muito alavancadas.\n"
                    "- Escolha operações mais assimétricas (spreads) e com risco bem definido.\n"
                    "- Deixe o gerenciamento de risco mais pesado do que o apetite direcional.\n"
                )

# ------------------------------------------------------------
# 3. TABELA DE OPORTUNIDADES
# ------------------------------------------------------------
    st.divider()
    if df_valid.empty:
        st.warning("Nenhum ativo passou nos filtros de segurança hoje.")
    else:
        st.subheader(f"Oportunidades ({len(df_valid)})")

        opcoes = df_valid["Estratégia"].unique()
        filtro = st.sidebar.multiselect(
            "Filtrar Estratégia:",
            opcoes,
            default=[x for x in opcoes if x != "Aguardar"]
        )

        if filtro:
            df_show = df_valid[df_valid["Estratégia"].isin(filtro)].copy()
        else:
            df_show = df_valid.copy()

        df_show.reset_index(drop=True, inplace=True)
        df_show.index = df_show.index + 1

        cols_to_show = ["Ticker", "Preço", "Estratégia", "Strikes (Ref)", "Vencimento", "Motivo"]

        def apply_row_colors(row):
            idx = row.name
            bg_color = df_show.loc[idx, "_cor_fundo"]
            text_color = df_show.loc[idx, "_cor_texto"]
            return [f'background-color: {bg_color}; color: {text_color}' for _ in row]

        st.dataframe(
            df_show[cols_to_show].style.apply(apply_row_colors, axis=1),
            use_container_width=True,
            height=600
        )

        # ------------------------------------------------------------
        # 5. HEDGES (SEGUROS)
        # ------------------------------------------------------------
        st.divider()
        st.subheader("🛡️ Hedges recomendados (seguros para o portfólio)")

        if not df_results.empty:

            avg_score_all = df_results[df_results["Filtro_OK"] == True]["Score"].mean()

            if avg_score_all > 0.5:
                hedge_side = "bear"
                hedge_assets = [
                    ("VXX", "Compra de PUT no SPY é cara – compre CALL longa de VXX"),
                    ("UVXY", "CALL longa (60-120 dias)"),
                    ("GLD", "CALL moderada (90 dias)"),
                    ("TLT", "CALL moderada (90 dias)"),
                    ("UUP", "CALL longa")
                ]
            elif avg_score_all < -0.5:
                hedge_side = "bull"
                hedge_assets = [
                    ("SPY", "CALL longa (ATM ou leve OTM, 60-120 dias)"),
                    ("QQQ", "CALL longa"),
                    ("XLE", "CALL longa"),
                    ("SLV", "CALL longa"),
                    ("XLF", "CALL longa")
                ]
            else:
                hedge_side = "neutral"
                hedge_assets = [
                    ("VXX", "CALL longa"),
                    ("GLD", "CALL moderada"),
                    ("TLT", "CALL moderada (60-120 dias)")
                ]

            df_hedge = pd.DataFrame(hedge_assets, columns=["Ativo", "Estratégia sugerida"])

            st.dataframe(
                df_hedge,
                use_container_width=True,
                height=280
            )

        else:
            st.info("Sem dados para analisar hedges no momento.")

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
                line=dict(color='orange', width=1),
                name="MA20"
            ))
            fig.add_trace(go.Scatter(
                x=df_chart.index,
                y=df_chart["MA50"],
                line=dict(color='blue', width=2),
                name="MA50"
            ))
            fig.add_trace(go.Scatter(
                x=df_chart.index,
                y=donchian,
                line=dict(color='green', width=1, dash='dot'),
                name="Topo 20d"
            ))

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                title=f"{sel} - Diário",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro no gráfico: {e}")
