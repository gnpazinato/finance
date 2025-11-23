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
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y"
}

# ============================================================
# MÓDULO DE RISCO MACRO (NEWS)
# ============================================================

# Preencha esta lista com os principais eventos macro que te interessam.
# Exemplo:
# MACRO_EVENTS = [
#     {"name": "FOMC", "date": "2025-12-17", "impact": -2},
#     {"name": "CPI EUA", "date": "2025-12-12", "impact": -1},
# ]
MACRO_EVENTS = []
NEWS_WINDOW_DAYS = 1  # quantos dias antes/depois do evento considerar "zona de risco"


def get_macro_risk_score(current_date: date):
    """
    Calcula um score de risco de notícias (NScore) para a data atual,
    com base na lista MACRO_EVENTS.

    Retorna:
    - score (int)
    - lista de eventos relevantes próximos
    """
    if isinstance(current_date, pd.Timestamp):
        current_date = current_date.date()

    score = 0
    active_events = []

    for ev in MACRO_EVENTS:
        try:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        diff = abs((ev_date - current_date).days)
        if diff <= NEWS_WINDOW_DAYS:
            impact = ev.get("impact", -1)
            score += impact
            active_events.append(f"{ev['name']} ({ev['date']})")

    return score, active_events


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


def anti_po_filter(direction, df, ma20, ma50, ma200, rsi_series, atr_series):
    """
    Filtro "anti-pó" para evitar operações em condições de risco extremo.

    direction: "bull", "bear" ou "none"
    Retorna (ok, motivo_filtro)
    """
    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        price = close.iloc[-1]
        curr_rsi = rsi_series.iloc[-1]

        # ATR percentual (volatilidade média)
        curr_atr = atr_series.iloc[-1]
        curr_atr_pct = curr_atr / price if pd.notna(curr_atr) and price > 0 else 0.0

        # Range do último candle (ex: exaustão)
        rng = (high.iloc[-1] - low.iloc[-1]) / price if price > 0 else 0.0

        # Distância da MA50
        curr_ma50 = ma50.iloc[-1]
        dist_ma50 = abs(price - curr_ma50) / price if price > 0 and pd.notna(curr_ma50) else 0.0

        reasons = []
        ok = True

        # 1) Volatilidade média muito alta (ex.: > 6%)
        if curr_atr_pct > 0.06:
            ok = False
            reasons.append("Volatilidade média muito alta (ATR% > 6%)")

        # 2) Candle de exaustão (range do dia > 8%)
        if rng > 0.08:
            ok = False
            reasons.append("Candle de exaustão (range diário > 8%)")

        # 3) RSI extremo contra a direção da operação
        if direction == "bull" and curr_rsi > 70:
            ok = False
            reasons.append("RSI sobrecomprado (> 70) para compra")
        if direction == "bear" and curr_rsi < 30:
            ok = False
            reasons.append("RSI sobrevendido (< 30) para venda")

        # 4) Preço muito distante da MA50 (ativo esticado demais)
        if dist_ma50 > 0.10:
            ok = False
            reasons.append("Preço muito distante da MA50 (> 10%)")

        if not reasons:
            return True, "-"

        return ok, "; ".join(reasons)

    except Exception as e:
        logging.exception(f"Erro no anti_po_filter: {e}")
        # Em caso de erro no filtro, melhor não bloquear
        return True, "Erro no filtro anti-pó (não aplicado)"


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
        atr = ta.atr(high, low, close, length=14)

        donchian_high = high.rolling(window=DONCHIAN_LEN).max()
        donchian_low = low.rolling(window=DONCHIAN_LEN).min()

        # Pega os valores atuais (última barra)
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
        direction = "none"  # bull, bear ou none

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
                direction = "bull"

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
                direction = "bull"

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
                direction = "bear"

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
                direction = "bear"

        # ===== APLICA FILTRO ANTI-PÓ =====
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
            "Motivo_Filtro": motivo_filtro,
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
# RESUMO + TERMÔMETRO + EXPOSIÇÃO + TABELA
# ============================================================

if not df_results.empty and "Estratégia" in df_results.columns:

    # Considera apenas os sinais aprovados pelo filtro anti-pó
    df_valid = df_results[df_results["Filtro_OK"]].copy()
    if df_valid.empty:
        st.warning("Nenhuma oportunidade aprovada pelos filtros de risco (anti-pó).")
    else:
        st.subheader("Resumo das Sinalizações")

        estrategias_prioritarias = [
            "COMPRA CALL (Seco)",
            "TRAVA DE ALTA (Call Spread)",
            "COMPRA PUT (Seco)",
            "TRAVA DE BAIXA (Put Spread)"
        ]

        cols = st.columns(len(estrategias_prioritarias))
        for col, est in zip(cols, estrategias_prioritarias):
            qtd = (df_valid["Estratégia"] == est).sum()
            col.metric(est, qtd)

        # ===== SCORE DIRECIONAL (Técnico) =====
        score_map = {
            "COMPRA CALL (Seco)": 2,
            "TRAVA DE ALTA (Call Spread)": 1,
            "COMPRA PUT (Seco)": -2,
            "TRAVA DE BAIXA (Put Spread)": -1,
            "Aguardar": 0
        }

        df_valid["ScoreDirecional"] = df_valid["Estratégia"].map(score_map).fillna(0)
        score_medio = df_valid["ScoreDirecional"].mean()
        total_sinais = len(df_valid)

        # ===== RISCO MACRO (News Score) =====
        if raw_data is not None and len(raw_data.index) > 0:
            data_ref = raw_data.index[-1].date()
        else:
            data_ref = datetime.today().date()

        macro_score, macro_events = get_macro_risk_score(data_ref)

        # Inputs de risco na sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📉 Gestão de Exposição")

        capital = st.sidebar.number_input("Capital total (USD):", value=10000.0, step=100.0)
        risco_pct = st.sidebar.number_input("Risco por operação (%):", value=1.0, step=0.5)
        limite_trades = st.sidebar.number_input("Limite máximo de trades:", value=5, step=1)

        risco_unit = capital * (risco_pct / 100.0)

        # Exposição planejada se entrar em todos os sinais aprovados
        exposicao_total_planejada = total_sinais * risco_unit
        exposicao_direcional = df_valid["ScoreDirecional"].sum() * risco_unit

        # Interpretação do termômetro técnico
        if score_medio > 1:
            sentimento = "Mercado com forte viés de ALTA (bullish concentrado)."
        elif score_medio > 0.3:
            sentimento = "Mercado com viés de alta."
        elif score_medio < -1:
            sentimento = "Mercado com forte viés de BAIXA (bearish concentrado)."
        elif score_medio < -0.3:
            sentimento = "Mercado com viés de baixa."
        else:
            sentimento = "Mercado mais neutro / indefinido pelos sinais do scanner."

        # Emojis de cor da exposição direcional
        if exposicao_direcional > 0:
            emoji_dir = "🟢"
            dir_txt = "Risco agregado apontando para movimentos de ALTA."
        elif exposicao_direcional < 0:
            emoji_dir = "🔴"
            dir_txt = "Risco agregado apontando para movimentos de BAIXA."
        else:
            emoji_dir = "⚪"
            dir_txt = "Risco agregado próximo de neutro."

        # Interpretação de NScore
        if macro_score < 0:
            macro_txt = "Risco macro elevado (eventos importantes próximos). Considere reduzir tamanho de posição ou adiar novas entradas."
        elif macro_score > 0:
            macro_txt = "Contexto macro levemente favorável conforme eventos cadastrados."
        else:
            macro_txt = "Nenhum evento macro relevante cadastrado para esta data (NScore = 0)."

        # Termômetro (métrica principal técnica)
        st.metric(
            "Termômetro Direcional Técnico (Score médio)",
            f"{score_medio:.2f}",
            sentimento
        )

        # Resumo de exposição (abaixo do termômetro)
        eventos_txt = ", ".join(macro_events) if macro_events else "Nenhum evento macro configurado na janela selecionada."
        st.markdown(
            f"""
**Resumo de Exposição (apenas sinais aprovados):**

{emoji_dir} **Exposição Direcional estimada:** `${exposicao_direcional:,.2f}`  
💰 **Exposição Total planejada (se entrar em todos os sinais aprovados):** `${exposicao_total_planejada:,.2f}`  

📰 **NScore (Risco de Notícias):** `{macro_score}`  
_Eventos macro próximos:_ {eventos_txt}  

_{sentimento}_  
_{dir_txt}_  
_{macro_txt}_
            """
        )

        # Resumo também na sidebar
        st.sidebar.markdown(f"**Sinais aprovados:** {total_sinais}")
        st.sidebar.markdown(f"**Risco unitário por operação:** `${risco_unit:,.2f}`")
        st.sidebar.markdown(f"{emoji_dir} **Exposição Direcional:** `${exposicao_direcional:,.2f}`")
        st.sidebar.markdown(f"💰 **Exposição Total planejada:** `${exposicao_total_planejada:,.2f}`")
        st.sidebar.markdown(f"📰 **NScore (Risco macro):** {macro_score}")

        if total_sinais > limite_trades:
            st.sidebar.error("🚨 A quantidade de sinais aprovados excede o limite de trades simultâneos definido.")
        else:
            st.sidebar.success("Quantidade de sinais dentro do limite definido. ✔")

        # ===== FILTROS E TABELA DE OPORTUNIDADES =====

        # Filtros na sidebar só consideram sinais aprovados
        opcoes = df_valid["Estratégia"].unique()
        default_filtro = [x for x in opcoes if x != "Aguardar"]

        filtro = st.sidebar.multiselect(
            "Filtrar por Operação:",
            options=opcoes,
            default=default_filtro
        )

        # Aplica filtro
        if filtro:
            df_final = df_valid[df_valid["Estratégia"].isin(filtro)].copy()
        else:
            df_final = df_valid.copy()

        # Ajusta índice para começar em 1
        df_final.reset_index(drop=True, inplace=True)
        df_final.index = df_final.index + 1

        # Função de estilização por linha
        def apply_style(row):
            bg = row["_cor_fundo"]
            txt = row["_cor_texto"]
            return [f"background-color: {bg}; color: {txt}" for _ in row]

        st.subheader("Oportunidades Identificadas (após filtros de risco)")

        # DataFrame estilizado, ocultando colunas de cor e de controle
        styled = (
            df_final
            .style
            .apply(apply_style, axis=1)
            .hide(axis="columns", subset=["_cor_fundo", "_cor_texto", "Filtro_OK", "Motivo_Filtro"])
        )

        st.dataframe(
            styled,
            use_container_width=True,
            height=600
        )

        # Botão para exportar sinais em CSV (mantém as colunas internas no arquivo)
        st.download_button(
            "📥 Baixar sinais em CSV (inclui colunas de filtro)",
            df_final.to_csv(index=True).encode("utf-8"),
            file_name=f"trend_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

        # Explicação opcional
        with st.expander("Como interpretar as estratégias e filtros?"):
            st.markdown(
                """
- **COMPRA CALL (Seco)**: operação direcional apostando em alta forte de curto prazo.
- **TRAVA DE ALTA (Call Spread)**: operação direcional de alta com risco e ganho máximos limitados.
- **COMPRA PUT (Seco)**: operação direcional apostando em queda forte de curto prazo.
- **TRAVA DE BAIXA (Put Spread)**: operação direcional de baixa com risco e ganho máximos limitados.
- **Aguardar**: nenhum setup claro de acordo com os critérios definidos.

### Filtro Anti-Pó (técnico)

Um sinal só aparece aqui se o filtro aprovar, levando em conta:
- Volatilidade média (ATR%) muito alta (evita entrar em ambientes caóticos).
- Candle de exaustão (range diário muito grande).
- RSI extremo contra a direção da operação (evita comprar topo e vender fundo).
- Preço muito distante da MA50 (ativo esticado demais).

### NScore (Risco Macro)

Você pode cadastrar eventos macro em `MACRO_EVENTS` (FOMC, CPI, etc.).
O NScore negativo indica maior cautela para novas entradas.

> Este painel é apenas um scanner técnico + filtros de risco e **não constitui recomendação de investimento.**
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

        # --- Donchian High no gráfico (máxima 20d) ---
        donchian_high_chart = df_chart["High"].rolling(window=DONCHIAN_LEN).max()
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=donchian_high_chart,
            line=dict(color="green", width=1, dash="dot"),
            name=f"Resistência {DONCHIAN_LEN}d (Rompimento)"
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
                st.table(sel_info[["Estratégia", "Strikes (Ref)", "Vencimento", "Motivo", "Motivo_Filtro"]])

    except Exception as e:
        st.error(f"Gráfico indisponível para {sel}: {e}")
