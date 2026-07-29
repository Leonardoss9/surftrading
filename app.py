import streamlit as st
import yfinance as yf
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import google.genai as genai

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="WAVE_UP - Pro Trading & AI Analyzer",
    page_icon="📈",
    layout="wide"
)

# Sidebar per le impostazioni e l'API Key
st.sidebar.title("⚙️ Configurazione")
api_key = st.sidebar.text_input("Inserisci Google Gemini API Key:", type="password")

# Inizializzazione Client AI
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
        st.sidebar.success("API Key collegata con successo!")
    except Exception as e:
        st.sidebar.error("Errore nel collegamento dell'API Key.")

# --- FUNZIONI DI CALCOLO TECNICO ---
def analizza_asset_avanzato(ticker_symbol):
    asset = yf.Ticker(ticker_symbol)
    df = asset.history(period="2y", interval="1d") 
    if df.empty: return None, None
        
    df.ta.rsi(append=True)
    df.columns = [str(c) for c in df.columns]
    
    df['Macro_Min'] = df['Low'].rolling(window=40, center=True).min()
    df['Macro_Max'] = df['High'].rolling(window=40, center=True).max()
    
    macro_pivots = []
    for i in range(40, len(df)-40):
        if df['Low'].iloc[i] == df['Macro_Min'].iloc[i]:
            macro_pivots.append({"tipo": "MIN_MACRO", "data": str(df.index[i].date()), "prezzo": float(df['Low'].iloc[i])})
        elif df['High'].iloc[i] == df['Macro_Max'].iloc[i]:
            macro_pivots.append({"tipo": "MAX_MACRO", "data": str(df.index[i].date()), "prezzo": float(df['High'].iloc[i])})
            
    fib_levels = {}
    massimi = [p for p in macro_pivots if p["tipo"] == 'MAX_MACRO']
    minimi = [p for p in macro_pivots if p["tipo"] == 'MIN_MACRO']
    if massimi and minimi:
        max_st, min_st = massimi[-1]["prezzo"], minimi[-1]["prezzo"]
        distanza = max_st - min_st
        fib_levels['Fib_0.618'] = round(max_st - (distanza * 0.618), 2)
        fib_levels['Est_1.618'] = round(min_st, 2)
            
    return df, {"pivots": macro_pivots[-5:], "fibonacci": fib_levels}

def ottieni_sentiment_notizie(ticker_symbol):
    analyzer = SentimentIntensityAnalyzer()
    url = f"https://news.google.com/rss/search?q={ticker_symbol}&hl=it&gl=IT&ceid=IT:it"
    titoli = []
    try:
        response = requests.get(url, timeout=5)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        for item in root.findall('.//item')[:8]:
            titoli.append(item.find('title').text)
        if not titoli: return "Nessuna notizia recente.", 0
        total_score = sum([analyzer.polarity_scores(t)['compound'] for t in titoli])
        avg_score = total_score / len(titoli)
        return f"Analizzate {len(titoli)} notizie (Score: {avg_score:.2f})", avg_score
    except:
        return "Sentiment non disponibile.", 0

def genera_report_ai(ticker, ultimo_prezzo, rsi, pivots, sentiment_testo):
    if not client:
        return "⚠️ Inserisci una chiave API di Gemini valida nella barra laterale a sinistra per generare l'analisi reportistica."
    prompt = f"""
    Agisci come l'algoritmo proprietario senior del canale 'WAVE_UP', specializzato in Onde di Elliott, Fibonacci e Pattern 1-2-3 Intraday/Swing.
    Analizza {ticker}. Prezzo: {ultimo_prezzo:.2f}, RSI: {rsi:.2f}, Pivots/Fibonacci: {pivots}, Sentiment: {sentiment_testo}.

    FORMATTA IL REPORT PER LA PAGINA WEB:
    ### **{ticker} – REPORT STRUTTURALE**
    * **Descrizione Struttura**: (Sintesi concisa dello stato del trend)
    * **Area tecnica di riferimento (Ingresso)**: Indicata nei dati convertiti
    * **Livello tecnico di invalidazione (Stop Loss)**: Indicato nei dati convertiti

    #### **GESTIONE DEL RISCHIO (Breakeven)**:
    - *Spostare Stop IN PARI quando il prezzo raggiunge*: (Valore a R/R 1:2 per azzerare il rischio)

    #### **PROIEZIONI TEORICHE DEL MODELLO (Target Prices)**:
    - **TP1**: Target 1 (R/R: 2.0) – Statistica: 65%
    - **TP2**: Target 2 (R/R: 3.5) – Statistica: 50%
    - **TP3**: Target 3 (R/R: 5.19) – Statistica: 35%

    *Tutela del patrimonio e minimo guadagno garantito.*
    """
    try:
        return client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
    except Exception as e:
        return f"Errore durante la generazione AI: {str(e)}"

def disegna_grafico_web(df, ticker, ingresso, stop_loss, tp1, tp2, tp3):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_width=[0.3, 0.7])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Prezzo'), row=1, col=1)
    
    col_ema50 = [c for c in df.columns if 'EMA' in c and '50' in c]
    if col_ema50: fig.add_trace(go.Scatter(x=df.index, y=df[col_ema50[0]], line=dict(color='orange', width=1.2), name='EMA 50'), row=1, col=1)
    
    ultima_data = df.index[-1]
    date_future = [ultima_data + pd.Timedelta(days=i) for i in range(1, 20)]
    
    # Rettangoli Risk/Reward stile TradingView
    fig.add_shape(type="rect", x0=df.index[-5], y0=stop_loss, x1=date_future[8], y1=ingresso, fillcolor="rgba(255, 0, 0, 0.2)", line=dict(color="red", width=1), row=1, col=1)
    fig.add_shape(type="rect", x0=df.index[-5], y0=ingresso, x1=date_future[8], y1=tp1, fillcolor="rgba(0, 180, 216, 0.25)", line=dict(color="#00b4d8", width=1), row=1, col=1)
    fig.add_shape(type="rect", x0=date_future[4], y0=ingresso, x1=date_future[13], y1=tp2, fillcolor="rgba(0, 150, 199, 0.2)", line=dict(color="#0096c7", width=1), row=1, col=1)
    fig.add_shape(type="rect", x0=date_future[9], y0=ingresso, x1=date_future[18], y1=tp3, fillcolor="rgba(3, 4, 94, 0.15)", line=dict(color="#03045e", width=1), row=1, col=1)
    
    fig.add_hline(y=ingresso, line_dash="solid", line_color="gray", row=1, col=1)
    fig.update_layout(title=f"Grafico Analitico Proiettato: {ticker}", xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    return fig

# --- INTERFACCIA WEB PRINCIPALE ---
st.title("📈 WAVE_UP - Piattaforma di Analisi & Trading Algoritmo")

modalita = st.sidebar.radio("Seleziona Modalità:", ["Analisi Singolo Asset", "Screener di Mercato Automatico"])

# Tasso cambio globale
try: tasso_cambio = float(yf.Ticker("EURUSD=X").history(period="1d")['Close'].iloc[-1])
except: tasso_cambio = 1.09

# MODALITÀ 1: ANALISI SINGOLO ASSET
if modalita == "Analisi Singolo Asset":
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Inserisci il Ticker da analizzare (es. BTC-USD, NVDA, STLAM.MI, NIO):", value="BTC-USD").upper()
    with col2:
        st.write("")
        st.write("")
        btn_analizza = st.button("Avvia Analisi", type="primary")

    if btn_analizza or ticker_input:
        with st.spinner(f"Elaborazione dati matematici per {ticker_input}..."):
            df, struttura = analizza_asset_avanzato(ticker_input)
            
            if df is not None:
                sentiment_text, _ = ottieni_sentiment_notizie(ticker_input)
                valuta = "EUR" if (".MI" in ticker_input or "EUR" in ticker_input or ".DE" in ticker_input) else "USD"
                chiusura = float(df['Close'].iloc[-1])
                rsi_val = float(df[[c for c in df.columns if 'RSI' in c][0]].iloc[-1])
                
                # Calcoli operativi
                p_in = chiusura
                p_stop = p_in * 0.955
                p_tp1, p_tp2, p_tp3 = p_in * 1.06, p_in * 1.12, p_in * 1.18
                
                def conv(v): return f"${v:,.2f} (€{v/tasso_cambio:,.2f})" if valuta == "USD" else f"€{v:,.2f} (${v*tasso_cambio:,.2f})"
                
                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prezzo Attuale", conv(p_in))
                m2.metric("Stop Loss (-4.5%)", conv(p_stop))
                m3.metric("Target TP1 (+6%)", conv(p_tp1))
                m4.metric("RSI (14d)", f"{rsi_val:.1f}")

                # Grafico e Report affiancati o sovrapposti
                st.plotly_chart(disegna_grafico_web(df, ticker_input, p_in, p_stop, p_tp1, p_tp2, p_tp3), use_container_width=True)
                
                st.markdown("---")
                istruzioni_v = f"Prezzo: {conv(p_in)}, Stop: {conv(p_stop)}, TP1: {conv(p_tp1)}, TP2: {conv(p_tp2)}, TP3: {conv(p_tp3)}"
                report = genera_report_ai(ticker_input, chiusura, rsi_val, f"{str(struttura)} \n VALUTE: {istruzioni_v}", sentiment_text)
                st.markdown(report)
            else:
                st.error(f"Impossibile trovare dati per il ticker '{ticker_input}'.")

# MODALITÀ 2: SCREENER DI MERCATO AUTOMATICO
else:
    st.subheader("🔍 Screener Multi-Asset in Tempo Reale")
    st.write("Scansiona contemporaneamente Crypto, Azioni USA, Azioni Europee ed ETF per individuare i migliori setup di ingresso.")
    
    if st.button("Avvia Scansione Mercati", type="primary"):
        paniere = {
            "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "AZIONI USA": ["AAPL", "NVDA", "TSLA", "PLUG", "RIOT", "SMCI", "NIO"],
            "AZIONI EU": ["STLAM.MI", "RACE.MI", "UCG.MI"],
            "ETF": ["SPY", "QQQ", "VWCE.DE"]
        }
        risultati = []
        progress_bar = st.progress(0)
        total_items = sum([len(v) for v in paniere.values()])
        curr_item = 0
        
        for cat, tickers in paniere.items():
            for t in tickers:
                curr_item += 1
                progress_bar.progress(curr_item / total_items)
                try:
                    df = yf.Ticker(t).history(period="1y", interval="1d")
                    if df.empty or len(df) < 50: continue
                    df.ta.rsi(append=True)
                    df.ta.ema(length=50, append=True)
                    df.columns = [str(c) for c in df.columns]
                    
                    pr = float(df['Close'].iloc[-1])
                    rsi_v = float(df[[c for c in df.columns if 'RSI' in c][0]].iloc[-1])
                    ema50_v = float(df[[c for c in df.columns if 'EMA' in c and '50' in c][0]].iloc[-1])
                    min_r = float(df['Low'].rolling(window=30).min().iloc[-1])
                    
                    score = 0
                    cond = []
                    if 40 <= rsi_v <= 58: score += 35; cond.append("RSI Ottimale")
                    if abs(pr - ema50_v) / ema50_v <= 0.04: score += 35; cond.append("Su EMA 50")
                    if (pr - min_r) / min_r <= 0.06: score += 30; cond.append("Vicino Minimo")
                    
                    if score >= 40:
                        v_str = "EUR" if (".MI" in t or "EUR" in t or ".DE" in t) else "USD"
                        risultati.append({"Ticker": t, "Categoria": cat, "Prezzo": round(pr, 2), "Valuta": v_str, "RSI": round(rsi_v, 1), "Score Setup": score, "Motivazione": " & ".join(cond)})
                except: continue
                
        progress_bar.empty()
        
        if risultati:
            df_res = pd.DataFrame(risultati).sort_values(by="Score Setup", ascending=False).reset_index(drop=True)
            st.success("Scansione completata!")
            st.dataframe(df_res, use_container_width=True)
        else:
            st.warning("Nessun asset soddisfa i criteri ottimali in questo momento.")
