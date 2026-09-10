"""
Live Macro & Market Data — a Streamlit app that fetches data from FRED and
Yahoo Finance on demand (button click), with caching, secrets-based API keys,
and per-source error handling.

Local run:
    pip install -r requirements.txt
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    # edit .streamlit/secrets.toml and add your free FRED API key
    streamlit run app.py

Deploy: see README.md for Streamlit Community Cloud instructions.
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Live Macro & Market Data", layout="wide")
st.title("Live Data Demo: FRED + Yahoo Finance")
st.caption(
    "Click the button to pull fresh data. Results are cached briefly "
    "(see TTLs below) so repeated clicks don't hammer the APIs."
)

# ---------------------------------------------------------------------------
# Cached fetch functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)  # re-fetch at most once an hour
def fetch_fred_series(series_id: str) -> pd.Series:
    from fredapi import Fred
    api_key = st.secrets.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Add it to .streamlit/secrets.toml locally, "
            "or in the app's Secrets panel on Streamlit Community Cloud."
        )
    fred = Fred(api_key=api_key)
    return fred.get_series(series_id)


@st.cache_data(ttl=900, show_spinner=False)  # re-fetch at most every 15 minutes
def fetch_yahoo_prices(ticker: str, period: str = "6mo") -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    return data


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    fred_series = st.text_input(
        "FRED series ID", value="CPIAUCSL",
        help="e.g. CPIAUCSL (CPI), UNRATE (unemployment), GDP, FEDFUNDS",
    )
with col2:
    ticker = st.text_input(
        "Ticker (Yahoo Finance)", value="^GSPC",
        help="e.g. ^GSPC (S&P 500), ^VIX, AAPL, GC=F (gold futures)",
    )

fetch_clicked = st.button("Fetch live data", type="primary")

if fetch_clicked:
    with st.spinner("Fetching..."):
        fred_col, yahoo_col = st.columns(2)

        with fred_col:
            st.subheader(f"FRED: {fred_series}")
            try:
                fred_data = fetch_fred_series(fred_series)
                st.line_chart(fred_data)
                st.caption(f"Latest value: {fred_data.dropna().iloc[-1]:,.2f} "
                           f"(as of {fred_data.dropna().index[-1].date()})")
            except Exception as e:
                st.error(f"Could not fetch FRED series '{fred_series}': {e}")

        with yahoo_col:
            st.subheader(f"Yahoo Finance: {ticker}")
            try:
                price_data = fetch_yahoo_prices(ticker)
                st.line_chart(price_data["Close"])
                st.caption(f"Latest close: {float(price_data['Close'].iloc[-1]):,.2f} "
                           f"(as of {price_data.index[-1].date()})")
            except Exception as e:
                st.error(f"Could not fetch ticker '{ticker}': {e}")
else:
    st.info("Enter a FRED series ID and/or ticker, then click 'Fetch live data'.")

st.divider()
st.caption(
    "Data sources: FRED (Federal Reserve Bank of St. Louis) and Yahoo Finance "
    "via yfinance. This is a demo template — extend `fetch_*` functions to add "
    "more sources (e.g. World Bank, ECB SDW, Alpha Vantage)."
)
