# Live Macro & Market Data (Streamlit)

A small Streamlit app that fetches live data from **FRED** and **Yahoo Finance**
on a button click, with response caching, secrets-based API keys, and
per-source error handling. Meant as a starting template — add more `fetch_*`
functions for other platforms (World Bank, ECB SDW, Alpha Vantage, etc.).

## Project structure

```
live-data-app/
├── app.py                          # the Streamlit app
├── requirements.txt                # Python dependencies
├── .gitignore                      # excludes real secrets from git
└── .streamlit/
    └── secrets.toml.example        # template — copy to secrets.toml locally
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste in your own FRED API key
# (free key: https://fred.stlouisfed.org/docs/api/api_key.html)

streamlit run app.py
```

The app opens at `http://localhost:8501`. Yahoo Finance data needs no key;
FRED data needs the key above.

## Push to GitHub

If you're adding this to your existing `Streamlit-Projects` repo, drop these
files into a subfolder (e.g. `live-data-app/`) and commit:

```bash
git add live-data-app/
git commit -m "Add live FRED/Yahoo Finance data demo"
git push
```

Or, to make it a standalone repo:

```bash
cd live-data-app
git init
git add .
git commit -m "Initial commit: live macro/market data Streamlit app"
git branch -M main
git remote add origin https://github.com/hormozramian/<new-repo-name>.git
git push -u origin main
```

`.streamlit/secrets.toml` is excluded by `.gitignore`, so your real API key
never gets committed — only `secrets.toml.example` does.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick the repo/branch, and set the main file path to
   `app.py` (or `live-data-app/app.py` if it's a subfolder of a larger repo).
4. In the app's **Settings → Secrets** panel, paste the same key you put in
   `secrets.toml` locally:
   ```toml
   FRED_API_KEY = "your_fred_api_key_here"
   ```
5. Deploy. The app rebuilds automatically on every push to the connected
   branch.

## Extending to more data sources

Each source should get its own cached fetch function, following the same
pattern as `fetch_fred_series` / `fetch_yahoo_prices` in `app.py`:

```python
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank_series(indicator: str, country: str) -> pd.DataFrame:
    import requests
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()[1])
```

Keep each source's failure isolated (its own `try/except` in the UI section)
so one API being down doesn't take out the whole page.
