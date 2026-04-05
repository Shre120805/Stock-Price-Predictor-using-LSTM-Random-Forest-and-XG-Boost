import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import time
from pytrends.request import TrendReq

# Optional libraries
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHAV_AVAILABLE = True
except Exception:
    ALPHAV_AVAILABLE = False

try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except Exception:
    KITE_AVAILABLE = False


@st.cache_data
def load_data_yfinance(symbol, years):
    """Fetches primary stock data from Yahoo Finance."""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        st.warning(f"yfinance failed to get data for {symbol}.")
        return None
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data
def load_macro_data(years):
    """Fetches VIX, S&P 500, and 10Y Treasury data."""
    end = datetime.today()
    start = end - timedelta(days=years * 365)

    macro_tickers = {
        'VIX_Close': '^VIX',
        'SPY_Close': 'SPY',
        'TNX_Close': '^TNX'
    }

    df_macro = pd.DataFrame()

    for col_name, ticker in macro_tickers.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not data.empty:
                data = data[['Close']].rename(columns={'Close': col_name})
                if df_macro.empty:
                    df_macro = data
                else:
                    # Use join to handle different date indices properly
                    df_macro = df_macro.join(data, how='outer')
        except Exception as e:
            st.warning(f"Could not download macro data for {ticker}: {e}")

    if df_macro.empty:
        st.warning("Failed to fetch any macro data.")
        return None

    df_macro = df_macro.reset_index()
    df_macro['Date'] = pd.to_datetime(df_macro['Date']).dt.normalize()

    # Forward-fill macro data for weekends/holidays
    df_macro = df_macro.ffill()

    # Flatten MultiIndex columns if they exist
    if isinstance(df_macro.columns, pd.MultiIndex):
        df_macro.columns = df_macro.columns.get_level_values(0)

    # Ensure 'Date' column is present after potential flattening
    if 'Date' not in df_macro.columns:
         if 'index' in df_macro.columns: # Sometimes reset_index keeps 'index'
              df_macro.rename(columns={'index':'Date'}, inplace=True)
         elif df_macro.index.name == 'Date':
              df_macro = df_macro.reset_index()
         else:
             st.error("Critical Error: 'Date' column lost in load_macro_data after flattening.")
             return None

    # Ensure Date column is the correct type
    df_macro['Date'] = pd.to_datetime(df_macro['Date']).dt.normalize()
    return df_macro

@st.cache_data(ttl=86400) # Cache for 1 day
def load_google_trends(ticker, years):
    """Fetches Google Trends data, cached daily."""
    pytrends = TrendReq(hl='en-US', tz=360)
    end = datetime.today()
    start = end - timedelta(days=years * 365)

    # Format for pytrends
    timeframe = f'{start.strftime("%Y-%m-%d")} {end.strftime("%Y-%m-%d")}'

    try:
        pytrends.build_payload([ticker], cat=0, timeframe=timeframe, geo='', gprop='')
        df_trends = pytrends.interest_over_time()

        if df_trends.empty or ticker not in df_trends.columns:
            st.warning(f"No Google Trends data found for '{ticker}'.")
            return None

        df_trends = df_trends.reset_index().rename(columns={'date': 'Date', ticker: 'Google_Trend'})
        df_trends = df_trends[['Date', 'Google_Trend']]
        df_trends['Date'] = pd.to_datetime(df_trends['Date']).dt.normalize()

        return df_trends

    except Exception as e:
        # Handle 429 Too Many Requests or other errors
        st.warning(f"Google Trends fetch failed: {e}. May be rate-limited.")
        return None


@st.cache_data
def load_data_alpha_vantage(symbol, api_key, years):
    """Fetches data from Alpha Vantage as a fallback."""
    if not ALPHAV_AVAILABLE:
        st.error("alpha_vantage library not found.")
        return None
    if not api_key:
        st.warning("Alpha Vantage API key not provided.")
        return None

    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta = ts.get_daily_adjusted(symbol, outputsize='full')
        data = data.reset_index().rename(columns={'date': 'Date'})
        data['Date'] = pd.to_datetime(data['Date'])
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        data = data[(data['Date'] >= start) & (data['Date'] <= end)]
        remap = {}
        for col in data.columns:
            if col.lower().startswith('1. open'): remap[col] = 'Open'
            if col.lower().startswith('2. high'): remap[col] = 'High'
            if col.lower().startswith('3. low'): remap[col] = 'Low'
            if col.lower().startswith('4. close'): remap[col] = 'Close'
            # Alpha Vantage adjusted close is often preferred
            if col.lower().startswith('5. adjusted close'): remap[col] = 'Adj Close'
            if col.lower().startswith('6. volume'): remap[col] = 'Volume'
        data.rename(columns=remap, inplace=True)
        # Prioritize Adj Close if available, otherwise use Close
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        elif 'Close' not in data.columns:
            st.warning("Alpha Vantage data missing 'Close' or 'Adj Close'.")
            return None

        available = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
        return data[available]
    except Exception as e:
        st.warning(f"Alpha Vantage failed: {e}")
        return None


def load_data_kite(symbol, years, api_key=None, api_secret=None):
    """Placeholder for KiteConnect data fetching."""
    if not KITE_AVAILABLE:
        st.error("kiteconnect library not found.")
        return None
    st.info("Kite integration is a placeholder and requires manual token exchange & implementation.")
    return None