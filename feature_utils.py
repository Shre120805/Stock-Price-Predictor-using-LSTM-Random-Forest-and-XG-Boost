import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

# Utilities & robust column cleaning

def clean_close_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # If MultiIndex columns, reduce to top level names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # If Close isn't present try some common alternatives
    if 'Close' not in df.columns:
        # try 'Adj Close', 'adjusted close', 'close' lowercase, or first column that looks like close
        alt = None
        for candidate in ['Adj Close', 'Adj_Close', 'adjusted close', 'adjusted_close', 'close']:
            if candidate in df.columns:
                alt = candidate;
                break
        if alt:
            df['Close'] = df[alt]
        else:
            # attempt to find a single column with float-like values and call it Close (last resort)
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) >= 1:
                # choose the column named 'close' ignoring case if exists
                for c in numeric_cols:
                    if str(c).lower() == 'close':
                        df['Close'] = df[c];
                        break
                else:
                    # fallback: assume last numeric column is Close
                    df['Close'] = df[numeric_cols[-1]]
            else:
                raise ValueError("Could not find a 'Close' column in data.")
    # Now ensure it's a Series
    close_col = df['Close']
    if isinstance(close_col, pd.DataFrame):
        # If it's a single-column DataFrame, squeeze to Series
        if close_col.shape[1] == 1:
            close_col = close_col.iloc[:, 0]
        else:
            # Try to select the most-likely close column by name
            if 'Close' in close_col.columns:
                close_col = close_col['Close']
            else:
                # squeeze by converting to numpy 1D if possible
                try:
                    close_col = pd.Series(close_col.values.squeeze())
                except Exception:
                    raise TypeError("df['Close'] is not a 1D column or cannot be squeezed.")
    # Convert to numeric safely
    df['Close'] = pd.to_numeric(close_col, errors='coerce')
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    return df


def compute_RSI(series, period=14):
    s = pd.Series(series).astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def compute_MACD(series, short=12, long=26, signal=9):
    s = pd.Series(series).astype(float)
    ema_short = s.ewm(span=short, adjust=False).mean()
    ema_long = s.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, signal_line.values


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = clean_close_column(df)
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close']).diff()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    # Bollinger Bands
    df['BB_std'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_high'] = df['SMA_20'] + (df['BB_std'] * 2)
    df['BB_low'] = df['SMA_20'] - (df['BB_std'] * 2)
    df['BB_width'] = (df['BB_high'] - df['BB_low']) / (df['SMA_20'] + 1e-12)
    if 'Volume' in df.columns and pd.api.types.is_numeric_dtype(df['Volume']):
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['RealizedVol_14'] = df['Return'].rolling(14, min_periods=1).std() * np.sqrt(252)
    df['EWMA_vol'] = df['Return'].ewm(span=21, adjust=False).std() * np.sqrt(252)
    if {'High', 'Low'}.issubset(df.columns):
        # Ensure High/Low are numeric before calculation
        if pd.api.types.is_numeric_dtype(df['High']) and pd.api.types.is_numeric_dtype(df['Low']):
            df['HL_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-12)
            df['ATR_14'] = df['HL_range'].rolling(14, min_periods=1).mean()
        else:
             df['HL_range'] = np.nan
             df['ATR_14'] = np.nan
    macd, sig = compute_MACD(df['Close'])
    df['MACD'] = macd
    df['Signal_Line'] = sig
    df['RSI'] = compute_RSI(df['Close'])

    # Use ffill() and bfill() instead of fillna(method=...)
    df = df.ffill().bfill()

    return df


def make_sequences(df: pd.DataFrame, feature_cols, seq_len):
    df2 = df.copy().reset_index(drop=True)
    # Drop rows where *any* feature or 'Close' is NaN before creating sequences
    df2 = df2.dropna(subset=feature_cols + ['Close']).reset_index(drop=True)
    X, y, dates = [], [], []
    # Ensure seq_len is valid
    safe_seq_len = max(1, seq_len)
    if len(df2) < safe_seq_len:
         return np.array([]), np.array([]), np.array([]) # Return empty if not enough data

    for i in range(safe_seq_len, len(df2)):
        X.append(df2.loc[i - safe_seq_len:i - 1, feature_cols].values)
        y.append(df2.loc[i, 'Close'])
        # Use index if 'Date' column is missing after potential drops
        dates.append(df2.loc[i, 'Date'] if 'Date' in df2.columns else df2.index[i])
    return np.array(X), np.array(y), np.array(dates)


def rmse(a, b):
    # Ensure inputs are numpy arrays and handle potential NaNs
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    mask = ~np.isnan(a_np) & ~np.isnan(b_np)
    if np.sum(mask) == 0:
        return np.nan # Or float('inf')
    return math.sqrt(mean_squared_error(a_np[mask], b_np[mask]))