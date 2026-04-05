import os
import time
from datetime import datetime
import sqlite3
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Import from our custom modules
from config import DEVICE, FEATURE_CANDIDATES
from data_loader import (
    load_data_yfinance, load_data_alpha_vantage, load_data_kite,
    load_macro_data,
    load_google_trends,
    ALPHAV_AVAILABLE, KITE_AVAILABLE
)
from sentiment_utils import get_finbert_sentiment, get_reddit_sentiment
from feature_utils import add_indicators
# Import prepare_and_train from ml_pipeline
from ml_pipeline import prepare_and_train, OPTUNA_AVAILABLE, HeteroLSTM
from trading_logic import predict_tomorrow, generate_suggestion_and_quantity

# --- Database Setup ---
def init_database():
    conn = sqlite3.connect('trades.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        timestamp TEXT, ticker TEXT, action TEXT,
        quantity INTEGER, price REAL, reason TEXT
    )''')
    conn.commit()
    conn.close()

def log_trade_to_db(ticker, action, quantity, price, reason):
    conn = sqlite3.connect('trades.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, ticker, action, quantity, price, reason))
    conn.commit()
    conn.close()

# --- Session State ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'cash': 10000.0, 'shares': 0, 'entry_price': 0.0
    }

# --- Streamlit page config ---
st.set_page_config(page_title="Hybrid Stock Predictor", layout="wide")
st.title("Hybrid Stock Predictor")
init_database()

# --- Sidebar ---
st.sidebar.header("Configuration")
TICKER = st.sidebar.text_input("Ticker (e.g., AAPL)", "AAPL").upper()
YEARS = st.sidebar.slider("Years of history", 1, 12, 5)
LOOKBACK = st.sidebar.slider("LSTM lookback (days)", 20, 120, 60)

st.sidebar.header("Tuning & Data")
USE_OPTUNA = st.sidebar.checkbox("Use Optuna for tuning (optional)", value=True if OPTUNA_AVAILABLE else False, disabled=not OPTUNA_AVAILABLE)
OPTUNA_TRIALS = st.sidebar.slider("Optuna trials (per model)", 10, 60, 20)
ENABLE_ALPHA = st.sidebar.checkbox("Enable Alpha Vantage fallback", value=False, disabled=not ALPHAV_AVAILABLE)
ALPHA_KEY = st.sidebar.text_input("Alpha Vantage API Key", value="", type="password")
ENABLE_KITE = st.sidebar.checkbox("Enable Zerodha/Kite placeholder", value=False, disabled=not KITE_AVAILABLE)
KITE_API_KEY = st.sidebar.text_input("Kite API Key (optional)", value="", type="password")
KITE_API_SECRET = st.sidebar.text_input("Kite API Secret (optional)", value="", type="password")

st.sidebar.subheader("Reddit API (Optional)")
st.sidebar.info("Required for r/wallstreetbets sentiment. [How to get keys](https://www.reddit.com/prefs/apps)")
REDDIT_CLIENT_ID = st.sidebar.text_input("Reddit Client ID", value="", type="password")
REDDIT_CLIENT_SECRET = st.sidebar.text_input("Reddit Client Secret", value="", type="password")
REDDIT_USER_AGENT = st.sidebar.text_input("Reddit User Agent (e.g., 'MyStockApp v1.0')", value="")

st.sidebar.header("Trading Configuration")
# Use 100.0 (float) for min_value
PORTFOLIO_VALUE = st.sidebar.number_input("Portfolio Cash ($)", min_value=100.0, value=st.session_state.portfolio['cash'])
RISK_PER_TRADE = st.sidebar.slider("Risk % per Trade", 0.5, 5.0, 1.0) / 100.0
ENABLE_TRADING = st.sidebar.checkbox("Enable Trading Panel", value=True)
st.session_state.portfolio['cash'] = PORTFOLIO_VALUE

st.sidebar.header("Run")
RELEASE_MODE = st.sidebar.checkbox("Release mode (show disclaimer)", value=True)
RUN_BUTTON = st.sidebar.button("Fetch, Train & Visualize")
st.sidebar.write(f"Compute device: {DEVICE}")

# --- Plotting Utilities ---
# Updated plot_data_distributions function
def plot_data_distributions(df: pd.DataFrame, features: list):
    """Plots histograms for key features with added validation."""
    st.subheader("Data & Feature Distributions")

    available_features = [f for f in features + ['Close'] if f in df.columns]
    if not available_features:
        st.warning("No features found to plot distributions.")
        return

    default_ix = 0
    if 'Close' in available_features:
        try:
            default_ix = available_features.index('Close')
        except ValueError:
            default_ix = 0

    feature = st.selectbox(
        "Select feature to view distribution",
        available_features,
        index=default_ix,
        key="dist_select" # Add key
    )

    if feature in df.columns:
        if df[feature].isnull().all():
            st.warning(f"Feature '{feature}' contains only null values. Cannot plot histogram.")
            return
        try:
            data_to_plot = pd.to_numeric(df[feature], errors='coerce').copy()
            data_to_plot.replace([np.inf, -np.inf], np.nan, inplace=True)
            data_to_plot.dropna(inplace=True)
        except Exception as e:
            st.warning(f"Could not process data for feature '{feature}': {e}")
            return
        if data_to_plot.empty:
            st.warning(f"No valid numeric data to plot for '{feature}' after cleaning.")
            return

        try:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data_to_plot, name=feature, marker_color='cyan'))
            fig.update_layout(
                title=f"Distribution of {feature}",
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating histogram for '{feature}': {e}")
    else:
        st.warning(f"Selected feature '{feature}' unexpectedly not found in DataFrame columns during plot attempt.")


def plot_model_vs_actual(dates, y_true, y_pred, model_name, ci=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, name='Actual', line=dict(color='white', width=3)))
    colors = {'LSTM': 'cyan', 'XGBoost': 'magenta', 'RandomForest': 'yellow', 'Stacked Ensemble': 'lime'}
    fig.add_trace(
        go.Scatter(x=dates, y=y_pred, name=model_name, line=dict(color=colors.get(model_name, 'orange'), width=2)))
    # Add length check for CI arrays
    if ci is not None and len(ci) == 2 and ci[0] is not None and ci[1] is not None and len(ci[0]) == len(dates) and len(ci[1]) == len(dates):
        lower_ci, upper_ci = ci
        fig.add_trace(go.Scatter(
            x=dates, y=upper_ci, name='Upper CI', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=lower_ci, name='Lower CI', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(0,255,255,0.12)', showlegend=False
        ))
    fig.update_layout(
        title=f'{model_name} Predictions vs Actual',
        xaxis_title='Date', yaxis_title='Price',
        plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df: pd.DataFrame, features: list):
    st.subheader("Feature Correlation Heatmap")
    available_features = [f for f in features + ['Close'] if f in df.columns]
    if not available_features:
        st.warning("No features found for correlation heatmap.")
        return
    # Exclude non-numeric columns before calculating correlation
    numeric_df = df[available_features].select_dtypes(include=np.number)
    if numeric_df.empty:
         st.warning("No numeric features found for correlation heatmap.")
         return
    corr_df = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
        colorscale='Viridis', zmin=-1, zmax=1, hoverongaps=False
    ))
    fig.update_layout(
        title="Feature Correlation (vs. 'Close' price and each other)",
        plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_advanced_residuals(residuals: np.ndarray):
    st.subheader("Advanced Residual Analysis (Stacked Ensemble)")
    if residuals is None or len(residuals) == 0 or np.isnan(residuals).all():
         st.warning("No valid residuals available for advanced analysis.")
         return

    col1, col2 = st.columns(2)
    with col1:
        st.write("Q-Q Plot (Normality Check)")
        try:
            res_clean = residuals[~np.isnan(residuals)]
            if len(res_clean) < 2:
                 st.warning("Not enough valid residuals for Q-Q plot.")
            else:
                 qq_data = stats.probplot(res_clean, dist="norm", fit=False)
                 fig_qq = go.Figure()
                 fig_qq.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Data Points', marker_color='cyan'))
                 fig_qq.add_trace(go.Scatter(x=qq_data[1][0], y=qq_data[1][1], mode='lines', name='Normal Line', line=dict(color='white', dash='dash')))
                 fig_qq.update_layout(
                     title="Q-Q Plot: Residuals vs. Normal Distribution",
                     xaxis_title="Theoretical Quantiles (Normal)", yaxis_title="Sample Quantiles (Residuals)",
                     plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white')
                 )
                 st.plotly_chart(fig_qq, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate Q-Q plot: {e}")
    with col2:
        st.write("Autocorrelation (ACF) Plot")
        try:
            res_clean = residuals[~np.isnan(residuals)]
            if len(res_clean) < 2:
                 st.warning("Not enough valid residuals for ACF plot.")
            else:
                 plt.style.use('dark_background')
                 fig_acf, ax = plt.subplots(figsize=(10, 4), facecolor='black')
                 max_lags = min(40, len(res_clean) - 1)
                 if max_lags > 0:
                      plot_acf(res_clean, ax=ax, lags=max_lags, color='cyan', vlines_kwargs={"colors": "cyan"})
                      ax.set_title("Residual Autocorrelation", color='white')
                      ax.set_facecolor('black')
                      ax.tick_params(colors='white', which='both')
                      ax.xaxis.label.set_color('white')
                      ax.yaxis.label.set_color('white')
                      for spine in ax.spines.values():
                          spine.set_edgecolor('white')
                      fig_acf.tight_layout()
                      st.pyplot(fig_acf)
                 else:
                      st.warning("Not enough data points for ACF plot lags.")
                 plt.style.use('default')
        except Exception as e:
            st.warning(f"Could not generate ACF plot: {e}")


# --- Main Application Logic ---
if RUN_BUTTON:
    # --- Data Loading and Merging ---
    df_merged_clean = None # Initialize
    try:
        st.info("Fetching base data (yfinance primary)...")
        df = None
        if ENABLE_KITE and KITE_API_KEY and KITE_API_SECRET:
            df = load_data_kite(TICKER, YEARS, KITE_API_KEY, KITE_API_SECRET)
        if df is None and ENABLE_ALPHA and ALPHAV_AVAILABLE and ALPHA_KEY:
            st.info("Attempting Alpha Vantage fallback.")
            df = load_data_alpha_vantage(TICKER, ALPHA_KEY, YEARS)
        if df is None:
            st.info("Using yfinance as primary data source.")
            df = load_data_yfinance(TICKER, YEARS)

        if df is None or df.empty:
            st.error("Failed to fetch base stock data. Check ticker and API keys.")
            st.stop()
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()

        st.info("Fetching macro-economic data (VIX, SPY, TNX)...")
        df_macro = load_macro_data(YEARS)
        if df_macro is not None:
            df = pd.merge(df, df_macro, on='Date', how='left')
            df = df.ffill()
            st.success("Macro data integrated.")
        else:
            st.warning("Could not load macro data. Proceeding without it.")

        st.info(f"Fetching Google Trends data for {TICKER}...")
        df_trends = load_google_trends(TICKER, YEARS)
        if df_trends is not None:
            df = pd.merge(df, df_trends, on='Date', how='left')
            df['Google_Trend'] = df['Google_Trend'].ffill().fillna(0.0)
            st.success("Google Trends data integrated.")
        else:
            st.warning("No Google Trends data found. Proceeding without it.")
            df['Google_Trend'] = 0.0

        st.info(f"Fetching news and running FinBERT for {TICKER}...")
        st.warning("**Limitation Notice:** `yfinance` only provides *recent* news.")
        sentiment_df = get_finbert_sentiment(TICKER)
        if not sentiment_df.empty:
            df = pd.merge(df, sentiment_df, on='Date', how='left')
            df['FinBERT_Score'] = df['FinBERT_Score'].fillna(0.0)
            st.success("FinBERT sentiment integrated.")
        else:
            st.warning("No FinBERT sentiment data found. Proceeding without it.")
            df['FinBERT_Score'] = 0.0

        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT:
            st.info(f"Fetching Reddit sentiment for {TICKER}...")
            st.warning("**Limitation Notice:** Reddit API only provides *recent* posts.")
            reddit_df = get_reddit_sentiment(TICKER, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
            if not reddit_df.empty:
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize().dt.tz_localize(None)
                reddit_df['Date'] = pd.to_datetime(reddit_df['Date']).dt.normalize().dt.tz_localize(None)
                df = pd.merge(df, reddit_df, on='Date', how='left')
                df['Reddit_Score'] = df['Reddit_Score'].fillna(0.0)
                st.success("Reddit sentiment integrated.")
            else:
                st.warning("No Reddit sentiment data found. Proceeding without it.")
                df['Reddit_Score'] = 0.0
        else:
            st.warning("Reddit API keys not provided. Skipping Reddit sentiment.")
            df['Reddit_Score'] = 0.0

        st.success(f"Fetched and merged all data for {len(df)} rows.")
        st.dataframe(df.tail(10))
        df_merged_clean = df.copy() # Keep clean copy before training modifies it

    except Exception as e:
        st.error(f"Error during data fetching/merging: {e}")
        st.exception(e)
        st.stop()

    # --- Training Pipeline (Potentially cached) ---
    results = None
    if df_merged_clean is not None and not df_merged_clean.empty: # Check if data loading succeeded
        try:
            with st.spinner("Training models (this may be cached)..."):
                results = prepare_and_train( # Pass Ticker first for cache invalidation
                    TICKER,
                    df_merged_clean,
                    seq_len=LOOKBACK,
                    use_optuna=USE_OPTUNA,
                    optuna_trials=OPTUNA_TRIALS
                )
            results['ticker'] = TICKER
            results['years'] = YEARS
            results['lookback'] = LOOKBACK
            results['risk_per_trade'] = RISK_PER_TRADE

        except ValueError as ve:
             st.error(f"Training Data Error: {ve}")
             st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during training: {e}")
            st.exception(e)
            st.stop()
    else:
        st.error("Cannot proceed to training, data loading failed or resulted in empty DataFrame.")
        st.stop()


    # --- Calculate Indicators for Plotting AFTER getting results ---
    st.header("Data Exploration & Results")
    df_with_indicators = pd.DataFrame() # Initialize empty
    try:
        with st.spinner("Calculating indicators for plots..."):
             if not df_merged_clean.empty:
                 df_with_indicators = add_indicators(df_merged_clean.copy())
             else:
                 st.warning("Input data for indicator calculation is empty.")
    except Exception as e:
        st.warning(f"Could not calculate indicators for plots: {e}")

    # --- Now call the plots using the calculated df_with_indicators ---
    if not df_with_indicators.empty:
        plot_data_distributions(df_with_indicators, FEATURE_CANDIDATES)
        plot_correlation_heatmap(df_with_indicators, FEATURE_CANDIDATES)
    else:
        st.warning("Skipping exploration plots due to indicator calculation failure or empty data.")


    # --- Show Training Results ---
    st.header("Model Performance on Test Set")
    if results and 'metrics' in results:
        metrics_df = pd.DataFrame({k: v for k, v in results['metrics'].items() if v is not None}).T
        st.dataframe(metrics_df.round(4))
    else:
        st.warning("Metrics not available in training results.")

    st.subheader("LSTM Training History")
    hist = results.get('train_hist', {})
    if hist and hist.get('train_loss') and hist.get('val_rmse'):
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(y=hist['train_loss'], name='Train Loss'))
        fig_hist.add_trace(go.Scatter(y=hist['val_rmse'], name='Val RMSE'))
        fig_hist.update_layout(title='LSTM Training History', xaxis_title='Epoch', yaxis_title='Loss / RMSE')
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
         st.info("LSTM training history not available.")

    st.subheader("Price & Indicators (sample)")
    if not df_with_indicators.empty:
         df_plot = df_with_indicators.tail(250)
         if 'Date' in df_plot.columns and 'Close' in df_plot.columns:
             figp = go.Figure()
             figp.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Close'], name='Close'))
             if 'SMA_20' in df_plot.columns: figp.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['SMA_20'], name='SMA20'))
             if 'EWMA_vol' in df_plot.columns and not df_plot['Close'].empty:
                 figp.add_trace(
                     go.Scatter(x=df_plot['Date'], y=df_plot['EWMA_vol'] * df_plot['Close'].mean(), name='EWMA_vol (scaled)')
                 )
             figp.update_layout(title=f"{TICKER} Price & Indicators")
             st.plotly_chart(figp, use_container_width=True)
         else:
              st.warning("Required columns ('Date', 'Close') missing for Price/Indicator plot.")
    else:
        st.warning("Skipping Price/Indicator plot.")


    if results and 'dates_test' in results and 'y_test' in results and 'preds' in results:
        dates = pd.to_datetime(results['dates_test'])
        y_test = results.get('y_test', np.array([])) # Use .get for safety
        preds = results.get('preds', {})
        ci = results.get('ci', {})
        min_len = len(dates)

        # Check if dates is empty or y_test length mismatch
        if min_len == 0 or len(y_test) != min_len:
            st.warning(f"Length mismatch or zero length between dates ({len(dates)}) and y_test ({len(y_test)}). Skipping prediction plots.")
            pred_stack = None
        else:
            pred_stack = None # Initialize
            # Plot only if prediction exists and has correct length
            if 'lstm' in preds and len(preds['lstm']) == min_len:
                lstm_ci_data = (ci.get('lstm_lower'), ci.get('lstm_upper'))
                plot_model_vs_actual(dates, y_test, preds['lstm'], 'LSTM', ci=lstm_ci_data)
            if 'xgb' in preds and len(preds['xgb']) == min_len:
                plot_model_vs_actual(dates, y_test, preds['xgb'], 'XGBoost')
            if 'rf' in preds and len(preds['rf']) == min_len:
                plot_model_vs_actual(dates, y_test, preds['rf'], 'RandomForest')
            if 'stack' in preds and len(preds['stack']) == min_len:
                stack_ci_data = (ci.get('lower'), ci.get('upper'))
                pred_stack = preds['stack']
                plot_model_vs_actual(dates, y_test, pred_stack, 'Stacked Ensemble', ci=stack_ci_data)

                # Plot residuals only if stack prediction was successful
                st.subheader("Residuals (Stacked Ensemble)")
                resid = y_test - pred_stack
                fig_r = go.Figure();
                fig_r.add_trace(go.Scatter(y=resid, mode='markers', marker_color='cyan'))
                fig_r.update_layout(title='Residuals', xaxis_title='Index', yaxis_title='Residual', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
                st.plotly_chart(fig_r, use_container_width=True)
                plot_advanced_residuals(resid)
            else:
                st.warning("Stacked prediction not available or length mismatch.")
    else:
        st.warning("Prediction results missing or incomplete. Cannot display prediction plots.")
        pred_stack = None

    st.subheader("Feature Importances (LGBM Meta-Learner)")
    if results and 'models' in results and 'meta' in results['models'] and results['models']['meta'] is not None and hasattr(results['models']['meta'], 'feature_importances_'):
        try:
            meta_imp = results['models']['meta'].feature_importances_
            meta_features = ['LSTM', 'XGBoost', 'RandomForest', 'RecentVol']
            if len(meta_imp) == len(meta_features):
                df_imp = pd.DataFrame({'feature': meta_features, 'importance': meta_imp})
                df_imp = df_imp.sort_values(by='importance', ascending=False)
                st.bar_chart(df_imp.set_index('feature'))
            else:
                st.info(f"Meta-model importance length mismatch (Expected {len(meta_features)}, Got {len(meta_imp)}).")
        except Exception as e:
            st.info(f"Could not compute Meta-model feature importances: {e}")
    else:
        st.info("Meta-model not trained or importances not available.")

    st.subheader("Download Predictions & Metrics")
    if results and 'dates_test' in results and 'y_test' in results and pred_stack is not None and 'ci' in results:
         dates = results['dates_test']
         y_test = results.get('y_test', np.array([]))
         lower = results['ci'].get('lower')
         upper = results['ci'].get('upper')
         min_len = min(len(dates), len(y_test), len(pred_stack), len(lower) if lower is not None and len(lower)>0 else float('inf'), len(upper) if upper is not None and len(upper)>0 else float('inf'))

         if min_len > 0 and len(dates) == min_len and len(y_test) == min_len and len(pred_stack) == min_len:
              out_data = {'Date': dates[:min_len], 'Actual': y_test[:min_len], 'Predicted_Stack': pred_stack[:min_len]}
              if lower is not None and len(lower) == min_len: out_data['CI_Lower'] = lower[:min_len]
              if upper is not None and len(upper) == min_len: out_data['CI_Upper'] = upper[:min_len]
              out_df = pd.DataFrame(out_data)
              st.download_button("Download predictions CSV", data=out_df.to_csv(index=False).encode('utf-8'), file_name=f"{TICKER}_predictions.csv", key="pred_csv")
         else:
              st.warning("Could not create predictions CSV due to length mismatch.")
    else:
        st.warning("Prediction data incomplete, cannot generate CSV.")
    if 'metrics_df' in locals():
         st.download_button("Download metrics CSV", data=metrics_df.reset_index().to_csv(index=False).encode('utf-8'), file_name=f"{TICKER}_metrics.csv", key="metrics_csv")

    st.success("Training and Historical Prediction Complete.")

    try:
        if results:
             with open('results.pkl', 'wb') as f:
                 pickle.dump(results, f)
             st.success("Trained model (results.pkl) saved successfully for backtesting.")
        else:
            st.error("Training results are missing, cannot save model.")
    except Exception as e:
        st.error(f"Failed to save model: {e}")

    if ENABLE_TRADING:
        st.header("📈 Live Trading Panel")
        st.subheader("Current Portfolio")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cash", f"${st.session_state.portfolio['cash']:,.2f}")
        col2.metric(f"Shares Held ({TICKER})", f"{st.session_state.portfolio['shares']}")
        col3.metric("Entry Price", f"${st.session_state.portfolio['entry_price']:,.2f}")

        if results and 'last_sequence' in results and isinstance(results['last_sequence'], np.ndarray) and results['last_sequence'].size > 0:
            try:
                with st.spinner("Generating next-day prediction..."):
                    models = results.get('models')
                    last_sequence = results.get('last_sequence')
                    feature_cols = results.get('feature_cols')
                    current_price = results.get('last_price')
                    last_std_dev = results.get('last_std_dev')

                    if not all([models, isinstance(last_sequence, np.ndarray), feature_cols, isinstance(current_price, (int, float)), isinstance(last_std_dev, float)]):
                         raise ValueError("Required data missing from training results for prediction.")
                    pred_price, lower_ci, upper_ci, std_dev = predict_tomorrow(models, last_sequence, feature_cols, last_std_dev)

                suggestion, quantity, reason = generate_suggestion_and_quantity(current_price, pred_price, lower_ci, upper_ci, std_dev, st.session_state.portfolio['cash'], RISK_PER_TRADE, st.session_state.portfolio['shares'])

                st.subheader(f"Signal for {TICKER}: {suggestion}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${current_price:,.2f}")
                col2.metric("Predicted Price", f"${pred_price:,.2f}")
                col3.metric("Suggested Quantity", f"{quantity} shares")
                confidence = np.clip( (0.1 - (std_dev/current_price)) / 0.1, 0.0, 1.0)*100 if current_price != 0 else 0
                col4.metric("Model Confidence", f"{confidence:.0f}%")
                st.info(f"**Reasoning:** {reason}")
                st.warning(f"**95% Confidence Interval:** ${lower_ci:,.2f} - ${upper_ci:.2f}")

                st.subheader("Trade Execution (Placeholder)")
                if (suggestion.startswith("BUY") or suggestion.startswith("SELL")) and quantity > 0:
                    if st.button(f"Execute {suggestion} order for {quantity} shares", key="trade_button"):
                        if ENABLE_KITE and KITE_API_KEY:
                            st.info("DUMMY: Attempting trade via Kite API...")
                        try:
                            log_trade_to_db(TICKER, suggestion, quantity, current_price, reason)
                            if suggestion == "BUY":
                                cost = quantity * current_price
                                st.session_state.portfolio['cash'] -= cost
                                current_total_value = st.session_state.portfolio['entry_price'] * st.session_state.portfolio['shares']
                                new_total_shares = st.session_state.portfolio['shares'] + quantity
                                if new_total_shares > 0:
                                     st.session_state.portfolio['entry_price'] = (current_total_value + cost) / new_total_shares
                                st.session_state.portfolio['shares'] += quantity
                            elif suggestion == "SELL":
                                proceeds = quantity * current_price
                                st.session_state.portfolio['cash'] += proceeds
                                st.session_state.portfolio['shares'] -= quantity
                                if st.session_state.portfolio['shares'] == 0:
                                    st.session_state.portfolio['entry_price'] = 0.0
                            st.success(f"DUMMY: Trade Executed & Logged. Portfolio updated.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Trade logging/state update failed: {e}")
                else:
                    st.write("No trade suggested or button not clicked.")

                st.subheader("Recent Trade Log")
                try:
                    conn = sqlite3.connect('trades.db')
                    log_df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10", conn)
                    conn.close()
                    st.dataframe(log_df)
                except Exception as e:
                    st.error(f"Could not read trade log: {e}")
            except ValueError as ve:
                 st.error(f"Error generating trading suggestion: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred generating trading suggestion: {e}")
                st.exception(e)
        else:
             st.warning("Could not generate trading suggestion: Invalid 'last_sequence' or missing 'results' from training.")

# --- Correct Indentation: This 'else' aligns with 'if RUN_BUTTON:' ---
else:
    st.info("Configure settings in the sidebar and press 'Fetch, Train & Visualize' to begin.")