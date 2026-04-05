import torch
import lightgbm as lgb

# --- Global Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Feature Engineering Config ---
# List of all feature candidates. The pipeline will use whichever of these it finds.
FEATURE_CANDIDATES = [
    # Technical Indicators
    'SMA_20', 'EMA_20', 'Momentum', 'RealizedVol_14', 'EWMA_vol',
    'ATR_14', 'HL_range', 'MACD', 'Signal_Line', 'RSI',
    'SMA_50', 'BB_width', 'OBV',

    # Macro-Economic Features
    'VIX_Close', 'SPY_Close', 'TNX_Close',

    # Alternative Data Features
    'FinBERT_Score',
    'Google_Trend',
    'Reddit_Score'
]

# --- Model Parameter Config ---
XGB_PARAMS_DEFAULT = {
    'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'verbosity': 0
}

RF_PARAMS_DEFAULT = {
    'n_estimators': 150, 'max_depth': 8,
    'random_state': 42, 'n_jobs': -1
}

LGBM_PARAMS_DEFAULT = {
    'objective': 'regression_l1', 'metric': 'rmse',
    'n_estimators': 100, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'num_leaves': 16, 'verbose': -1, 'n_jobs': -1,
    'seed': 42, 'boosting_type': 'gbdt',
}

LSTM_CFG_DEFAULT = {
    'hidden_size': 96, 'num_layers': 3, 'dropout': 0.25,
    'lr': 5e-4, 'epochs': 100, 'batch_size': 32
}