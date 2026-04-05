import os
import math
import time
import inspect
import pickle
import numpy as np
import pandas as pd
import streamlit as st # Need st for the cache decorator

# ML libraries
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

# PyTorch for heteroscedastic LSTM
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Optional libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# Import from our own modules
from config import (
    DEVICE, FEATURE_CANDIDATES,
    XGB_PARAMS_DEFAULT, RF_PARAMS_DEFAULT,
    LGBM_PARAMS_DEFAULT, LSTM_CFG_DEFAULT
)
from feature_utils import add_indicators, make_sequences, rmse


def xgb_fit_compat(model, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
    """
    Try modern fit signature; fallback to basic fit if not supported.
    """
    try:
        sig = inspect.signature(model.fit)
        params = sig.parameters
        accepts_eval = 'eval_set' in params
        accepts_es = 'early_stopping_rounds' in params
    except Exception:
        accepts_eval = accepts_es = True
    try:
        if accepts_eval and accepts_es and eval_set is not None and early_stopping_rounds is not None:
            model.fit(X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        elif accepts_eval and eval_set is not None:
            model.fit(X, y, eval_set=eval_set, verbose=verbose)
        else:
            model.fit(X, y)
    except TypeError:
        # Fallback for older XGBoost versions or specific fit issues
        model.fit(X, y)
    except Exception as e:
        # Catch other potential fit errors
        st.warning(f"XGBoost fit failed with standard methods, falling back to basic fit. Error: {e}")
        model.fit(X, y) # Final fallback
    return model


# Heteroscedastic LSTM (PyTorch)

class HeteroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        # Ensure hidden_size is appropriate
        actual_hidden_size = hidden_size if hidden_size > 0 else 128
        self.lstm = nn.LSTM(input_size, actual_hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(actual_hidden_size, 1)
        self.fc_logvar = nn.Linear(actual_hidden_size, 1)
        self.hidden_size = actual_hidden_size # Store for reference
        self.num_layers = num_layers

    def forward(self, x):
        device = x.device
        # Use stored attributes for consistency
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]
        mu = self.fc_mu(last).squeeze(-1) # Use squeeze(-1) for robustness
        logvar = self.fc_logvar(last).squeeze(-1)
        return mu, logvar


def hetero_loss(mu, logvar, y):
    var = torch.exp(logvar) + 1e-6
    loss = 0.5 * ((y - mu) ** 2 / var + logvar)
    return loss.mean()


# -------------------------
# Main pipeline: prepare, tune, train, predict
# --- ADD CACHE DECORATOR ---
@st.cache_data(show_spinner=False) # Spinner handled in app.py
# --- ADD ticker AS THE FIRST ARGUMENT ---
def prepare_and_train(ticker, _df_raw, seq_len=60, use_optuna=False, optuna_trials=20):
    """
    Prepares data, tunes hyperparameters (optional), trains models,
    and returns predictions and trained components.
    Input DataFrame _df_raw is copied to avoid modifying the cached input.
    Ticker is added to ensure cache invalidation on ticker change.
    """
    df_raw = _df_raw.copy() # Work on a copy

    # 1) build indicators & clean
    df = add_indicators(df_raw)

    # Use feature set from config
    feature_cols = [f for f in FEATURE_CANDIDATES if f in df.columns]

    if not feature_cols:
        raise ValueError("No features available after processing. Check data and feature candidates.")

    if 'Close' not in df.columns:
        raise ValueError("No Close column after cleaning.")

    # scale features
    scaler_rob = RobustScaler()
    scaler_std = StandardScaler()

    # Ensure all feature columns are numeric BEFORE scaling
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where *any* feature is missing AFTER coercing
    df = df.dropna(subset=feature_cols + ['Close'])

    if df.empty:
         raise ValueError("DataFrame empty after adding indicators and dropping NaNs.")

    feats = df[feature_cols].values
    feats_rob = scaler_rob.fit_transform(feats)
    feats_std = scaler_std.fit_transform(feats_rob)
    # Use .loc to assign back safely
    df.loc[:, feature_cols] = feats_std

    # sequences
    X_seq, y_seq, dates = make_sequences(df, feature_cols, seq_len)
    if len(X_seq) < 40:
        raise ValueError(f"Not enough samples ({len(X_seq)}) after sequence creation. Increase data or reduce lookback.")

    n = len(X_seq)
    test_size = max(int(0.15 * n), 10)
    val_size = test_size
    n_test = test_size;
    n_val = val_size;
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough train samples after splitting.")

    X_train = X_seq[:n_train];
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val];
    y_val = y_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:];
    y_test = y_seq[n_train + n_val:]
    dates_test = dates[n_train + n_val:]

    # flatten for tree models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # defaults from config
    xgb_params = XGB_PARAMS_DEFAULT.copy()
    rf_params = RF_PARAMS_DEFAULT.copy()
    lstm_cfg = LSTM_CFG_DEFAULT.copy()

    # Optional tuning (XGB & RF)
    if use_optuna and OPTUNA_AVAILABLE:
        st.info("Running Optuna tuning (XGBoost & RandomForest)...") # Show this info in Streamlit

        # Tune XGB
        def xgb_obj(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                # --- FIX DEPRECATED ---
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                # --- END FIX ---
                'random_state': 42, 'verbosity': 0
            }
            # Combine train+val for cross-validation within Optuna objective
            X_t = np.vstack([X_train_flat, X_val_flat]);
            y_t = np.concatenate([y_train, y_val])
            n_t = len(X_t)
            rmses = []
            # Simplified validation: Use a single split within the objective
            tr_end = int(n_t * 0.8)
            val_start = tr_end
            if n_t - tr_end < 5: # Ensure validation set is large enough
                return 1e9 # Return a large error if not enough data
            mdl = XGBRegressor(**params)
            try:
                # Use fit directly, handle potential errors
                mdl.fit(X_t[:tr_end], y_t[:tr_end],
                        eval_set=[(X_t[val_start:], y_t[val_start:])],
                        early_stopping_rounds=20, verbose=False)
            except Exception as e:
                 print(f"Optuna XGB fit failed: {e}")
                 return 1e9 # Return large error if fit fails

            pred = mdl.predict(X_t[val_start:])
            # Handle potential NaNs in predictions
            valid_preds = ~np.isnan(pred)
            if np.sum(valid_preds) == 0:
                return 1e9 # No valid predictions
            rmse_val = math.sqrt(mean_squared_error(y_t[val_start:][valid_preds], pred[valid_preds]))
            return rmse_val if not np.isnan(rmse_val) else 1e9


        study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        # Use with statement for spinner
        with st.spinner("Optuna tuning XGBoost..."):
             study_xgb.optimize(xgb_obj, n_trials=optuna_trials, show_progress_bar=False, n_jobs=1) # Use n_jobs=1 for simplicity
        # Check if study has trials before accessing best_params
        if study_xgb.trials:
             xgb_params.update(study_xgb.best_params)
        else:
             st.warning("Optuna study for XGBoost completed without any successful trials.")


        # RF randomized search (Keep as is, less prone to issues than Optuna CV)
        param_dist = {'n_estimators': list(range(50, 401, 50)), 'max_depth': [None] + list(range(4, 21)),
                      'min_samples_split': [2, 3, 5], 'min_samples_leaf': [1, 2, 4]}
        best_rf = None;
        best_score = float('inf')
        X_t_rf = np.vstack([X_train_flat, X_val_flat]);
        y_t_rf = np.concatenate([y_train, y_val])
        n_t_rf = len(X_t_rf)
        tr_end_rf = int(n_t_rf * 0.8)
        val_start_rf = tr_end_rf

        if n_t_rf - tr_end_rf >= 5: # Check validation size for RF as well
            with st.spinner("Optuna tuning RandomForest..."):
                for i, params in enumerate(
                        ParameterSampler(param_dist, n_iter=max(12, int(optuna_trials / 2)), random_state=42)):
                    mdl = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
                    mdl.fit(X_t_rf[:tr_end_rf], y_t_rf[:tr_end_rf])
                    pred = mdl.predict(X_t_rf[val_start_rf:])
                    score = math.sqrt(mean_squared_error(y_t_rf[val_start_rf:], pred))

                    if score < best_score:
                        best_score = score;
                        best_rf = params
            if best_rf: rf_params.update(best_rf)
        else:
            st.warning("Skipping RandomForest tuning due to insufficient validation data.")


    # Train final models on train+val
    X_final = np.vstack([X_train_flat, X_val_flat]);
    y_final = np.concatenate([y_train, y_val])

    xgb_model = XGBRegressor(**xgb_params)
    try:
        # Use simpler fit compatible call here for final model
        xgb_model.fit(X_final, y_final)
    except Exception as e:
        st.error(f"Final XGBoost training failed: {e}")
        raise

    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_final, y_final)

    # Train heteroscedastic LSTM (PyTorch)
    X_lstm_final = np.vstack([X_train, X_val])
    y_lstm_final = np.concatenate([y_train, y_val])
    ds = TensorDataset(torch.tensor(X_lstm_final, dtype=torch.float32), torch.tensor(y_lstm_final, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=lstm_cfg['batch_size'], shuffle=True)

    # Ensure valid LSTM parameters
    lstm_hidden_size = lstm_cfg.get('hidden_size', 96)
    lstm_num_layers = lstm_cfg.get('num_layers', 3)
    lstm_dropout = lstm_cfg.get('dropout', 0.25)
    if lstm_hidden_size <= 0 or lstm_num_layers <= 0:
        st.warning("Invalid LSTM parameters, using defaults.")
        lstm_hidden_size = 96
        lstm_num_layers = 3

    lstm = HeteroLSTM(input_size=X_lstm_final.shape[2], hidden_size=lstm_hidden_size,
                      num_layers=lstm_num_layers, dropout=lstm_dropout).to(DEVICE)
    opt = torch.optim.AdamW(lstm.parameters(), lr=lstm_cfg['lr'], weight_decay=1e-4)
    train_losses = [];
    val_rmse_hist = []
    epochs = lstm_cfg['epochs']
    # Use st.status for progress reporting within the cached function
    status = st.status("Training LSTM...", expanded=False)
    lstm_pbar = st.progress(0)
    nan_loss_detected = False # Flag to track NaN loss
    for ep in range(epochs):
        lstm.train()
        total_loss = 0.0;
        cnt = 0
        for bx, by in loader:
            bx = bx.to(DEVICE);
            by = by.to(DEVICE)
            opt.zero_grad()
            mu, logvar = lstm(bx)
            loss = hetero_loss(mu, logvar, by)
            # Check for NaN loss
            if torch.isnan(loss):
                 status.update(label=f"NaN loss detected at epoch {ep+1}. Stopping LSTM training.", state="error")
                 nan_loss_detected = True
                 break # Stop training if loss becomes NaN
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            opt.step()
            total_loss += loss.item();
            cnt += 1
        # Break outer loop if NaN loss detected
        if nan_loss_detected:
             break
        train_losses.append(total_loss / max(1, cnt))
        # quick val
        lstm.eval()
        with torch.no_grad():
            if X_test.shape[0] > 0: # Only validate if test set exists
                 mu_test, logvar_test = lstm(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
                 # Ensure y_test and mu_test are numpy arrays for metrics
                 y_test_np = np.asarray(y_test)
                 mu_test_np = mu_test.cpu().numpy()
                 # Ensure shapes match before calculating RMSE
                 if y_test_np.shape == mu_test_np.shape:
                      val_rmse = rmse(y_test_np, mu_test_np) # Use updated rmse func
                      val_rmse_hist.append(val_rmse)
                 else:
                      # Handle potential shape mismatch (e.g., if last batch was smaller)
                      min_len = min(len(y_test_np), len(mu_test_np))
                      if min_len > 0:
                           val_rmse = rmse(y_test_np[:min_len], mu_test_np[:min_len])
                           val_rmse_hist.append(val_rmse)
                      else:
                           val_rmse_hist.append(float('inf')) # Indicate error or no overlap
            else:
                 val_rmse_hist.append(float('inf')) # Indicate no validation possible

        status.update(label=f"Training LSTM... Epoch {ep + 1}/{epochs}", state="running")
        lstm_pbar.progress(min((ep + 1) / epochs, 1.0))
    # Close the status widget
    status.update(label="LSTM Training Complete." if not nan_loss_detected else "LSTM Training Halted (NaN Loss).",
                  state="complete" if not nan_loss_detected else "error",
                  expanded=False)


    # test predictions
    preds_lstm = np.array([])
    var_lstm = np.array([])
    if X_test.shape[0] > 0 and not nan_loss_detected: # Only predict if LSTM trained properly
        lstm.eval()
        with torch.no_grad():
            mu_test, logvar_test = lstm(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
            preds_lstm = mu_test.cpu().numpy()
            var_lstm = np.exp(logvar_test.cpu().numpy())
    elif nan_loss_detected:
        st.error("LSTM training failed (NaN Loss), cannot generate LSTM predictions.")
    else:
        st.warning("Test set is empty, cannot generate test predictions.")


    preds_xgb = xgb_model.predict(X_test_flat) if X_test_flat.shape[0] > 0 else np.array([])
    preds_rf = rf_model.predict(X_test_flat) if X_test_flat.shape[0] > 0 else np.array([])

    # meta-learner: needs test predictions
    preds_stack = np.array([])
    meta = None
    y_test_aligned = y_test # Initialize aligned y_test
    # Ensure all base predictions have the same length as y_test
    min_pred_len = min(len(preds_lstm) if len(preds_lstm)>0 else float('inf'),
                       len(preds_xgb) if len(preds_xgb)>0 else float('inf'),
                       len(preds_rf) if len(preds_rf)>0 else float('inf'))

    if min_pred_len == len(y_test) and min_pred_len > 0:
        # Align predictions before stacking
        preds_lstm_aligned = preds_lstm[:min_pred_len]
        preds_xgb_aligned = preds_xgb[:min_pred_len]
        preds_rf_aligned = preds_rf[:min_pred_len]
        y_test_aligned = y_test[:min_pred_len]
        X_test_aligned = X_test[:min_pred_len] # For vol feature

        last_step_feats = X_test_aligned[:, -1, :]
        if 'EWMA_vol' in feature_cols:
            vol_idx = feature_cols.index('EWMA_vol')
            recent_vol = last_step_feats[:, vol_idx]
        elif 'RealizedVol_14' in feature_cols:
            vol_idx = feature_cols.index('RealizedVol_14')
            recent_vol = last_step_feats[:, vol_idx]
        else:
            recent_vol = np.std(y_test_aligned) * np.ones(len(y_test_aligned))

        # Check for zero std deviation before normalizing
        vol_std = recent_vol.std()
        if vol_std > 1e-12:
            recent_vol_norm = (recent_vol - recent_vol.mean()) / vol_std
        else:
            recent_vol_norm = np.zeros_like(recent_vol) # Avoid division by zero


        meta_X = np.vstack([preds_lstm_aligned, preds_xgb_aligned, preds_rf_aligned, recent_vol_norm]).T

        meta = LGBMRegressor(**LGBM_PARAMS_DEFAULT)
        meta.fit(meta_X, y_test_aligned)
        preds_stack = meta.predict(meta_X)
    else:
         st.warning(f"Base model prediction lengths mismatch or empty (LSTM:{len(preds_lstm)}, XGB:{len(preds_xgb)}, RF:{len(preds_rf)}, y_test:{len(y_test)}). Skipping meta-learner.")


    # volatility-adjusted dynamic weighting
    preds_adj = np.array([])
    weights_adj = np.array([1/3, 1/3, 1/3]) # Default weights
    if 'y_test_aligned' in locals() and len(y_test_aligned) > 20: # Use aligned y_test length
        def short_rmse(true, pred, window=20):
            w = min(window, len(true))
            if w == 0: return float('inf')
            # Ensure predictions align with true values for RMSE calculation
            pred_aligned = pred[-w:]
            true_aligned = true[-w:]
            if len(pred_aligned) != len(true_aligned): # Should not happen if aligned before
                 return float('inf')
            return rmse(true_aligned, pred_aligned) # Use updated rmse func


        rmse_l = short_rmse(y_test_aligned, preds_lstm_aligned) if 'preds_lstm_aligned' in locals() and len(preds_lstm_aligned) > 0 else float('inf')
        rmse_x = short_rmse(y_test_aligned, preds_xgb_aligned) if 'preds_xgb_aligned' in locals() and len(preds_xgb_aligned) > 0 else float('inf')
        rmse_r = short_rmse(y_test_aligned, preds_rf_aligned) if 'preds_rf_aligned' in locals() and len(preds_rf_aligned) > 0 else float('inf')
        inv = np.array([1.0 / (rmse_l + 1e-9), 1.0 / (rmse_x + 1e-9), 1.0 / (rmse_r + 1e-9)])

        if np.any(np.isinf(inv)) or np.any(np.isnan(inv)):
             st.warning("Infinite or NaN RMSE detected in weighting, using default weights.")
        else:
             inv_sum = inv.sum()
             if inv_sum > 1e-9:
                 inv /= inv_sum
                 if 'recent_vol_norm' in locals():
                     vol_factor = np.clip(1.0 / (1.0 + np.abs(recent_vol_norm).mean()), 0.6, 1.4)
                     weights_adj = inv * np.array([vol_factor, 1.0, 1.0])
                     weights_adj_sum = weights_adj.sum()
                     if weights_adj_sum > 1e-9:
                         weights_adj /= weights_adj_sum
                     else: weights_adj = np.array([1/3, 1/3, 1/3])
                 else: weights_adj = inv

                 # Ensure base predictions exist before combining
                 if 'preds_lstm_aligned' in locals() and 'preds_xgb_aligned' in locals() and 'preds_rf_aligned' in locals():
                      preds_adj = weights_adj[0] * preds_lstm_aligned + weights_adj[1] * preds_xgb_aligned + weights_adj[2] * preds_rf_aligned
             else:
                 st.warning("Sum of inverse RMSEs too small, using default weights.")

    # CI calculations (only if meta-learner ran and preds_stack generated)
    final_lower, final_upper = np.array([]), np.array([])
    lstm_lower, lstm_upper = np.array([]), np.array([])
    lower_boot, upper_boot = np.array([]), np.array([])

    if meta is not None and len(preds_stack) > 0:
         # Align var_lstm with preds_stack for combined CI
        var_lstm_aligned = var_lstm[:len(preds_stack)] if len(var_lstm) >= len(preds_stack) else np.zeros_like(preds_stack)

        # In-sample predictions for bootstrapping (needs careful alignment)
        # Simplified: Use test residuals if available and lengths match
        residuals_test = y_test_aligned - preds_stack if 'y_test_aligned' in locals() and len(y_test_aligned) == len(preds_stack) else np.array([])

        # Bootstrap CI (using test residuals if available)
        if len(residuals_test) > 0:
             rng = np.random.default_rng(42)
             n_boot = 300
             boot_preds = []
             for i in range(n_boot):
                 samp = rng.choice(residuals_test, size=len(preds_stack), replace=True)
                 boot_preds.append(preds_stack + samp)
             boot = np.array(boot_preds)
             lower_boot = np.percentile(boot, 2.5, axis=0)
             upper_boot = np.percentile(boot, 97.5, axis=0)

        # Hetero LSTM CI (aligned)
        if 'preds_lstm_aligned' in locals() and len(preds_lstm_aligned) > 0:
             z = 1.96
             lstm_var_safe = np.maximum(var_lstm_aligned, 0)
             lstm_lower = preds_lstm_aligned - z * np.sqrt(lstm_var_safe)
             lstm_upper = preds_lstm_aligned + z * np.sqrt(lstm_var_safe)

             # Combined CI (if bootstrap ran)
             if len(lower_boot) > 0:
                 boot_std = np.std(boot, axis=0)
                 combined_std = np.sqrt(lstm_var_safe + boot_std ** 2)
                 final_lower = preds_stack - z * combined_std
                 final_upper = preds_stack + z * combined_std


    # metrics - Use aligned data
    metrics = {}
    def calculate_metrics(name, y_true, y_pred):
        # Ensure lengths match and data is valid before calculating
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true) == len(y_pred) and len(y_true) > 0:
             mask = ~np.isnan(y_pred) & ~np.isnan(y_true) # Also check y_true for safety
             if np.sum(mask) > 0:
                 y_true_valid = y_true[mask]
                 y_pred_valid = y_pred[mask]
                 # Add check for constant predictions which lead to R2 issues
                 if np.std(y_true_valid) < 1e-9 or np.std(y_pred_valid) < 1e-9:
                      r2 = np.nan # Cannot calculate R2 if true or pred is constant
                 else:
                      try:
                          r2 = r2_score(y_true_valid, y_pred_valid)
                      except ValueError: # Handle cases like single sample
                          r2 = np.nan
                 try:
                      rmse_val = rmse(y_true_valid, y_pred_valid)
                      mae_val = mean_absolute_error(y_true_valid, y_pred_valid)
                 except ValueError:
                      rmse_val, mae_val = np.nan, np.nan

                 return {'RMSE': rmse_val, 'MAE': mae_val, 'R2': r2}
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

    # Use locals() to safely access aligned predictions if they exist
    preds_l = locals().get('preds_lstm_aligned', np.array([]))
    preds_x = locals().get('preds_xgb_aligned', np.array([]))
    preds_r = locals().get('preds_rf_aligned', np.array([]))
    preds_s = locals().get('preds_stack', np.array([]))
    preds_a = locals().get('preds_adj', np.array([]))
    y_test_a = locals().get('y_test_aligned', np.array([]))


    metrics['LSTM'] = calculate_metrics('LSTM', y_test_a, preds_l)
    metrics['XGBoost'] = calculate_metrics('XGBoost', y_test_a, preds_x)
    metrics['RandomForest'] = calculate_metrics('RandomForest', y_test_a, preds_r)
    metrics['Stacking'] = calculate_metrics('Stacking', y_test_a, preds_s)
    metrics['AdjEnsemble'] = calculate_metrics('AdjEnsemble', y_test_a, preds_a)


    # Get last sequence for tomorrow's prediction
    last_std_dev = 1.0 # Default uncertainty
    last_seq_scaled = np.array([])
    if len(df) >= seq_len: # Ensure enough data in the original df
        last_sequence_features = df[feature_cols].values[-seq_len:]
        if last_sequence_features.shape[0] == seq_len:
             last_seq_scaled = scaler_std.transform(
                 scaler_rob.transform(last_sequence_features)
             )
             # Get uncertainty for the last sequence (only if LSTM trained ok)
             if not nan_loss_detected:
                 lstm.eval()
                 with torch.no_grad():
                     _, last_logvar = lstm(torch.tensor(last_seq_scaled.reshape(1, seq_len, -1), dtype=torch.float32).to(DEVICE))
                     last_logvar_np = last_logvar.cpu().numpy()[0]
                     if not np.isnan(last_logvar_np):
                         last_std_dev = float(np.sqrt(np.exp(last_logvar_np)))
                     else:
                         st.warning("NaN logvar detected for last sequence uncertainty.")
             else:
                  st.warning("LSTM failed, using default uncertainty for prediction.")
        else:
            st.warning("Could not extract last sequence for prediction (shape mismatch).")
    else:
        st.warning("Not enough data in the final DataFrame to extract last sequence.")


    # Ensure all returned arrays have consistent lengths, aligned with y_test_aligned
    final_dates_test = dates_test[:len(y_test_a)] if 'y_test_a' in locals() and len(dates_test) >= len(y_test_a) else np.array([])


    results = {
        'feature_cols': feature_cols,
        'scalers': {'rob': scaler_rob, 'std': scaler_std},
        'models': {'xgb': xgb_model, 'rf': rf_model, 'lstm': lstm, 'meta': meta},
        'preds': {'lstm': preds_l, 'xgb': preds_x, 'rf': preds_r, 'stack': preds_s, 'adj': preds_a},
        'y_test': y_test_a, # Return aligned y_test
        'dates_test': final_dates_test, # Return aligned dates
        'ci': {'lower': final_lower if len(final_lower) == len(final_dates_test) else np.array([]),
               'upper': final_upper if len(final_upper) == len(final_dates_test) else np.array([]),
               'lstm_lower': lstm_lower if len(lstm_lower) == len(final_dates_test) else np.array([]),
               'lstm_upper': lstm_upper if len(lstm_upper) == len(final_dates_test) else np.array([]),
               'boot_lower': lower_boot if len(lower_boot) == len(final_dates_test) else np.array([]),
               'boot_upper': upper_boot if len(upper_boot) == len(final_dates_test) else np.array([])},
        'metrics': metrics,
        'train_hist': {'train_loss': train_losses, 'val_rmse': val_rmse_hist},
        'weights_adj': weights_adj,
        # Items for prediction
        'last_sequence': last_seq_scaled,
        'last_price': df_raw['Close'].iloc[-1] if not df_raw.empty else 0,
        'last_std_dev': last_std_dev
    }
    return results