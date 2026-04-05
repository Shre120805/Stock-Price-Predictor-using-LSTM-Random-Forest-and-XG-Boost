import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from config import DEVICE

def predict_tomorrow(
    models: Dict[str, Any],
    last_sequence: np.ndarray,
    feature_cols: List[str],
    last_std_dev: float
) -> Tuple[float, float, float, float]:
    """
    Predicts the next day's price using the trained models.
    Returns (prediction, lower_ci, upper_ci, std_dev).
    """
    # Ensure models exist
    if not all(k in models for k in ['lstm', 'xgb', 'rf', 'meta']):
        raise ValueError("One or more required models are missing.")

    # Reshape for models: (1, seq_len, n_features)
    seq_len = last_sequence.shape[0]
    n_features = last_sequence.shape[1]
    if n_features != len(feature_cols):
        raise ValueError(f"Feature dimension mismatch: sequence has {n_features}, expected {len(feature_cols)}.")

    X_future_seq = torch.tensor(last_sequence.reshape(1, seq_len, n_features), dtype=torch.float32).to(DEVICE)
    # Reshape for flat models: (1, seq_len * n_features)
    X_future_flat = last_sequence.reshape(1, -1)

    # Get predictions from base models
    lstm_model = models['lstm']
    lstm_model.eval() # Ensure LSTM is in eval mode
    with torch.no_grad():
        mu, logvar = lstm_model(X_future_seq)
        pred_lstm = mu.cpu().numpy()[0]
        # We use the std_dev passed from the training pipeline
        std_dev_tomorrow = last_std_dev

    pred_xgb = models['xgb'].predict(X_future_flat)[0]
    pred_rf = models['rf'].predict(X_future_flat)[0]

    # Get recent volatility (same logic as in training)
    if 'EWMA_vol' in feature_cols:
        vol_idx = feature_cols.index('EWMA_vol')
    elif 'RealizedVol_14' in feature_cols:
        vol_idx = feature_cols.index('RealizedVol_14')
    else:
        vol_idx = -1 # Fallback

    # Use the last value of the last step in the unscaled sequence (careful with indexing)
    # This assumes vol feature is present and scaled; index might need adjustment
    recent_vol_norm = last_sequence[-1, vol_idx] if vol_idx != -1 else 0.0

    # Meta-learner prediction
    meta_model = models['meta']
    if meta_model is None:
         # Fallback if meta model failed: average base predictions
         final_pred = np.mean([pred_lstm, pred_xgb, pred_rf])
         print("Warning: Meta model not available, using average of base models.")
    else:
        meta_X = np.array([[pred_lstm, pred_xgb, pred_rf, recent_vol_norm]])
        final_pred = meta_model.predict(meta_X)[0]


    # Estimate CI for tomorrow
    z = 1.96

    # Use the stacking prediction (or average) as the final "mean"
    final_lower = final_pred - z * std_dev_tomorrow
    final_upper = final_pred + z * std_dev_tomorrow

    # Convert to float to ensure type consistency
    return float(final_pred), float(final_lower), float(final_upper), float(std_dev_tomorrow)


def generate_suggestion_and_quantity(
    current_price: float,
    pred_price: float,
    lower_ci: float,
    upper_ci: float,
    std_dev: float, # For dynamic sizing
    portfolio_value: float, # Represents available cash for new trades
    risk_per_trade: float,
    current_shares: int # For statefulness
) -> Tuple[str, int, str]:
    """
    Generates a Buy/Sell/Hold signal and calculates trade quantity
    based on model confidence and current portfolio state.
    """
    suggestion = "HOLD"
    quantity = 0
    reason = ""

    # Basic check for valid prices
    if current_price <= 0:
         return "HOLD", 0, "Invalid current price."

    # --- 1. Dynamic Sizing based on Confidence ---
    relative_std = std_dev / current_price if current_price > 0 else float('inf')
    confidence_factor = np.clip( (0.1 - relative_std) / 0.1, 0.0, 1.0)

    if confidence_factor == 0.0:
        return "HOLD", 0, f"Model uncertainty ({relative_std*100:.1f}%) is too high. No signal."

    # --- 2. Risk Management (base quantity) ---
    cash_to_risk = portfolio_value * risk_per_trade

    # --- 3. Stateful Strategy Logic ---

    # --- A) LOGIC FOR OPENING A NEW LONG POSITION ---
    if current_shares == 0:
        buy_threshold = current_price * 1.01 # Example: 1% expected gain
        # Condition: Prediction is above threshold AND lower CI is above current price (high confidence buy)
        if (pred_price > buy_threshold) and (lower_ci > current_price):
            suggestion = "BUY"
            # Use the lower CI as a theoretical stop-loss for risk calculation
            stop_loss_price = lower_ci
            risk_per_share = current_price - stop_loss_price

            if risk_per_share <= 0:
                risk_per_share = current_price * 0.01 # Fallback: Assume 1% risk if CI is too tight/below current

            if risk_per_share > 0 and cash_to_risk > 0:
                 base_quantity = int(cash_to_risk / risk_per_share)
                 # Adjust quantity by model confidence
                 quantity = int(base_quantity * confidence_factor)
                 # Ensure we don't try to buy more than we can afford
                 max_affordable_qty = int(portfolio_value / current_price)
                 quantity = min(quantity, max_affordable_qty)

            if quantity > 0:
                 reason = (f"BUY Signal (New Position). Pred: {pred_price:.2f}, "
                           f"Lower CI: {lower_ci:.2f}. Confidence: {confidence_factor*100:.0f}%. "
                           f"Risk/Share: ${risk_per_share:.2f}.")
            else:
                 suggestion = "HOLD" # Not enough cash or risk too high
                 reason = (f"Potential BUY signal, but quantity is zero (Affordability/Risk). Pred: {pred_price:.2f}.")


    # --- B) LOGIC FOR CLOSING AN EXISTING LONG POSITION ---
    elif current_shares > 0:
        sell_threshold = current_price * 0.99 # Example: 1% expected drop
        # Condition: Prediction is below threshold AND upper CI is below current price (high confidence sell)
        if (pred_price < sell_threshold) and (upper_ci < current_price):
            suggestion = "SELL"
            quantity = current_shares # Sell the entire position
            reason = (f"SELL Signal (Close Long). Pred: {pred_price:.2f}, "
                      f"Upper CI: {upper_ci:.2f}. Model suggests downturn.")
        else:
            # Default HOLD reason when already long
            reason = (f"HOLD Signal. Currently long {current_shares} shares. "
                      f"No strong sell signal detected. Pred: {pred_price:.2f}, CI: ({lower_ci:.2f}-{upper_ci:.2f}).")

    # (Optional: Add logic for short selling if current_shares == 0 and sell conditions met)

    if quantity == 0 and suggestion == "HOLD":
        if reason == "": # Add default hold reason if none provided yet
            reason = (f"No strong BUY/SELL signal based on prediction ({pred_price:.2f}) "
                      f"and CI ({lower_ci:.2f} - {upper_ci:.2f}).")

    # Final safety check
    if quantity < 0: quantity = 0

    return suggestion, quantity, reason