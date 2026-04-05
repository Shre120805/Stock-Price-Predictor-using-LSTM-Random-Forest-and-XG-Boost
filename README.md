Stock Price Prediction using Ensemble Learning
Overview

This project implements a stock price prediction system using an ensemble of LSTM, Random Forest, and XGBoost models. It combines deep learning and traditional machine learning techniques to improve prediction accuracy and robustness.

Objective

The objective of this project is to model stock price movements by capturing both temporal dependencies and non-linear feature relationships using a hybrid approach.

Methodology
Data Preprocessing
Collection of historical stock price data (Open, High, Low, Close, Volume)
Handling missing values
Normalization of data for LSTM
Creation of time-series sequences using a sliding window approach
Models Used
LSTM (Long Short-Term Memory): Captures sequential and temporal dependencies in stock price data
Random Forest Regressor: Handles non-linear relationships and reduces overfitting through bagging
XGBoost Regressor: Provides high performance using gradient boosting with regularization
Ensemble Technique

Predictions from all models are combined using an averaging method to improve generalization and reduce variance.

Evaluation Metrics
Root Mean Square Error (RMSE)
Mean Absolute Error (MAE)
R-squared Score (R²)
Results

The ensemble model demonstrates improved performance compared to individual models by leveraging the strengths of each approach.

Limitations
Does not account for external factors such as news sentiment or macroeconomic indicators
Stock market behavior is inherently volatile and difficult to predict with high accuracy
Future Work
Incorporate technical indicators such as RSI and MACD
Integrate sentiment analysis from financial news
Deploy the model using web frameworks such as Flask or Streamlit
Implement real-time prediction using live data APIs
Technologies Used
Python
NumPy, Pandas
Scikit-learn
TensorFlow / Keras
XGBoost
Matplotlib / Seaborn
Usage
Clone the repository
Install dependencies
Run the preprocessing and training scripts
Evaluate model performance
