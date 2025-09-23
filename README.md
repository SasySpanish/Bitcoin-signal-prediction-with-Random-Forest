# Bitcoin-signal-prediction-with-Random-Forest
## Project Overview

This project applies machine learning techniques to financial time series, with a focus on predicting buy/sell signals for Bitcoin.
The core idea is to use a binary classification model (Decision Trees and Random Forest) to identify trading signals derived from the relationship between short-term and long-term exponential weighted moving averages (EWMAs).

## Dataset

Source: Bitstamp

Time Interval: Minute-level data

Period: Dec 2011 – Mar 2021 (≈ 9.24 years)

Total Records: 4,857,377 (risk of overfitting due to high granularity)

Main Variables: Timestamp, Close

## Methodology
Signal Definition

- Golden Cross (Buy = 1):
EWM[Short Term Price] > EWM[Long Term Price]

- Death Cross (Sell = 0):
EWM[Short Term Price] < EWM[Long Term Price]

Short-term EWM: 50 days (72,000 minutes)

Long-term EWM: 200 days (288,000 minutes)

## Models Used

- Decision Trees

- Random Forest (ensemble learning)

## Workflow

- Data preprocessing and feature engineering

- Label creation (buy/sell signals)

- Train-test split (80% / 20%)

Model training and evaluation

Cross-validation (4-fold)

## Libraries

- Pandas → data manipulation

- NumPy → numerical operations

- Matplotlib → visualization

- Scikit-learn → ML models and evaluation

## Model Evaluation
Metrics

- Accuracy Score → fraction of correct predictions

- Precision → TP / (TP + FP)

- Recall → TP / (TP + FN)

- F1-score → harmonic mean of precision and recall

- Confusion Matrix → true/false positives and negatives

- AUC-ROC Curve → trade-off between TPR and FPR

## Results

- Overall Accuracy: 94%

- Precision for Sell Signals (0): 74%

- Precision for Buy Signals (1): 99%

# Conclusions

- The Random Forest model effectively predicts Bitcoin buy/sell signals.

- High accuracy achieved, but sell signals are harder to classify compared to buy signals.

- EWMA-based features provide a robust framework for detecting trend reversals.

### Future Improvements

- Test on other cryptocurrencies or financial assets

- Hyperparameter tuning for Random Forest

- Experiment with deep learning (LSTM, GRU) for time series forecasting
