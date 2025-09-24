# Bitcoin-signal-prediction-with-Random-Forest
## Project Overview

This project applies machine learning techniques to financial time series, with a focus on predicting buy/sell signals for Bitcoin.
The core idea is to use a binary classification model (Decision Trees and Random Forest) to identify trading signals derived from the relationship between short-term and long-term exponential weighted moving averages (EWMAs).

## Dataset

Source: Bitstamp

Time Interval: Minute-level data

Period: Dec 2011 – Mar 2021 (≈ 9.24 years)

Total Records: 4,857,377 (risk of overfitting due to high granularity) divided by 4 = 1,214,344 records each. 

Main Variables: Timestamp, Close

## Process Overview  

To evaluate the model on realistic, unseen data, the dataset was divided into **four sequential parts**:  

1. **Dataset Split**  
   - Due to file size limitations and to simulate a real forecasting scenario, the full dataset was split into four chronological parts.  

2. **Training Phase**  
   - The **third dataset** was used to train the **Random Forest model**.  
   - This set covers the period immediately before the test set, ensuring that the model learns from past information.  

3. **Testing / Prediction Phase**  
   - The trained model was then applied to the **fourth dataset**, which contains the **immediately following data**.  
   - Predictions were generated on this unseen portion, simulating a real-world scenario where models are trained on past data and then used to predict future market signals.  

4. **Goal**  
   - This setup ensures that evaluation is not biased by data leakage and that the model’s performance reflects its ability to generalize to **future Bitcoin price movements**.  


## Methodology
Signal Definition

- Golden Cross (Buy = 1):
EWM[Short Term Price] > EWM[Long Term Price]
A bullish technical pattern that occurs when the short-term moving average rises above the long-term moving average, suggesting upward momentum and a potential price increase.

- Death Cross (Sell = 0):
EWM[Short Term Price] < EWM[Long Term Price]
A bearish technical pattern that occurs when the short-term moving average falls below the long-term moving average, indicating downward momentum and a potential price decline.

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
