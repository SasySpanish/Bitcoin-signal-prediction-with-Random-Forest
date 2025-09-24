# Project Results

Below are the key results obtained from the Bitcoin price/volume dataset and the Random Forest model trained for predicting buy/sell signals.

---

## Performance Metrics

- **Accuracy**: ~ 94% overall  
- **Precision (Buy = 1)**: ~ 99%  
- **Precision (Sell = 0)**: ~ 74%  
- **Other metrics**:
  - Recall, F1-score, support for each class were also calculated (Buy vs Sell)  
  - Confusion matrix: high number of true positives and true negatives, but with more false positives/false negatives in the `Sell` class compared to `Buy`  

---

## Observations

- The model is **very good at identifying “Buy” signals**, almost perfect in precision.  
- It is less accurate for “Sell” signals — more misclassifications there.  
- The class distribution is relatively balanced, but possibly some imbalance contributed to the performance asymmetry.  
- Using the training period (**Dataset Part 3**) and then applying on **Part 4** allowed testing on **future unseen data**, which gives credibility to the predictive power.  

---

## Limitations & Notes

- The high precision for “Buy” might indicate overfitting toward that class or bias due to thresholds.  
- The model’s performance on “Sell” is weaker, which might reduce profitability if sell signals are necessary for trading strategy.  
- **Time-dependence / temporal drift**: since financial data changes over time, performance might degrade if market regime shifts.  

---

## Implications

- The model seems suitable for scenarios where false “Buy” signals are expensive: very few of those occur.  
- For practical trading strategies, one may want to improve detection of “Sell” signals (e.g. by adjusting decision threshold, using cost-sensitive learning, or adding more features).  
- Because the evaluation was done on time-forward data (train on Part 3, test on Part 4), success suggests potential for real use, though backtested performance over different market regimes should be checked.  

- The current model is of low complexity and cannot be used reliably for real trading decisions.

- Misclassifications in the “Sell” class indicate that the model struggles to generalize across all signals.

### Future improvements could include:

- Using more complex models (e.g., ensemble methods, deep learning)

- Feature engineering to capture additional market information

- Addressing any subtle class imbalance or temporal dependencies

- Backtesting on larger and more varied datasets
