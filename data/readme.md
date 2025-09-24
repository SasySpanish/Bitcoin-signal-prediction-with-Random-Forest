# Dataset Information  

Due to GitHub’s file size upload limitations, the full dataset was divided into **four parts**.  

- The dataset is organized into separate folders:  
  - **`data/`** → contains the third part of the dataset  
  - **`data2/`** → contains the fourth part of the dataset  

---

## Dataset Split  

- **Part 3 (Folder: `data/`)**  
  - Period: **Wed Aug 17 2016 11:53:00 – Sat Dec 08 2018 17:56:00**  
  - Purpose: **Used to train the Random Forest algorithm**  

- **Part 4 (Folder: `data2/`)**  
  - Period: **Sat Dec 08 2018 17:57:00 – Wed Mar 31 2021 02:00:00** (immediately following Part 3) 
  - Purpose: **Used to test the model and generate predictions**  

---

## ⚙️ Usage in the Project  

1. **Training**  
   - The **third part of the dataset** is used to train the **Random Forest model**.  

2. **Testing / Prediction**  
   - The **trained model** is then applied to the **fourth part of the dataset**.  
   - This setup ensures that predictions are made on **new, unseen data**, preserving a realistic time-series forecasting scenario.  

---

## Summary  

- Dataset split due to GitHub limitations  
- **Part 3 → Training**  
- **Part 4 → Prediction / Testing**  
- Structure allows evaluation of the model’s ability to generalize to future, unseen data  

---

## 🔗 Workflow Diagram  

```mermaid
flowchart LR
    A[Dataset Part 3<br/>Training Data] --> B[Train Random Forest Model]
    B --> C[Dataset Part 4<br/>Testing Data]
    C --> D[Predictions on Buy/Sell Signals]

