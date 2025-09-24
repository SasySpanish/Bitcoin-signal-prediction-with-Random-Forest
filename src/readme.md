## Source Code (`src/`)

The `src/` folder contains the Python scripts used for data preparation, model training, and signal prediction.  
Scripts are organized as follows:

- **General Workflow Script** (ML_script_general.py)
  - A clean path that includes:
    - Dataset creation and preprocessing  
    - Signal construction (Golden Cross / Death Cross)  
    - Random Forest training
    - Model validation
    - Model application and signal prediction  

- **Project-Focused Script** (model3_github.py)
  - Scripts dedicated to the **third** and **fourth parts of the dataset**:  
    - **Part 3:** Training the Random Forest model  
    - **Part 4:** Applying the trained model to unseen data and generating predictions  

---

### Workflow Recap
1. Create dataset and build trading signals  
2. Train Random Forest algorithm on **Dataset Part 3**  
3. Apply the trained model on **Dataset Part 4**  
4. Predict buy/sell signals for future data  

---

This structure allows both:
- **Reproducibility** → anyone can follow the general script to replicate the full pipeline  
- **Modularity** → the dataset-specific scripts highlight the exact process used for training and prediction in this project  
