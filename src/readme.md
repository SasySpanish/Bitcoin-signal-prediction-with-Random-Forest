## Source Code (`src/`)

The `src/` folder contains the Python scripts used for data preparation, model training, and signal prediction.  
Scripts are organized as follows:

- **General Workflow Script**
  - A complete pipeline that includes:
    - Dataset creation and preprocessing  
    - Signal construction (Golden Cross / Death Cross)  
    - Random Forest training  
    - Model application and signal prediction  

- **Specific Scripts (Datasets 3 & 4)**
  - Scripts dedicated to the **third** and **fourth parts of the dataset**:  
    - **Part 3:** Training the Random Forest model  
    - **Part 4:** Applying the trained model to unseen data and generating predictions  

---

### Workflow Recap
1. Create dataset and build trading signals  
2. Train Random Forest algorithm on **Dataset Part 3**  
3. Apply the trained mo

