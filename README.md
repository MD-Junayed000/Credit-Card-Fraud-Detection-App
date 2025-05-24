# 💳 Credit Card Fraud Detection System

This project is a fully interactive **machine learning web app** built using **Streamlit** that allows you to:

✅ Detect fraudulent credit card transactions  
📂 Upload CSV files for batch predictions  
📊 Visualize fraud distribution, amounts, and ratios  
📥 Download results with fraud probabilities

---

## 📦 Dataset Information

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **Features**:
  - V1–V28 → Principal Components (PCA-transformed)
  - Amount → Raw transaction amount
  - TIme
  - Class → Target (1 = Fraud, 0 = Not Fraud)

---

## 🧠 Machine Learning Model

We use **Random Forest Classifier** trained on a balanced subset using **undersampling**:

### ✅ Preprocessing:

- Removed duplicate transactions
- Sampled 473 normal transactions + all frauds
- Dropped `Time` column for consistency
- Split into train/test using `train_test_split`

### ✅ Model Training Script:

`train_and_save_model.py` handles preprocessing, training, and model export:

##  Running Locally

**Install requirements**(creating a python environment is more preferable
```bash
pip install -r requirements.txt
```
**Train model (optional)**
```bash
python train_and_save_model.py
```
Output: rf_fraud_model.pkl (saved using joblib)

**Launch app**
```bash
streamlit run app.py
```
