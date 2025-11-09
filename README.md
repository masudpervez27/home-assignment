# Anti-Money Laundering (AML) Transaction Monitoring
## XXXX Bank - SAR Prediction Analysis

---

## 📋 Project Overview

This project develops machine learning models to improve **XXXX Bank's** Anti-Money Laundering (AML) transaction monitoring system. The goal is to predict Suspicious Activity Reports (SARs) more effectively than the current rule-based approach.

### Business Context
- **Bank**: XXXX (Nordic bank, 1,000 customers, 1 year old)
- **Current Rule**: Investigate customers with turnover > €9,000/month
- **Problem**: Missing suspicious customers, inefficient investigations
- **Regulatory Requirement**: FSA mandates better transaction monitoring

### Objectives
1. ✅ **Find more suspicious customers (SARs)** - Improve detection rate
2. ✅ **Reduce False Positives** - Minimize unnecessary investigations  
3. ✅ **Create sustainable controls** - Justifiable and robust monitoring

---

## 📊 Data Description

### Files in `data/` folder:

| File | Description | Records |
|------|-------------|---------|
| `df_kyc.csv` | Customer background information | 1,000 customers |
| `df_transactions.csv` | All customer transactions | ~165,000 transactions |
| `df_label.csv` | SAR filings per customer-month | 12,000 rows (1000×12 months) |

### Key Features:
- **KYC Data**: Age, Sex, High Risk Country, Vulnerable Area, Payment Intentions
- **Transaction Data**: Transaction value, counterparty, type, month
- **Labels**: Binary SAR flag (0=No SAR, 1=SAR Filed)

---

## 🔧 Setup and Installation

### Prerequisites
- Python 3.9+
- UV package manager (or pip)

### Installation

```bash
# Clone the repository
git clone https://github.com/masudpervez27/home-assignment.git
cd home-assignment

# Install dependencies using UV
uv sync

# Or using pip
pip install -r requirements.txt
```

### Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

---

## 🚀 Usage

### Option 1: Run Python Script
```bash
python main.py
```
This will:
- Load and process data
- Engineer features
- Train Random Forest model
- Display performance metrics
- Save trained model to `rf_sar_model.pkl`

### Option 2: Jupyter Notebook (Recommended for Analysis)
```bash
jupyter notebook "Exploration and Data Analysis.ipynb"
```
The notebook contains:
- Comprehensive data exploration
- Statistical analysis
- Feature engineering
- Model development and comparison
- Threshold optimization
- Detailed recommendations

---

## Model Deployment

   - Batch scoring pipeline with monthly scheduling
   - Alert generation
   - Model monitoring, retraining, and governance workflows
   - Integration with existing case management systems

---

## 🔍 Model Artifacts

After running the analysis, the following files are generated:

- `rf_sar_model.pkl` - Trained Random Forest model
- `feature_scaler.pkl` - StandardScaler for feature normalization
- `feature_names.pkl` - List of feature column names

These can be loaded for deployment in production:

```python
import pickle

# Load model
with open('rf_sar_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```
