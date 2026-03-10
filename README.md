# Telecom Customer Churn Prediction
### Predicting which customers are likely to cancel, before they do


## Problem Statement

Telecom companies lose significant revenue when customers cancel their subscriptions, a problem known as churn. Using a dataset of 7,043 telecom customers, I built a binary classification model to predict whether a customer is likely to churn based on their demographics, subscription type, contract details and usage patterns. This helps the telecom business proactively identify at-risk customers and intervene before they leave reducing revenue loss and improving customer retention.


## Dataset

| Property | Details |
|----------|---------|
| Source | Telecom Customer Dataset |
| Rows | 7,043 |
| Features | 21 (reduced to 12 after engineering) |
| Target | `Churn` (Yes/No) |
| Task | Binary Classification |
| Class Imbalance | ~73% No Churn / ~27% Churn |

## Project Workflow

```
Data Loading → EDA → Feature Engineering & Selection
→ Preprocessing Pipeline → Baseline Model → Model Comparison (CV)
→ Hyperparameter Tuning → Final Evaluation → Feature Importance
```

---

## Exploratory Data Analysis

### Key Findings

- Customers who churned have significantly **lower tenure** — new customers are the highest risk group
- **Higher monthly charges** correlate strongly with churn — price sensitive customers look for alternatives
- **Contract type** is one of the strongest predictors — month to month customers churn dramatically more than those on longer contracts
- **Fiber optic** customers churn the most despite paying a premium, suggesting a gap between price and perceived value
- **Gender** and **PhoneService** showed identical churn rates across categories — no predictive signal

### Categorical Correlation (Cramér's V)
Used Cramér's V instead of standard correlation since most features are categorical. Key findings:
- `StreamingMovies` and `StreamingTV` are strongly correlated (V=0.77)
- `PhoneService` and `MultipleLines` are perfectly correlated (V=1.0)
- `Contract` shows the strongest association with churn among all categorical features

---

## Feature Engineering & Selection

**Dropped features:**
- `customerID` — unique identifier, no predictive value
- `gender` — zero association with churn
- `PhoneService` — perfectly correlated with MultipleLines, redundant
- `TotalCharges` — highly correlated with tenure (0.83). Tenure retained as it shows stronger correlation with churn (-0.35 vs -0.20) and is a more direct measure of loyalty

**Engineered features:**
- `num_addons` combined 6 correlated add-on features (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) into a single feature representing overall customer engagement

**Type correction:**
- `SeniorCitizen` was stored as integer (0/1) but represents a binary category, converted to object type before encoding

---

## Preprocessing Pipeline

| Column Group | Transformer |
|-------------|-------------|
| Numerical | `RobustScaler` |
| Categorical | `OneHotEncoder(drop='first')` |

`RandomOverSampler` was applied inside the pipeline on training folds only to handle class imbalance without leaking into validation data.

---

### Why Recall?

In churn prediction, **missing a churner is far more costly than a false alarm**. If the model misses a churner, that customer cancels and that revenue is gone. If the model incorrectly flags a loyal customer, the worst case is an unnecessary retention offer, a small cost by comparison. This is why **recall was used as the primary evaluation metric** throughout.

---

## Model Comparison (5-Fold Cross Validation)

All models were evaluated using cross validation on the training set. The test set was never touched during this phase.

| Model | Train Recall | Test Recall | Precision | F1 |
|-------|-------------|-------------|-----------|-----|
| **Logistic Regression** | **0.794** | **0.783** | **0.501** | **0.611** |
| Gradient Boosting | 0.833 | 0.789 | 0.509 | 0.619 |
| SVM | 0.825 | 0.781 | 0.493 | 0.604 |
| CatBoost | 0.910 | 0.715 | 0.528 | 0.607 |
| LightGBM | 0.908 | 0.710 | 0.525 | 0.603 |
| KNN | 0.896 | 0.697 | 0.450 | 0.547 |
| XGBoost | 0.957 | 0.644 | 0.537 | 0.585 |
| Random Forest | 0.999 | 0.566 | 0.575 | 0.570 |

**Logistic Regression was selected** not because it had the absolute highest test recall, but because it had the smallest gap between train and test scores, showing the best generalization. Complex models like Random Forest (Train: 0.999, Test: 0.566) were clearly memorizing the training data.

---

## Hyperparameter Tuning

Tuned Logistic Regression using `RandomizedSearchCV` with 5-fold cross validation on training data only.

```python
param_distributions = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear']
}
```

| | Before Tuning | After Tuning |
|-|--------------|-------------|
| CV Recall | 0.783 | 0.813 |

---

## Final Model Performance

**Model:** Tuned Logistic Regression
**Evaluated once on held-out test set:**

| Metric | Score |
|--------|-------|
| Recall | 0.850 |
| Precision | 0.500 |
| F1 Score | 0.630 |
| Accuracy | 0.740 |

**Confusion Matrix:**
```
                Predicted No    Predicted Yes
Actual No           723             313
Actual Yes           56             317
```

Out of every 100 customers who will churn, the model catches 85 of them giving the business a window to intervene before they cancel. The model significantly outperforms the baseline DummyClassifier which achieved only 0.24 recall.

---

### Feature Importance

Based on Logistic Regression coefficients:

**Strongest churn drivers (increase churn risk):**
- High `MonthlyCharges` — expensive customers actively seek cheaper alternatives
- `Fiber optic` internet — premium price, higher churn
- `Electronic check` payment — less committed payment method

**Strongest retention factors (decrease churn risk):**
- `tenure` — the longer a customer stays, the less likely they are to leave
- `Two year contract` — long term commitment dramatically reduces churn
- Having `Dependents` — family responsibilities correlate with stability

---

## Business Recommendations

1. **Target new customers first** — tenure is the strongest churn predictor. The first few months are the highest risk window. Early engagement programs could significantly reduce churn.

2. **Push long term contracts** — two year contract customers are the most loyal segment. Discounts or perks for longer commitments is likely the most effective retention strategy.

3. **Investigate fiber optic service quality** — fiber customers pay more but churn more. A service quality review or targeted retention offers for this segment is overdue.

4. **Incentivize automatic payments** — electronic check payers churn more. Nudging customers toward automatic bank transfers or card payments could improve retention.

5. **Watch high monthly charge customers** — they have the most financial incentive to find cheaper alternatives. Periodic plan reviews or loyalty rewards could reduce their risk.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Imoleoluwanimi/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook TELECOM_CHURN_PREDICTION (4).ipynb

# Run the Streamlit app
streamlit run app.py
```

### Load Saved Model
```python
import joblib
import pandas as pd

pipeline = joblib.load('telecom_churn_pipeline.pkl')
prediction = pipeline.predict(new_customer_data)
probability = pipeline.predict_proba(new_customer_data)[:, 1]
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
joblib
streamlit
jupyter
```


### Author

Delight Abioye
Data Scientist | WQU Applied Data Science Lab Graduate

