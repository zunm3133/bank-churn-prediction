# 🏦 Bank Customer Churn Prediction

An end-to-end machine learning project that predicts which bank customers are likely to churn — covering data cleaning, feature engineering, model training, evaluation, and actionable business insights.

---

## 📌 Project Overview

Customer churn costs banks significantly more than retention. This project builds a classification pipeline on 10,000 customer records to identify high-risk customers early, enabling proactive retention strategies.

**Dataset:** [Kaggle — Bank Customer Churn (Churn_Modelling.csv)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)  
10,000 rows · 11 features · 20.4% churn rate (imbalanced)

---

## 🗂️ Project Structure

```
bank-churn-prediction/
│
├── data/
│   ├── raw/                        # Original Churn_Modelling.csv
│   └── processed/                  # Cleaned data, model comparison, scored customers
│
├── models/
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── logistic_regression.pkl     # Trained Logistic Regression model
│   └── random_forest.pkl           # Trained Random Forest model
│
├── notebooks/
│   ├── 01_data_loading_eda.ipynb          # Data loading & exploratory analysis
│   ├── 02_feature_engineering.ipynb       # Encoding, new features, SMOTE, scaling
│   ├── 03_model_training.ipynb            # Model training & comparison
│   ├── 04_model_evaluation.ipynb          # Detailed evaluation & visualisations
│   └── 05_business_insights.ipynb         # Churn risk scoring & customer profiling
│
└── README.md
```

---

## 🔧 Pipeline

```
Raw Data → EDA → Feature Engineering → SMOTE Balancing → Scaling → Model Training → Evaluation → Business Insights
```

---

## ⚙️ Feature Engineering

Beyond the raw features, four new variables were created to improve model signal:

| New Feature | Description |
|---|---|
| `balance_salary_ratio` | Balance relative to estimated salary |
| `age_group` | Age bucketed into 5 bands (<30, 30–40, 40–50, 50–60, 60+) |
| `products_per_year` | Number of products divided by tenure |
| `zero_balance` | Binary flag for customers with zero balance |

**Class imbalance** was addressed using **SMOTE**, balancing the training set from 6,370 / 1,630 to 6,370 / 6,370 before model training.

---

## 🤖 Model Results

Two classifiers were trained and evaluated on a held-out test set of 2,000 customers:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 74.50% | 40.65% | 55.04% | 46.76% | 0.7463 |
| **Random Forest** | **81.75%** | **54.18%** | **66.83%** | **59.85%** | **0.8522** |

**✅ Random Forest** was selected as the final model for its superior performance across all metrics — particularly ROC-AUC (0.85), which is critical for imbalanced churn detection.

---

## 📊 Business Insights

The final model was used to score all 10,000 customers into risk tiers:

| Risk Tier | Customers | Churn Probability Range |
|---|---|---|
| 🔴 High Risk | 1,142 (11.4%) | ≥ 0.75 |
| 🟡 Medium Risk | 1,204 (12.0%) | 0.50 – 0.75 |
| 🟢 Low Risk | 2,053 (20.5%) | 0.25 – 0.50 |
| ⚪ Very Low Risk | 5,601 (56.0%) | < 0.25 |

### High-Risk Customer Profile

Compared to the average customer, high-risk customers are:

- **Older** — avg. age 47.9 vs 38.9 overall
- **Higher balance** — avg. £94,963 vs £76,486 overall
- **Concentrated in Germany** — 632 of 1,142 high-risk customers (55%)
- **Predominantly female** — 788 out of 1,142 (69%)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Class Balancing | imbalanced-learn (SMOTE) |
| Model Persistence | Joblib |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/zunm3133/bank-churn-prediction.git
cd bank-churn-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

**3. Add the dataset**  
Download `Churn_Modelling.csv` from [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) and place it in `data/raw/`.

**4. Run the notebooks in order**
```
01_data_loading_eda.ipynb
02_feature_engineering.ipynb
03_model_training.ipynb
04_model_evaluation.ipynb
05_business_insights.ipynb
```

---

## 👤 Author

**zunm3133** · [GitHub Profile](https://github.com/zunm3133)
