# 🧠 Customer Churn Prediction – SRM University Case Study

## 🎯 Project Overview

Customer churn is one of the biggest challenges for telecom and subscription-based businesses.  
This project aims to **predict customer churn** using statistical and machine learning techniques — helping organizations identify customers likely to leave and take proactive actions to retain them.

The project demonstrates:
- Data cleaning and statistical analysis
- Exploratory Data Analysis (EDA)
- Model development using **CHAID** and **GAM**
- Model evaluation and validation
- Model deployment using **Pickle**

---

## 📊 Dataset Details

**Dataset Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
**Total Records:** 7,043  
**Features:** 21 (Demographic, Service, and Billing attributes)  
**Target Variable:** `Churn` (1 = Churned, 0 = Retained)

| Type | Example Features | Description |
|------|------------------|-------------|
| **Demographic** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer profile |
| **Service Info** | `InternetService`, `TechSupport`, `StreamingTV` | Subscribed services |
| **Billing Info** | `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | Billing and contracts |
| **Target** | `Churn` | Customer retention status |

---

## 🧹 Data Preparation & Cleaning

Performed:
- Conversion of `TotalCharges` to numeric  
- Handling missing values (median imputation)  
- Removal of duplicates  
- Encoding categorical features using one-hot encoding  
- Splitting data into train/test sets (70/30 stratified)

✅ Final dataset shape after encoding: **(7043, 31)**

---

## 📈 Exploratory Data Analysis (EDA)

Key Insights:
- ~26.5% customers have churned → mild class imbalance  
- Higher `MonthlyCharges` and `Fiber optic` internet correlate with churn  
- Long-term contracts reduce churn probability  

**Visualizations:**
- 📊 Churn distribution  
- 📈 Monthly charges vs churn (boxplot)  
- 🔥 Correlation heatmap  
- 🧮 ROC Curve (GAM)

---

## 🧮 Model Development

Two models were developed and compared:

### **1️⃣ CHAID Decision Tree**
- Based on **Chi-squared Automatic Interaction Detection**
- Generates interpretable rules linking categorical factors with churn
- Key Rule Example:


If Contract = 'Month-to-month' and InternetService = 'Fiber optic' → High churn (~55%)
If Contract = 'Two year' → Low churn (~1%)

- **Accuracy:** 80%  
- **AUC:** 0.82

---

### **2️⃣ Generalized Additive Model (GAM)**
- Extends logistic regression with smooth nonlinear terms  
- Captures complex effects of `tenure`, `MonthlyCharges`, and contract types  
- Implemented using **PyGAM (LogisticGAM)**

**Evaluation Results:**

| Metric | Value |
|--------|--------|
| Accuracy | 79.2% |
| AUC | 0.828 |
| Precision (Churn) | 0.65 |
| Recall (Churn) | 0.47 |
| F1-Score | 0.54 |

---

## ⚖️ Model Comparison

| Model | Accuracy | AUC | Key Strength |
|--------|-----------|-----|--------------|
| **CHAID Decision Tree** | 0.80 | 0.82 | Interpretable business rules |
| **GAM Model** | 0.79 | 0.83 | Captures nonlinear patterns |

**Observation:**  
- CHAID → Better interpretability  
- GAM → Higher AUC and flexible prediction

---

## 🚀 Model Deployment

Deployment handled via **Pickle serialization**:

import pickle

# Save model
pickle.dump(gam, open('churn_gam_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Load and predict
model = pickle.load(open('churn_gam_model.pkl', 'rb'))
sample_prediction = model.predict(sample_scaled)

**Deployment Pipeline:**

1. Save model after training
2. Integrate into a Flask/FastAPI API
3. Predict churn for new customer data in real-time
4. Schedule retraining with new data using CRON or Airflow

---

## 🔁 Model Updating

To maintain accuracy over time:

* Collect new customer churn data monthly
* Reapply preprocessing and retrain GAM model
* Automate pipeline for version control
* Log performance metrics (Accuracy, AUC) for each retrain cycle
---

## 🧰 Tools & Libraries Used

| Category          | Libraries                  |
| ----------------- | -------------------------- |
| **Data Analysis** | Pandas, NumPy              |
| **Visualization** | Matplotlib, Seaborn        |
| **Modeling**      | Scikit-learn, PyGAM, CHAID |
| **Deployment**    | Pickle, Flask-ready        |

---

## 🧾 Results Summary

| Model | Accuracy | AUC  | Interpretation                                 |
| ----- | -------- | ---- | ---------------------------------------------- |
| CHAID | 80%      | 0.82 | Rule-based model for business decisions        |
| GAM   | 79%      | 0.83 | Statistical model capturing nonlinear patterns |

* **Best Predictors:** `Contract`, `InternetService`, `tenure`, and `MonthlyCharges`
* **Actionable Insight:** Offer loyalty benefits for *month-to-month* fiber customers.

---

## 🧠 Key Insights

* Contract length is the **strongest predictor** of churn.
* Customers with **Fiber optic + Month-to-month** contracts are most at risk.
* Predictive analytics can enable **targeted retention campaigns**.
* CHAID helps managers understand **why** customers churn, while GAM tells **who** is likely to churn.

---

## 📚 References

1. Kaggle Dataset: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
2. PyGAM Documentation: [https://pygam.readthedocs.io](https://pygam.readthedocs.io)
3. IBM SPSS Modeler: CHAID Algorithm Guide
4. Hastie, Tibshirani & Friedman (2017) – *Elements of Statistical Learning*
5. Scikit-learn Developer Docs


## 🏁 Acknowledgment

This project was completed as part of the **course requirements** for
*21AIC401T – Inferential Statistics and Predictive Analytics* under the guidance of the **Department of Computational Intelligence, School of Computing, SRM University**.

---
