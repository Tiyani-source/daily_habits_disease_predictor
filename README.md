⸻


# 🧠 Disease Risk Prediction from Daily Habits  
*A Data Science Group Project *  

[![Streamlit App](https://img.shields.io/badge/🌐_Live_App-Link-blue?style=flat-square)](https://dailyhabitsdiseasepredictor-ifwfkpix8myuadevbeymnh.streamlit.app/)  
[![GitHub](https://img.shields.io/badge/📂_Repository-Link-green?style=flat-square)](https://github.com/Tiyani-source/daily_habits_disease_predictor.git)  
[![Dataset](https://img.shields.io/badge/📊_Kaggle_Dataset-Link-orange?style=flat-square)](https://www.kaggle.com/datasets/mahdimashayekhi/disease-risk-from-daily-habits)

---

## 📘 Overview
This project investigates whether **daily lifestyle and biometric data** can predict an individual's disease risk.  
We used a **large Kaggle dataset (100,000+ records, 43 features)** and applied full-scale **data mining and ML workflows** — from **cleaning and feature engineering** to **model training and Streamlit deployment**.

> ⚠️ The dataset, while comprehensive, showed **low predictive power**, emphasizing the role of **data quality and feature relevance** in health analytics.


---

## 🎯 Objectives
- Clean and preprocess lifestyle–health data for predictive modeling  
- Engineer and select features using statistical and correlation analysis  
- Train ML models and compare performance (Logistic, Tree-based, XGBoost)  
- Deploy the final model via a **Streamlit web app**

---

## 🧹 Data Preprocessing Summary
**Dataset:** 100,000 records × 43 features  
**Target Variable:** `target` (Healthy / Diseased)

**Key Steps**
- 🔍 **Missing Data:**  
  - Dropped: `alcohol_consumption`, `income`, `gene_marker_flag`  
  - Imputed: numerical columns via *Iterative Imputer (Bayesian Ridge)*  
  - Filled: categorical nulls with “Unknown” or “None”
- ⚙️ **Redundant Features Removed:**  
  - `bmi_scaled`, `bmi_corrected`, `bmi_estimated`, `survey_code`
- 📊 **Normalization & Type Cleaning:**  
  - Verified skewness ≈ 0 → no transformation needed  
- 🔁 **Correlation Check:**  
  - Dropped `height` & `weight` (high correlation with BMI)

---

## 🧩 Feature Engineering
- One-hot encoding for categorical features  
- Mutual Information (MI) analysis for importance ranking  
- T-tests & effect-size filtering for numeric predictors  
- Created derived composites like:
  - **Work–life balance** = `sleep_hours` vs `work_hours`  
  - **Lifestyle activity** = combined `daily_steps`, `physical_activity`, `screen_time`

**Final feature count:** 31  

---

## ⚙️ Model Development
| Model | Accuracy | ROC–AUC | F1-score |
|--------|-----------|----------|-----------|
| Logistic Regression | 0.66 | 0.71 | 0.66 |
| Decision Tree | 0.68 | 0.70 | 0.67 |
| **XGBoost (final)** | **0.69** | **0.72** | **0.69** |

Model threshold was fine-tuned using **ROC curve optimization** and deployed via Streamlit.

---

## 🧠 Insights
- Most lifestyle variables (e.g., diet, sleep, caffeine) were **weak individual predictors**.  
- Moderate influence found in **BMI**, **blood pressure**, and **daily steps**.  
- Indicates that **purely self-reported habits** have limited discriminative power for disease prediction.

---

## 🧰 Tech Stack
| Category | Tools |
|-----------|--------|
| **Data Analysis** | Pandas, NumPy, Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, XGBoost |
| **Deployment** | Streamlit |
| **Version Control** | GitHub |

---

## 🚀 Run Locally
```bash
# Clone the repo
git clone <<your-repo-link>>
cd Disease_Risk_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run frontend.py

requirements.txt

streamlit>=1.35
scikit-learn>=1.5
xgboost>=2.0
pandas>=2.1
numpy>=1.26
```

⸻

# 📂 File Structure

📦 Disease_Risk_Prediction
 ┣ 📜 daily_habits_disease_prediction.ipynb   # EDA + model pipeline
 ┣ 📜 frontend.py                             # Streamlit web app
 ┣ 📜 xgb_model.pkl                           # Trained model
 ┣ 📜 selected_features.pkl                   # Final features
 ┣ 📜 best_threshold.pkl                      # ROC-tuned threshold
 ┣ 📜 requirements.txt
 ┗ 📜 README.md


⸻

# 🔮 Future Work
	•	Add explainability using SHAP or LIME
	•	Address class imbalance with SMOTE or focal loss
	•	Integrate real clinical data for higher predictive reliability
	•	Implement a feedback retraining loop for model adaptation

⸻

# 🧾 Citation

Gurusinghe, T.M., Senaratna, S.T.S., Jayathilaka, K.A., & Wickramaarachchi, L.T.B. (2025). Disease Risk Prediction from Daily Habits – SLIIT, IT3051: Fundamentals of Data Mining.

---

## 👥 Team Members
| Name | Git Username |
|-------|------------------|
| S.T.S. Senaratna |  |
| K.A. Jayathilaka |  |
| T.M. Gurusinghe |  |
| L.T.B. Wickramaarachchi | |


⸻

# 📘 Summary

“Even the cleanest data pipeline cannot compensate for weakly predictive data —
this project highlights the importance of feature quality over model complexity.”

⸻
