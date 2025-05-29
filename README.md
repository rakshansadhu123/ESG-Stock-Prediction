
# 🧠 ESG-Integrated AI Stock Prediction Model

## 🔍 Project Overview

This project develops an AI-driven investment recommendation system by integrating **Environmental, Social, and Governance (ESG)** scores with traditional **financial stock indicators**. It uses advanced **ensemble machine learning techniques** (XGBoost, LightGBM, CatBoost) and a **rule-based decision system** to provide actionable buy/sell/hold signals.

The model is designed for **ethical, sustainable investing**, aligning high-accuracy predictions with responsible portfolio decisions.

---

## 📁 Dataset Sources

- Historical Stock Data: Kaggle (feature-engineered time-series)
- ESG Scores: Kaggle + S&P Global + Open ESG repositories

---

## ⚙️ Technologies & Libraries

- **Python**, Pandas, NumPy
- **Scikit-learn**, **XGBoost**, **LightGBM**, **CatBoost**
- **Optuna** (for hyperparameter tuning)
- **SHAP** (planned for model explainability)
- **Joblib** (model serialization)

---

## 🧠 Model Architecture

- **Base Models**: XGBoost, LightGBM, CatBoost
- **Stacked Ensemble**: Meta-learner built with XGBoost
- **Rule-Based Guardrail System**:
  - Prioritizes ESG outputs when confidence is high
  - Handles volatility checks and model agreement logic

---

## 🔬 Key Features

- ESG Score Weighting by Industry
- Feature Engineering: Moving Averages, Volatility, ESG Interaction Terms
- Confidence-Based Ensemble Decisions
- Guardrails for responsible investment logic

---

## 📈 Performance Metrics

- ESG Ensemble Model Accuracy: **96.57%**
- Final Stock Model Accuracy (RFC with 2 features): **99.96%**
- Macro F1, Precision, and Recall Scores ≥ **0.96**

---

## 🧪 Notebooks & Scripts

- `submission2.py`: Full end-to-end ESG + stock model
- `submission.py`: ESG-only model variant
- `EDA_Stock.ipynb`: Exploratory data analysis of financial time-series
- `EDA_ESG.ipynb`: ESG dataset exploration
- `stacking_model.pkl`: Final serialized model

---

## 📊 Visual Examples

Include screenshots of:
- Confusion matrices
- ESG vs Financial performance graphs
- Feature importance from SHAP (if added)

---

## 🧭 Project Impact

This system promotes **sustainable investing** by integrating ethical metrics directly into investment decision-making. The model supports:
- Investors seeking **ESG-aligned portfolios**
- Analysts aiming to balance **performance with principles**
- Financial platforms offering **AI-enhanced ethical investing tools**

---

## 🔮 Future Enhancements

- Add **SHAP/LIME** explainability tools
- Connect **real-time ESG feeds**
- Explore **reinforcement learning** for live portfolio adjustment
- Improve **cross-sector generalizability**

---

## 👨‍💻 Author

**Rakshan Sadhu**  
MSc Business Analytics  
[LinkedIn](https://linkedin.com/in/rakshan-sadhu-5726a4183) | rakshansadhu@gmail.com

---

> This project was developed as part of my MSc dissertation at the University of Exeter. It demonstrates the practical potential of AI in building ethical, data-driven financial systems.
