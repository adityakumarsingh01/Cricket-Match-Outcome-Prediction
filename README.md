# 🏏 Cricket Match Outcome Prediction

> An end-to-end Machine Learning project to predict the real-time outcome of an IPL cricket match based on the current state of a run chase.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Unit_4_Completed-brightgreen.svg)

---

## 📌 Project Overview

Cricket match outcomes are highly dynamic and depend on several fluid factors, such as the teams playing, venue, target score, remaining runs, balls left, wickets in hand, and the scoring rate.

This project aims to build a robust machine learning system that predicts the outcome of an IPL cricket match from any given state in the second innings (the run chase).

### 🎯 What Does the Project Predict?
The model acts as a binary classifier predicting the fate of the chasing team:
- `0` → Batting team loses
- `1` → Batting team wins

### 💡 Why Does It Matter?
A cricket match changes continuously after every delivery. By using the live match state, a machine learning model can estimate the likelihood of the batting team winning in real-time. This is highly relevant for:
- Live match analytics and broadcasting
- Win-probability systems (similar to Cricinfo's Forecaster)
- Sports analytics and strategy planning
- Demonstrating real-time predictive modelling capabilities

---

## 🎯 Objectives

This project follows a complete predictive analytics lifecycle:
- Perform in-depth Exploratory Data Analysis (EDA) on IPL ball-by-ball data.
- Uncover factors heavily influencing match outcomes.
- Engineer and clean features to create a robust dataset.
- Build a reusable, production-ready machine learning pipeline.
- Handle numerical and categorical features systematically.
- Train, evaluate, and compare multiple classification models.
- Select the best-performing model based on robust metrics (ROC-AUC).
- Serialize the trained model and preprocessing pipeline for deployment.
- (Upcoming) Develop a Flask-based web interface for model inference.

---

## 📊 Dataset

### Source
The data is sourced from **Kaggle** and comprises IPL ball-by-ball match records detailing the exact state of a batting team's run chase.

### Key Features
| Feature | Description |
|---|---|
| `batting_team` | Team currently batting |
| `bowling_team` | Team currently bowling |
| `city` | Match venue/city |
| `runs_left` | Runs remaining to win |
| `balls_left` | Legal balls remaining |
| `wickets` | Wickets remaining in hand |
| `target_runs` | Target score to win |
| `cur_run_rate` | Current run rate |
| `req_run_rate` | Required run rate |

### Dataset Note
*The provided dataset does **not** contain an explicit toss-result column. To maintain authenticity, toss information is not artificially engineered into the model. The model relies purely on the features actually available and reflective of the live match state.*

---

## 📈 Dataset Statistics

**Initial State:**
- Total Records: ~72,413 rows
- Total Columns: 13

**Post-Cleaning & Split:**
- Training samples: 56,928
- Testing samples: 14,233
- Input features: 9

**Post-Transformation:**
- Transformed features: 56 (due to One-Hot Encoding)

---

## 🏗️ Project Structure

```text
Cricket-Match-Outcome-Prediction/
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned train/test splits
│
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── logger.py            # Custom logging system
│   ├── exception.py         # Custom exception handling
│   │
│   └── components/
│       ├── __init__.py
│       ├── data_ingestion.py       # Data loading & splitting
│       ├── data_transformation.py  # Feature engineering & scaling
│       └── model_trainer.py        # Model training & evaluation
│
├── artifacts/
│   ├── model.pkl            # Serialized best model
│   └── preprocessor.pkl     # Serialized transformation pipeline
│
├── logs/                    # Execution logs
│
├── requirements.txt         # Project dependencies
├── README.md
└── .gitignore
```

---

## 🔄 Machine Learning Workflow

The project is built on an end-to-end pipeline architecture ensuring reproducibility:

1. **Data Ingestion:** Loads raw data, cleans structural issues, handles invalid states, and splits data into `train.csv` and `test.csv`.
2. **Data Transformation:** 
   - *Numerical Features:* Median Imputation → Standard Scaling
   - *Categorical Features:* Most Frequent Imputation → One-Hot Encoding
3. **Model Training:** Trains Logistic Regression (baseline) and Random Forest Classifier.
4. **Evaluation:** Compares models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
5. **Serialization:** Saves the best model and preprocessor as `.pkl` artifacts.

---

## 🤖 Models & Current Results

### 1. Logistic Regression (Baseline)
Provides a simple, interpretable benchmark with fast training and probability-based predictions.

### 2. Random Forest Classifier (Selected)
Used to capture non-linear relationships and complex interactions between match-state variables (e.g., how `runs_left` and `wickets` jointly impact the outcome).

### Performance Metrics

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 80.75% | 89.59% | 80.94% | 82.85% | 81.88% |
| **Random Forest** | **97.37%** | **99.76%** | **96.39%** | **98.69%** | **97.53%** |

*Note: The current 97.37% accuracy is based on a row-level random train/test split. Because ball-by-ball data contains multiple observations from the same match, chronological or match-aware validation strategies are planned to ensure robust real-world generalization.*

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Development & Tracking:** Jupyter, VS Code, Git, Custom Logger/Exceptions
- **Deployment (Upcoming):** Flask, HTML, CSS

---

## ⚙️ Environment Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/adityakumarsingh01/Cricket-Match-Outcome-Prediction.git
cd Cricket-Match-Outcome-Prediction
```

### 2. Virtual Environment Setup
**Windows:**
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```
**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

**Data Ingestion:**
```bash
python -c "from src.components.data_ingestion import DataIngestion; DataIngestion().initiate_data_ingestion()"
```

**Data Transformation:**
```bash
python -c "from src.components.data_ingestion import DataIngestion; from src.components.data_transformation import DataTransformation; i=DataIngestion(); train,test=i.initiate_data_ingestion(); DataTransformation().initiate_data_transformation(train,test)"
```

**Model Training:** *(To be integrated into training pipeline script)*
```bash
python -c "from src.components.model_trainer import ModelTrainer; # (Execution logic)"
```

---

## 📝 Error Handling & Logging

- **Logging:** Implemented in `src/logger.py`, capturing all pipeline executions, data states, and info messages.
- **Exception Handling:** Implemented in `src/exception.py` to trace exact errors with filenames and line numbers.

---

## 📌 Current Project Status

- [x] **Unit 1** — Environment & Project Setup
- [x] **Unit 2** — Logging, Exception Handling & Git
- [x] **Unit 3** — Experiment Tracking & EDA
- [x] **Unit 4** — Data Ingestion & Data Transformation
- [ ] **Unit 5** — Model Training & Hyperparameter Tuning *(Next phase)*
- [ ] **Unit 6** — Prediction Pipeline & Deployment

---

## 🚀 Future Enhancements

- **Match-Aware Validation:** Implementing chronological splits to prevent data leakage.
- **Hyperparameter Tuning:** Utilizing `GridSearchCV` / `RandomizedSearchCV` for the Random Forest model.
- **Feature Importance:** Analyzing which match states drive the highest win probability shifts.
- **Web Application:** Building a Flask app for interactive real-time predictions.

---
*This project is being developed as part of the **Predictive Analytics Project** coursework.*

**Author:** Aditya Kumar Singh  
*B.Tech. (Hons.) (CSE- Data Science and Data Engineering) Student*