# IPL Cricket Match Outcome Prediction

This project's main goal is to predict the outcome of an IPL cricket match using historical match information such as team performance, toss result, venue, and other relevant match statistics. The project follows an end-to-end machine learning workflow, from data ingestion and experimentation to model training and deployment.

## Dataset

The project uses the **IPL Ball-by-Ball Dataset** from Kaggle.

The dataset contains historical IPL cricket match and ball-by-ball information that will be used to understand team performance and develop features for predicting match outcomes.

The dataset is kept locally under the `data/raw/` directory and is not included in this GitHub repository.

## Project Structure

```text
Cricket-Match-Outcome-Prediction/
│
├── data/
│   └── raw/
│       └── IPL Dataset
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py
│       └── predict_pipeline.py
│
├── artifacts/
│
├── logs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Clone this repository:

```bash
git clone <your-github-repository-url>
cd Cricket-Match-Outcome-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

### Activate in Windows

```bash
venv\Scripts\activate
```

### Activate in MacOS/Linux

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Update 1

**Environment Setup** is done

**Project Structure** is defined

**Necessary files** required till now are created

**Virtual Environment** is created and configured

**Git Repository** is initialized and connected with GitHub

## Update 2

**Logging** is implemented

**Custom Exception Handling** is implemented

**Data Ingestion** is implemented with logging and exception handling

**Git Branching & Pull Request Workflow** is completed

**Feature Branch** is created and merged into `main`

## Project Objective

The objective of this project is to build a machine learning system that can predict the winner of an IPL cricket match based on relevant historical match information.

The project will gradually progress through:

```text
Data Ingestion
      ↓
Data Exploration
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Best Model Selection
      ↓
Prediction Pipeline
      ↓
Web Application
      ↓
Deployment
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Flask
* Git
* GitHub

## Current Status

**Unit 1:** Environment and Project Setup — Completed

**Unit 2:** Logging, Exception Handling & Git Essentials — In Progress/Completed

**Unit 3:** Experiment Tracking & Pipeline Structuring — Upcoming

**Unit 4:** Data Ingestion & Data Transformation — Upcoming

**Unit 5:** Model Training & Hyperparameter Tuning — Upcoming

**Unit 6:** Prediction Pipeline & Model Deployment — Upcoming
