# 🏏 Cricket Match Outcome Prediction

An end-to-end machine learning project that predicts whether the batting team will win or lose an IPL cricket match based on the current state of a run chase.

The project uses live match-state features such as batting team, bowling team, venue, runs remaining, balls remaining, wickets, current run rate, required run rate, and target runs to build a binary classification model.

---

## 📌 Project Overview

### Problem Statement

Cricket match outcomes depend on several factors such as the teams involved, venue, target score, remaining runs, remaining balls, wickets, and scoring rate.

This project aims to build a machine learning system that predicts the outcome of an IPL cricket match from the current state of the match.

### What Does the Project Predict?

The model predicts:

- `0` → Batting team loses
- `1` → Batting team wins

### Why Does It Matter?

A cricket match changes continuously after every delivery. By using the current match state, a machine learning model can estimate the likelihood of the batting team winning.

This can be useful for:

- Live match analytics
- Win-probability systems
- Sports analytics
- Decision-support applications
- Demonstrating real-time predictive modelling

---

# 🎯 Project Objectives

The main objectives of this project are:

- Perform exploratory data analysis on IPL ball-by-ball data.
- Understand the factors influencing match outcomes.
- Clean and preprocess the dataset.
- Build a reusable machine learning pipeline.
- Handle numerical and categorical features separately.
- Train multiple classification models.
- Evaluate models using multiple performance metrics.
- Compare model performance.
- Select the best-performing model.
- Save the trained model and preprocessing pipeline.
- Build a prediction pipeline for future deployment.
- Develop a web-based interface for model inference.

---

# 📊 Dataset

## Dataset Name

**IPL Ball-by-Ball Dataset**

## Dataset Source

**Kaggle**

The project uses IPL ball-by-ball match data containing information about the state of a batting team's run chase.

### Dataset Features

| Feature | Description |
|---|---|
| `batting_team` | Team currently batting |
| `bowling_team` | Team currently bowling |
| `city` | Match venue/city |
| `runs_left` | Runs remaining to win |
| `balls_left` | Legal balls remaining |
| `wickets` | Wickets remaining |
| `target_runs` | Target score |
| `cur_run_rate` | Current run rate |
| `req_run_rate` | Required run rate |

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Batting team loses |
| `1` | Batting team wins |

### Important Dataset Note

The provided dataset does **not** contain an explicit toss-result column.

Therefore, toss information is not artificially added to the model.

The current model uses only features actually available in the dataset.

---

# 📈 Dataset Statistics

The original dataset contains approximately:

```text
72,413 rows
13 columns