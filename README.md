# ATP Tennis Match Outcome Predictor

A machine learning project that predicts the outcome of ATP tennis matches using historical match data from 2005 to 2025. The model is trained on engineered features covering player rankings, Elo ratings, head-to-head records, fatigue metrics, surface performance, and tournament context.

---

## Overview

This project builds a binary classifier to predict which of two players (Player A vs. Player B) wins a given match. The target variable is randomly assigned per match to avoid positional bias. Four algorithms are evaluated — Logistic Regression, Random Forest, XGBoost, and LightGBM — and the best-performing model is selected automatically for final evaluation.

---

## Project Structure

```text
.
├── project_final.ipynb       # Main notebook (all steps included)
├── datasets/                 # Downloaded ATP match files (.xls / .xlsx)
│   ├── [Datasets from 2000-2026]              
├── Plots/                    # Experiment plots
│   └── ROC_curve.png         # ROC comparison across all models
│   └── sensitivity_analysis_plots.png  # PDP & ICE sensitivity analysis
└── readme.md                 # Documentation
```
---

## Data

Data is sourced from [tennis-data.co.uk](http://tennis-data.co.uk), covering ATP matches from 2005 to 2025 (years 2000–2004 and 2026 are excluded due to inconsistent column availability). Only completed matches are retained, walkovers and retirements are filtered out.

**Key columns kept:** Player names, rankings, points, set/game scores, surface, court type, tournament series, round, and match date.

---

## Feature Engineering

Features are organized into three incremental blocks:

**Block 0 - Baseline**
- Rank difference (A vs B)
- Points difference (A vs B)
- Global Elo differential
- Surface-specific Elo differential
- Head-to-head win differential, total matches, and Bayesian-smoothed H2H win rate

**Block 1 - Fatigue**
- Rest days since last match (difference)
- EWMA (Exponentially Weighted Moving Average) of total sets and games played, capturing recent physical load

**Block 2 - Surface Performance**
- Historical win rate per surface for each player (difference)

**Block 3 - Context** *(additional)*
- Tournament importance tier (ordinal encoding of Series)
- Round depth / momentum (ordinal encoding of Round)

---

## Modeling

### Train / Validation / Test Split (time-based)
| Set        | Years       |
|------------|-------------|
| Training   | 2005–2019   |
| Validation | 2020–2023   |
| Test       | 2024–2025   |

### Algorithms Compared
- **Logistic Regression** — with StandardScaler and C hyperparameter tuning
- **Random Forest** — 200 estimators, max depth 10
- **XGBoost** — grid search over depth, learning rate, and estimators
- **LightGBM** — grid search over num_leaves, learning rate, and estimators

### Evaluation Metrics
- **Log-Loss** (primary selection criterion)
- **AUC-ROC**
- Accuracy + Classification Report on the test set

The best algorithm (lowest validation Log-Loss) is automatically selected and retrained on the combined train + validation set before final evaluation on the held-out 2024–2025 test data.

---

## Match Prediction

A `predict_match()` function allows inference on any future matchup given two player names, a surface, and a date. It reconstructs all features from historical data up to (but not including) the match date, preventing data leakage.

**Example:**
```python
predict_match("Tien L.", "Medvedev D.", "Hard", "2026-04-15", df, best_xgb_model, best_features)
```

---

## Sensitivity Analysis

Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) curves are generated for the top features (`Elo_Diff_A_B`, `Rank_Diff_A_B`, `Surface_Elo_Diff_A_B`, `Pts_Diff_A_B`) to visualize how each feature influences Player A's predicted win probability.

---

## Requirements
pandas
numpy
matplotlib
scikit-learn
xgboost
lightgbm
requests
xlrd

Install all dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm requests xlrd
```

---

## Usage

1. Run all cells sequentially in `project_final.ipynb`.
2. Data is downloaded automatically to the `datasets/` folder.
3. Feature engineering, model training, and evaluation run end-to-end.
4. Use `predict_match()` in Section 5 for live match predictions.
