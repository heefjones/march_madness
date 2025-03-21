# [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/overview) 🏀
## Intro
There are __9,223,372,036,854,775,808__ (2^63) possible bracket combinations. To guarantee a perfect bracket, each of the 8 billion people on Earth would need to generate over 1 billion unique brackets. This Kaggle competition tasked me with calculating win probabilities for all possible March Madness matchups in both the men's and women's tournaments.

## Data
### 1. Compact Dataset
- Regular season scoring and win/loss data from 1985–2025 for men (41 seasons, excluding COVID) and 1998–2025 for women (27 seasons).
- Aggregated by mean (general performance) and standard deviation (consistency).
- Key features: mean_points_for, mean_points_against, std_win_pct, etc.
### 2. Detailed Dataset
- Advanced stats (shooting, assists, rebounds, defensive metrics, turnovers, fouls) from 2003–2025 for men and 2010–2025 for women.
- Aggregated by mean and standard deviation for both team and opponent stats.
- Smaller sample size but richer feature set.
### 3. Historical Tournament Results
- Historical outcomes and seed information from 1985 for men and 1998 for women.
- Used as labels in the model training process.

## Modeling
A “chalk” bracket (always picking the higher seed or better win percentage when seeds are equal) offers a reliable baseline:
- Men's bracket: __71.01% accuracy__
- Women's bracket: __77.96% accuracy__
### Model Iteration
- Tested multiple classification models with hyperparameter tuning.
- Evaluated performance using 10-fold cross-validation.
- Best men's model: Logistic Regression on compact data with 71.54% accuracy (+0.53% vs. chalk).
- Best women's model: Logistic Regression on compact data with 78.97% accuracy (+1.01% vs. chalk).
- Despite minimal improvement, the unpredictability of March Madness makes significant accuracy gains challenging. This is also the reason so many people love to tune in.

## Files
📊 eda_compact.ipynb – EDA and feature engineering with compact data.
📊 eda_detailed.ipynb – EDA and feature engineering with detailed data.
🤖 preds_compact.ipynb – Model testing on compact features for both tournaments.
🤖 preds_detailed.ipynb – Model testing on detailed features for both tournaments.
🛠️ helper.py – Custom functions for data processing, visualization, and model training.

## Repository Structure
/march_madness
├── eda_compact.ipynb
├── preds_compact.ipynb
├── eda_detailed.ipynb
├── preds_detailed.ipynb
├── helper.py
├── /models
│   ├── models_compact.csv
│   └── models_detailed.csv
└── README.md