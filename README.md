# ReneWind a Cost-Sensitive Turbine Failure Predictor

Interactive machine learning app for wind turbine generator failure prediction, built around a cost-sensitive evaluation framework. The core insight: missing a failure costs 5x more than catching one early, so standard accuracy-maximizing models are the wrong choice here.

## Live Demo

[▶ Open Interactive App](https://renewind-classification-xoxdbyqwqsbhw8zd46gmgp.streamlit.app/)

## Business Problem

ReneWind operates wind turbines fitted with sensors that collect data on environmental and mechanical conditions. The goal is to predict generator failures before they occur so maintenance teams can intervene early — repairing a generator is far cheaper than replacing one.

**Asymmetric cost structure:**

| Outcome | Meaning | Cost |
|---------|---------|------|
| True Positive (TP) | Failure caught → repair | $10k |
| False Positive (FP) | False alarm → unnecessary inspection | $1k |
| False Negative (FN) | Missed failure → full replacement | $50k |

Because replacement costs 5× more than repair, **recall on the failure class is the primary metric**. A model that catches every failure at the cost of some false alarms is preferable to one that is "accurate" but misses failures.

## What It Does

Upload `Train.csv` and the app:

- **Trains XGBoost and Random Forest** - with cost-weighted class balancing to handle the 5.5% failure class imbalance
- **Cost Calculator** — plots total maintenance cost across every possible classification threshold, finds the optimal threshold automatically, and lets you drag a slider to explore the cost, recall, precision, confusion matrix, and cost breakdown at any threshold in real time
- **Model Performance** — compares models at their respective optimal thresholds: recall, precision, F1, missed failures, and minimum cost
- **Feature Importance** — top 15 predictive sensor features per model

## Key Design Decisions

- **Primary metric: Recall** on the failure class, the business cost of a missed failure (FN) far outweighs the cost of a false alarm (FP)
- **Class imbalance handling:** XGBoost uses `scale_pos_weight` proportional to class ratio; Random Forest uses `class_weight="balanced"`
- **Threshold optimization:** instead of using the default 0.5 cutoff, the app finds the threshold that minimizes total maintenance cost, this is the correct business-aligned decision boundary
- **Median imputation** for missing sensor values, fit on training data only to prevent leakage

## Dataset

- **Train.csv:** ~40,000 rows × 40+ anonymized sensor features + binary target (1=failure, 0=no failure)
- **Test.csv:** held-out evaluation set
- **Class distribution:** ~5.5% failures (class 1), ~94.5% no failure (class 0)
- **Source:** MIT IDSS coursework — not redistributable

See [`data/README.md`](data/README.md) for full structure details.

## Running Locally

```bash
git clone https://github.com/xmashaxxx/renewind-classification.git
cd renewind-classification
pip install -r requirements.txt
streamlit run app.py
```

Upload `Train.csv` when prompted.

## Skills Demonstrated

`Python` `scikit-learn` `XGBoost` `pandas` `plotly` `Streamlit` `Cost-Sensitive Learning` `Class Imbalance` `Threshold Optimization` `Precision-Recall Trade-off` `Predictive Maintenance`
