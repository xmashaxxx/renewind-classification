"""
ReneWind — Cost-Sensitive Wind Turbine Failure Prediction
Trains XGBoost and Random Forest with threshold optimization
Interactive cost calculator: see how threshold changes affect total maintenance cost
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ReneWind Failure Predictor", page_icon="🌬️", layout="wide")

# ── Cost constants ────────────────────────────────────────────────────────────
REPAIR_COST = 10      # TP: catch failure early, repair
INSPECT_COST = 1      # FP: unnecessary inspection
REPLACE_COST = 50     # FN: missed failure, full replacement


@st.cache_resource
def train_models(_train_df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    df = _train_df.copy()
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            eval_metric="logloss", verbosity=0
        ) if HAS_XGB else None,
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
    }
    models = {k: v for k, v in models.items() if v is not None}

    trained = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        trained[name] = model

    return trained, imputer, scaler, list(X.columns), target_col


def compute_cost(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total_cost = tp * REPAIR_COST + fp * INSPECT_COST + fn * REPLACE_COST
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Cost": total_cost, "Recall": recall, "Precision": precision,
        "F1": 2 * recall * precision / max(recall + precision, 1e-9)
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🌬️ ReneWind — Turbine Failure Predictor")
st.caption(
    "Cost-sensitive classification · Repair < Inspect < Replace · "
    "Recall is the primary metric"
)

st.sidebar.header("Setup")
train_file = st.sidebar.file_uploader("Upload Train.csv", type="csv", key="train")
test_file = st.sidebar.file_uploader("Upload Test.csv (optional)", type="csv", key="test")

if train_file is None:
    st.info(
        "**To get started:** upload `Train.csv` using the sidebar.\n\n"
        "The dataset is coursework property and is not included in this repository. "
        "See `data/README.md` for the expected structure."
    )
    st.markdown("---")
    st.subheader("About this project")

    col1, col2, col3 = st.columns(3)
    col1.metric("Repair Cost (TP)", f"${REPAIR_COST}k", help="Caught early — fix the generator")
    col2.metric("Inspection Cost (FP)", f"${INSPECT_COST}k", help="False alarm — unnecessary check")
    col3.metric("Replacement Cost (FN)", f"${REPLACE_COST}k", help="Missed failure — full replacement")

    st.markdown("""
This app extends my ReneWind classification analysis with an interactive cost calculator.
The core insight: for this problem, **False Negatives (missed failures) are 5x more expensive
than True Positives (caught failures)**. Standard accuracy-maximizing models are wrong here
— you need to optimize for recall on the failure class, even at the cost of more false alarms.

**Key design decisions:**
- Primary metric: **Recall on failure class** (class 1)
- Cost-weighted class balancing during training
- Interactive threshold tool showing real maintenance cost at every cutoff
- XGBoost with `scale_pos_weight` to handle the 5.5% class imbalance
    """)
    st.stop()

train_df = pd.read_csv(train_file)

with st.spinner("Training models..."):
    models, imputer, scaler, feature_cols, target_col = train_models(train_df)

# Prepare training predictions for threshold analysis
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_train_proc = scaler.transform(imputer.transform(X_train))

tab1, tab2, tab3 = st.tabs(["Cost Calculator", "Model Performance", "Feature Importance"])

# ── TAB 1 — Cost Calculator ───────────────────────────────────────────────────
with tab1:
    st.header("Interactive Cost Calculator")
    st.caption(
        f"Repair (TP): ${REPAIR_COST}k · "
        f"Inspection (FP): ${INSPECT_COST}k · "
        f"Replacement (FN): ${REPLACE_COST}k"
    )

    selected_model = st.selectbox("Select model", list(models.keys()))
    model = models[selected_model]

    # Compute cost curve across all thresholds
    y_prob = model.predict_proba(X_train_proc)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    cost_data = []
    for t in thresholds:
        result = compute_cost(y_train.values, y_prob, t)
        result["Threshold"] = round(t, 2)
        cost_data.append(result)
    cost_df = pd.DataFrame(cost_data)

    optimal_idx = cost_df["Cost"].idxmin()
    optimal_threshold = cost_df.loc[optimal_idx, "Threshold"]
    optimal_cost = cost_df.loc[optimal_idx, "Cost"]

    st.success(
        f"**Optimal threshold for {selected_model}:** {optimal_threshold:.2f} "
        f"→ Total cost: ${optimal_cost:,}k"
    )

    # Cost curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cost_df["Threshold"], y=cost_df["Cost"],
        mode="lines", name="Total Cost", line=dict(color="#4C8EDA", width=2)
    ))
    fig.add_vline(
        x=optimal_threshold, line_dash="dash", line_color="green",
        annotation_text=f"Optimal: {optimal_threshold:.2f}",
        annotation_position="top right"
    )
    fig.update_layout(
        title="Total Maintenance Cost vs Classification Threshold",
        xaxis_title="Threshold", yaxis_title="Total Cost ($k)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interactive threshold slider
    st.subheader("Explore a Specific Threshold")
    threshold = st.slider(
        "Classification threshold", 0.01, 0.99,
        float(optimal_threshold), 0.01
    )

    result = compute_cost(y_train.values, y_prob, threshold)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Cost", f"${result['Cost']:,}k")
    c2.metric("Recall", f"{result['Recall']:.3f}")
    c3.metric("Precision", f"{result['Precision']:.3f}")
    c4.metric("Missed Failures (FN)", f"{result['FN']:,}")
    c5.metric("False Alarms (FP)", f"{result['FP']:,}")

    # Confusion matrix
    cm_data = pd.DataFrame(
        [[result["TN"], result["FP"]], [result["FN"], result["TP"]]],
        index=["Actual: No Failure", "Actual: Failure"],
        columns=["Predicted: No Failure", "Predicted: Failure"]
    )
    fig_cm = px.imshow(
        cm_data, text_auto=True, color_continuous_scale="Blues",
        title=f"Confusion Matrix at threshold={threshold:.2f}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Cost breakdown
    col1, col2 = st.columns(2)
    with col1:
        breakdown = pd.DataFrame({
            "Category": ["Repairs (TP)", "Inspections (FP)", "Replacements (FN)"],
            "Count": [result["TP"], result["FP"], result["FN"]],
            "Unit Cost ($k)": [REPAIR_COST, INSPECT_COST, REPLACE_COST],
            "Total Cost ($k)": [
                result["TP"] * REPAIR_COST,
                result["FP"] * INSPECT_COST,
                result["FN"] * REPLACE_COST
            ]
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with col2:
        fig_pie = px.pie(
            breakdown, values="Total Cost ($k)", names="Category",
            title="Cost Breakdown by Category",
            color_discrete_sequence=["#2ECC71", "#F39C12", "#E74C3C"]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ── TAB 2 — Model Performance ─────────────────────────────────────────────────
with tab2:
    st.header("Model Performance at Optimal Thresholds")

    rows = []
    for name, model in models.items():
        y_prob_m = model.predict_proba(X_train_proc)[:, 1]
        cd = pd.DataFrame([
            {"t": t, **compute_cost(y_train.values, y_prob_m, t)}
            for t in np.linspace(0.01, 0.99, 99)
        ])
        best = cd.loc[cd["Cost"].idxmin()]
        rows.append({
            "Model": name,
            "Optimal Threshold": round(best["t"], 2),
            "Min Cost ($k)": int(best["Cost"]),
            "Recall": round(best["Recall"], 3),
            "Precision": round(best["Precision"], 3),
            "F1": round(best["F1"], 3),
            "Missed Failures": int(best["FN"]),
        })

    perf_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Recall vs Precision Trade-off")
    for name, model in models.items():
        y_prob_m = model.predict_proba(X_train_proc)[:, 1]
        curve_data = [
            {"Threshold": t,
             "Recall": compute_cost(y_train.values, y_prob_m, t)["Recall"],
             "Precision": compute_cost(y_train.values, y_prob_m, t)["Precision"],
             "Model": name}
            for t in np.linspace(0.01, 0.99, 50)
        ]
        curve_df = pd.DataFrame(curve_data)
        fig = px.line(
            curve_df, x="Recall", y="Precision",
            title=f"{name} — Precision-Recall Curve",
            labels={"Recall": "Recall (failure class)", "Precision": "Precision"}
        )
        fig.update_layout(xaxis_range=[0, 1.05], yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3 — Feature Importance ────────────────────────────────────────────────
with tab3:
    st.header("Feature Importance")
    for name, model in models.items():
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title=f"{name} — Top 15 Features",
            color="Importance", color_continuous_scale="Blues"
        )
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
