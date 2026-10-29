
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------- PAGE SETUP ----------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="Credit Default ML Research",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Credit Card Default Prediction")
st.caption(
    "Research dashboard | Imbalanced classification | "
    "UCI Default of Credit Card Clients"
)

# ---------------- LOAD RESULTS ----------------
def load_csv(filename):
    path = RESULTS / filename
    if path.exists():
        return pd.read_csv(path)
    return None


final = load_csv("final_test_metrics.csv")
comparison = load_csv("validation_model_comparison.csv")
thresholds = load_csv("validation_threshold_comparison.csv")
confusion = load_csv("final_confusion_matrix.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Model Comparison",
        "Confusion Matrix",
        "Threshold Analysis",
        "Research Findings",
    ],
)

# ---------------- OVERVIEW ----------------
if section == "Overview":
    st.header("Dataset and Experiment Overview")

    a, b, c = st.columns(3)
    a.metric("Total clients", "30,000")
    b.metric("Input features", "23")
    c.metric("Default rate", "22.12%")

    st.markdown(
        """
        This project investigates how machine-learning methods
        handle imbalanced credit-default classification.

        **Research question:** Can we detect more defaults while
        balancing precision and false-positive predictions?
        """
    )

    st.subheader("Class Distribution")
    image = RESULTS / "class_distribution.png"
    if image.exists():
        st.image(str(image), use_container_width=True)

    st.subheader("Final Test Performance")
    if final is not None and not final.empty:
        row = final.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Selected model", str(row["Selected Model"]))
        b.metric("Decision threshold", f'{row["Selected Threshold"]:.3f}')
        c.metric("Accuracy", f'{row["Accuracy"]:.2%}')

        a, b, c = st.columns(3)
        a.metric("Default precision", f'{row["Precision"]:.2%}')
        b.metric("Default recall", f'{row["Recall"]:.2%}')
        c.metric("Default F1-score", f'{row["F1"]:.2%}')
    else:
        st.warning("Run main.py to generate the final test metrics.")

# ---------------- MODEL COMPARISON ----------------
elif section == "Model Comparison":
    st.header("Candidate Model Comparison")
    st.caption("These candidate metrics are measured on validation data.")

    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)

        chart = RESULTS / "model_comparison.png"
        if chart.exists():
            st.image(str(chart), use_container_width=True)
    else:
        st.warning("Validation comparison CSV was not found.")

# ---------------- CONFUSION MATRIX ----------------
elif section == "Confusion Matrix":
    st.header("Final Test Confusion Matrix")

    chart = RESULTS / "final_confusion_matrix.png"
    if chart.exists():
        st.image(str(chart), use_container_width=True)

    if confusion is not None:
        st.subheader("Confusion Matrix Values")
        st.dataframe(confusion, use_container_width=True)

    st.markdown(
        """
        - **True negative:** Correctly predicted no default.
        - **False positive:** Predicted default when no default occurred.
        - **False negative:** Missed an actual default.
        - **True positive:** Correctly detected a default.
        """
    )

# ---------------- THRESHOLD ANALYSIS ----------------
elif section == "Threshold Analysis":
    st.header("Decision Threshold Explorer")

    if thresholds is not None:
        st.caption(
            "Explore validation results. The threshold was selected "
            "using validation data, not the final test set."
        )

        selected = st.slider(
            "Explore classification threshold",
            min_value=float(thresholds["Threshold"].min()),
            max_value=float(thresholds["Threshold"].max()),
            value=float(thresholds["Threshold"].iloc[0]),
            step=0.025,
        )

        nearest_index = (
            thresholds["Threshold"] - selected
        ).abs().idxmin()
        row = thresholds.loc[nearest_index]

        a, b, c = st.columns(3)
        a.metric("Precision", f'{row["Precision"]:.2%}')
        b.metric("Recall", f'{row["Recall"]:.2%}')
        c.metric("F1-score", f'{row["F1"]:.2%}')

        st.line_chart(
            thresholds.set_index("Threshold")[
                ["Precision", "Recall", "F1"]
            ]
        )

        chart = RESULTS / "threshold_analysis.png"
        if chart.exists():
            st.image(str(chart), use_container_width=True)
    else:
        st.warning("Validation threshold results were not found.")

# ---------------- RESEARCH FINDINGS ----------------
elif section == "Research Findings":
    st.header("Research Findings")

    st.markdown(
        """
        ### What this experiment investigates

        1. Logistic Regression provides a baseline.
        2. Class weighting increases the importance of minority-class errors.
        3. Random Forest captures nonlinear patterns.
        4. SMOTE generates synthetic minority-class training examples.
        5. Threshold selection changes the precision-recall trade-off.

        ### Evaluation metrics

        - **Precision:** How many predicted defaults were actual defaults?
        - **Recall:** How many actual defaults did the model detect?
        - **F1-score:** Harmonic mean of precision and recall.
        - **ROC-AUC:** Ranking performance across classification thresholds.
        - **PR-AUC:** Average precision across precision-recall thresholds.

        ### Limitations

        This is an educational experiment, not a validated credit-risk
        decision system. Results depend on the dataset, split, model
        settings and threshold. Further validation and cost-sensitive
        analysis would be required for real-world decisions.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Research project • Python • scikit-learn • Streamlit")
