
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Imbalanced Data ML Lab",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Imbalanced Data ML Lab")
st.caption(
    "Explore datasets, train models, compare performance, "
    "and investigate the precision-recall trade-off."
)


# ==================================================
# DATASET LOADING
# ==================================================

@st.cache_data(show_spinner="Loading built-in credit dataset...")
def load_builtin_dataset():
    dataset = fetch_ucirepo(id=350)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    target_name = str(targets.columns[0])
    data = features.copy()
    data[target_name] = targets.iloc[:, 0].values

    return data, target_name


def load_custom_dataset(uploaded_file):
    return pd.read_csv(uploaded_file)


# ==================================================
# DATA PREPROCESSING
# ==================================================

def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    return ColumnTransformer(transformers)


def metric_values(y_true, predictions, probabilities):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(
            y_true, probabilities
        ),
    }


# ==================================================
# MODEL EXPERIMENT
# ==================================================

def run_experiment(data, target_column, positive_class):
    data = data.dropna(subset=[target_column]).copy()

    X = data.drop(columns=[target_column])
    y_raw = data[target_column]

    # Convert the selected positive class to 1.
    y = (y_raw == positive_class).astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "The target must contain exactly two classes "
            "after selecting the positive class."
        )

    if y.value_counts().min() < 5:
        raise ValueError(
            "Each target class needs at least 5 rows "
            "for a stratified experiment."
        )

    if X.shape[1] == 0:
        raise ValueError("The dataset has no input features.")

    # Remove columns that contain no usable information.
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    # Test data stays separate from model and threshold selection.
    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=42,
        stratify=y_development,
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=2000)),
        ]),

        "Class-Weighted Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )),
        ]),

        "Random Forest": Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        "Random Forest + SMOTE": ImbPipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    validation_rows = []
    fitted_models = {}
    validation_probabilities = {}

    progress = st.progress(0, text="Preparing experiments...")

    for index, (name, model) in enumerate(candidates.items()):
        progress.progress(
            index / len(candidates),
            text=f"Training {name}...",
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        row = {"Model": name}
        row.update(
            metric_values(y_val, predictions, probabilities)
        )
        validation_rows.append(row)

        fitted_models[name] = model
        validation_probabilities[name] = probabilities

    progress.progress(1.0, text="Model comparison complete.")

    validation_results = pd.DataFrame(validation_rows)
    validation_results = validation_results.sort_values(
        "PR-AUC", ascending=False
    ).reset_index(drop=True)

    # Choose model based on validation average precision.
    best_name = validation_results.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    val_prob = validation_probabilities[best_name]

    # Select the decision threshold using validation data.
    threshold_rows = []

    for threshold in np.arange(0.20, 0.601, 0.025):
        predictions = (val_prob >= threshold).astype(int)

        threshold_rows.append({
            "Threshold": round(float(threshold), 3),
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_val, predictions, zero_division=0
            ),
        })

    threshold_results = pd.DataFrame(threshold_rows)

    best_row = threshold_results.loc[
        threshold_results["F1"].idxmax()
    ]
    best_threshold = float(best_row["Threshold"])

    # Final evaluation on the held-out test data.
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_prob >= best_threshold).astype(int)

    final_metrics = metric_values(
        y_test, test_predictions, test_prob
    )

    cm = confusion_matrix(y_test, test_predictions)

    report = classification_report(
        y_test,
        test_predictions,
        target_names=["No Default", "Default"],
        output_dict=True,
        zero_division=0,
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(y_test, test_prob)
    )

    return {
        "rows": len(data),
        "features": X.shape[1],
        "class_counts": y.value_counts().sort_index(),
        "validation": validation_results,
        "thresholds": threshold_results,
        "selected_model": best_name,
        "selected_threshold": best_threshold,
        "final_metrics": final_metrics,
        "confusion_matrix": cm,
        "classification_report": pd.DataFrame(report).T,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "test_size": len(y_test),
    }


# ==================================================
# SIDEBAR: DATASET SELECTION
# ==================================================

st.sidebar.header("🗂️ Dataset Lab")

dataset_choice = st.sidebar.radio(
    "Choose a data source",
    ["Built-in credit default dataset", "Upload your own CSV"],
)

data = None
target_default = None

try:
    if dataset_choice == "Built-in credit default dataset":
        data, target_default = load_builtin_dataset()
        st.sidebar.success("Built-in dataset ready.")

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV file",
            type=["csv"],
        )

        if uploaded_file is not None:
            data = load_custom_dataset(uploaded_file)

except Exception as error:
    st.error(f"Unable to load dataset: {error}")

if data is not None:
    data = data.loc[:, ~data.columns.duplicated()]

    st.subheader("1. Preview your dataset")

    st.write(
        f"**Rows:** {len(data):,}  |  "
        f"**Columns:** {len(data.columns)}"
    )

    st.dataframe(data.head(10), use_container_width=True)

    missing_count = int(data.isna().sum().sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(data):,}")
    col2.metric("Features + target", len(data.columns))
    col3.metric("Missing cells", f"{missing_count:,}")

    st.subheader("2. Configure the experiment")

    default_index = (
        list(data.columns).index(target_default)
        if target_default in data.columns
        else len(data.columns) - 1
    )

    target_column = st.selectbox(
        "Select the target column",
        options=list(data.columns),
        index=default_index,
    )

    available_classes = (
        data[target_column].dropna().unique().tolist()
    )

    if len(available_classes) == 2:
        minority_class = (
            data[target_column].value_counts().idxmin()
        )
        positive_index = available_classes.index(minority_class)

        positive_class = st.selectbox(
            "Which class represents the positive outcome?",
            options=available_classes,
            index=positive_index,
            format_func=str,
        )

        st.caption(
            f"The selected positive class, {positive_class!s}, "
            "will be encoded as 1. The other class will be 0."
        )

        st.subheader("3. Class distribution")

        distribution = (
            data[target_column]
            .value_counts(dropna=False)
            .rename_axis("Class")
            .reset_index(name="Count")
        )

        st.bar_chart(distribution.set_index("Class"))

        st.subheader("4. Run the experiment")

        st.info(
            "Four model configurations will be trained. "
            "SMOTE is applied to training data only. "
            "Model and threshold selection use validation data."
        )

        if st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Training models and evaluating results..."
                ):
                    st.session_state["analysis_results"] = (
                        run_experiment(
                            data,
                            target_column,
                            positive_class,
                        )
                    )

                    st.session_state["analysis_dataset"] = (
                        "Built-in credit default dataset"
                        if dataset_choice
                        == "Built-in credit default dataset"
                        else uploaded_file.name
                    )

                    st.session_state["analysis_target"] = (
                        target_column
                    )

                st.success("Analysis completed successfully!")

            except Exception as error:
                st.session_state.pop("analysis_results", None)
                st.error(f"Analysis failed: {error}")

    else:
        st.warning(
            "The selected target must contain exactly two "
            "non-missing classes. Choose another target column."
        )

else:
    st.info(
        "Choose the built-in dataset or upload a CSV file "
        "using the sidebar to begin."
    )


# ==================================================
# RESULTS DASHBOARD
# ==================================================

results = st.session_state.get("analysis_results")

if results is not None:
    st.divider()
    st.header("📈 Experiment Results")

    st.caption(
        f"Dataset: {st.session_state.get('analysis_dataset', 'Selected dataset')} "
        f"| Target: {st.session_state.get('analysis_target', '')}"
    )

    tabs = st.tabs([
        "Overview",
        "Model Comparison",
        "Final Evaluation",
        "Threshold Analysis",
        "Research Findings",
    ])

    with tabs[0]:
        st.subheader("Dataset Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows analyzed", f"{results['rows']:,}")
        c2.metric("Input features", results["features"])
        c3.metric("Final test records", results["test_size"])

        st.write("Target class counts (0 = other, 1 = positive):")
        st.bar_chart(results["class_counts"])

        st.write("**Data split:** 60% training, 20% validation, 20% test.")

    with tabs[1]:
        st.subheader("Validation Model Comparison")

        st.dataframe(
            results["validation"].style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
            }),
            use_container_width=True,
        )

        chart_data = (
            results["validation"]
            .set_index("Model")[
                ["Precision", "Recall", "F1", "PR-AUC"]
            ]
        )

        st.bar_chart(chart_data)

        st.success(
            f"Selected model by validation PR-AUC: "
            f"{results['selected_model']}"
        )

    with tabs[2]:
        st.subheader("Held-Out Test Evaluation")

        metrics = results["final_metrics"]

        columns = st.columns(3)
        for column, metric in zip(
            columns,
            ["Accuracy", "Precision", "Recall"],
        ):
            column.metric(metric, f"{metrics[metric]:.2%}")

        columns = st.columns(3)
        for column, metric in zip(
            columns,
            ["F1", "ROC-AUC", "PR-AUC"],
        ):
            column.metric(metric, f"{metrics[metric]:.3f}")

        st.write(
            f"**Selected threshold:** "
            f"{results['selected_threshold']:.3f}"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=results["confusion_matrix"],
            display_labels=["Other (0)", "Positive (1)"],
        ).plot(ax=ax, colorbar=False, values_format="d")

        ax.set_title("Final Test Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")
        st.dataframe(
            results["classification_report"].round(3),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            results["recall_curve"],
            results["precision_curve"],
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Test Precision-Recall Curve")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[3]:
        st.subheader("Validation Threshold Explorer")

        threshold_df = results["thresholds"]

        chosen_threshold = st.slider(
            "Explore a decision threshold",
            min_value=float(threshold_df["Threshold"].min()),
            max_value=float(threshold_df["Threshold"].max()),
            value=float(results["selected_threshold"]),
            step=0.025,
            key="threshold_explorer",
        )

        nearest = (
            threshold_df["Threshold"] - chosen_threshold
        ).abs().idxmin()

        row = threshold_df.loc[nearest]

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{row['Precision']:.2%}")
        c2.metric("Recall", f"{row['Recall']:.2%}")
        c3.metric("F1-score", f"{row['F1']:.2%}")

        st.line_chart(
            threshold_df.set_index("Threshold")[
                ["Precision", "Recall", "F1"]
            ]
        )

        st.caption(
            "The slider explores validation metrics. "
            "It does not retrain the model or change the final test result."
        )

    with tabs[4]:
        st.subheader("Research Findings")

        st.write(
            f"The model selected using validation average precision "
            f"was **{results['selected_model']}**."
        )

        st.write(
            f"The threshold selected using validation F1 was "
            f"**{results['selected_threshold']:.3f}**."
        )

        st.write(
            f"On the held-out test set, default/positive-class recall "
            f"was **{results['final_metrics']['Recall']:.1%}**, "
            f"and precision was "
            f"**{results['final_metrics']['Precision']:.1%}**."
        )

        st.markdown(
            """
            **Interpretation**

            - Precision measures how many predicted positive cases were correct.
            - Recall measures how many actual positive cases were detected.
            - F1 summarizes the precision-recall balance.
            - ROC-AUC and PR-AUC assess model ranking across thresholds.
            - The appropriate operating threshold depends on the cost of false
              positives versus missed positive cases.

            **Limitations:** Results depend on the dataset, split, and model
            settings. This research dashboard is not a validated system for
            real-world credit or other high-stakes decisions.
            """
        )

    st.subheader("Download Results")

    validation_csv = results["validation"].to_csv(index=False).encode("utf-8")
    threshold_csv = results["thresholds"].to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)

    c1.download_button(
        "Download model comparison CSV",
        data=validation_csv,
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    c2.download_button(
        "Download threshold results CSV",
        data=threshold_csv,
        file_name="threshold_comparison.csv",
        mime="text/csv",
    )

else:
    st.divider()
    st.info(
        "Your results will appear here after you select a dataset "
        "and click **Run Analysis**."
    )

st.sidebar.divider()
st.sidebar.caption("Python • scikit-learn • imbalanced-learn • Streamlit")
