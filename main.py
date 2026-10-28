
from ucimlrepo import fetch_ucirepo

# Download the UCI Credit Card Default dataset
print("Downloading the credit-default dataset...")

dataset = fetch_ucirepo(id=350)

# Extract features and target
X = dataset.data.features
y = dataset.data.targets

print("\nDataset downloaded successfully!")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFirst five rows of features:")
print(X.head())

print("\nTarget column:")
print(y.head())



# Step 14: Analyze target class distribution

print("\n--- Class Distribution ---")

class_counts = y.iloc[:, 0].value_counts().sort_index()
class_percentages = y.iloc[:, 0].value_counts(
    normalize=True
).sort_index() * 100

print("Class counts:")
print(class_counts)

print("\nClass percentages:")
print(class_percentages.round(2))

print("\nInterpretation:")
print(f"Total clients: {len(y)}")
print(f"Non-default clients: {class_counts.get(0, 0)}")
print(f"Default clients: {class_counts.get(1, 0)}")


# Step 15: Visualize class distribution

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

class_counts.plot(
    kind="bar",
    color=["seagreen", "coral"],
    edgecolor="black"
)

plt.title("Credit Card Default Class Distribution")
plt.xlabel("Class (0 = No Default, 1 = Default)")
plt.ylabel("Number of Clients")
plt.xticks(rotation=0)
plt.tight_layout()

import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/class_distribution.png", dpi=300)
plt.show()

print("\nClass distribution chart saved successfully!")



# Step 16: Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y.iloc[:, 0],
    test_size=0.20,
    random_state=42,
    stratify=y.iloc[:, 0]
)

print("\n--- Train-Test Split ---")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).sort_index().mul(100).round(2))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).sort_index().mul(100).round(2))



# Step 17: Train the baseline Logistic Regression model

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

print("\nTraining baseline Logistic Regression...")

baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
y_prob = baseline_model.predict_proba(X_test)[:, 1]

print("\n--- Baseline Logistic Regression Results ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Step 18: Class-weighted Logistic Regression

weighted_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

print("\nTraining class-weighted Logistic Regression...")

weighted_model.fit(X_train, y_train)

weighted_pred = weighted_model.predict(X_test)
weighted_prob = weighted_model.predict_proba(X_test)[:, 1]

print("\n--- Class-Weighted Logistic Regression Results ---")
print(f"Accuracy:  {accuracy_score(y_test, weighted_pred):.4f}")
print(f"Precision: {precision_score(y_test, weighted_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, weighted_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, weighted_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, weighted_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, weighted_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, weighted_pred))



# Step 19: Train a baseline Random Forest model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining baseline Random Forest...")

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Baseline Random Forest Results ---")
print(f"Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, rf_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, rf_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, rf_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, rf_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))



# Step 20: Random Forest with SMOTE

from imblearn.over_sampling import SMOTE

print("\n--- SMOTE Training Data ---")
print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train, y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

rf_smote_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest with SMOTE...")

rf_smote_model.fit(X_train_smote, y_train_smote)

smote_pred = rf_smote_model.predict(X_test)
smote_prob = rf_smote_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest + SMOTE Results ---")
print(f"Accuracy:  {accuracy_score(y_test, smote_pred):.4f}")
print(f"Precision: {precision_score(y_test, smote_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, smote_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, smote_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, smote_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, smote_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, smote_pred))



# Step 21: Compare all four models

import pandas as pd

results = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    },
    {
        "Model": "Weighted Logistic Regression",
        "Accuracy": accuracy_score(y_test, weighted_pred),
        "Precision": precision_score(y_test, weighted_pred, zero_division=0),
        "Recall": recall_score(y_test, weighted_pred, zero_division=0),
        "F1": f1_score(y_test, weighted_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, weighted_prob),
        "PR-AUC": average_precision_score(y_test, weighted_prob)
    },
    {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred, zero_division=0),
        "Recall": recall_score(y_test, rf_pred, zero_division=0),
        "F1": f1_score(y_test, rf_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, rf_prob),
        "PR-AUC": average_precision_score(y_test, rf_prob)
    },
    {
        "Model": "Random Forest + SMOTE",
        "Accuracy": accuracy_score(y_test, smote_pred),
        "Precision": precision_score(y_test, smote_pred, zero_division=0),
        "Recall": recall_score(y_test, smote_pred, zero_division=0),
        "F1": f1_score(y_test, smote_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, smote_prob),
        "PR-AUC": average_precision_score(y_test, smote_prob)
    }
])

print("\n--- Final Model Comparison ---")
print(results.round(4).to_string(index=False))

results.to_csv("results/model_comparison.csv", index=False)
print("\nModel comparison saved to results/model_comparison.csv")



# Step 22: Visualize model performance

import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

metrics_to_plot = [
    "Precision",
    "Recall",
    "F1",
    "PR-AUC"
]

plot_data = results.set_index("Model")[metrics_to_plot]

ax = plot_data.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.8
)

ax.set_title("Model Comparison on Imbalanced Data")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)

plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModel comparison chart saved successfully!")



# Step 23: Explore Random Forest decision thresholds

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

threshold_results = []

for threshold in thresholds:
    threshold_pred = (rf_prob >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test, threshold_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, threshold_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, threshold_pred, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

print("\n--- Random Forest Threshold Comparison ---")
print(threshold_df.round(4).to_string(index=False))

threshold_df.to_csv(
    "results/threshold_comparison.csv",
    index=False
)

print("\nThreshold comparison saved successfully!")



# Step 24: Visualize precision-recall trade-off

import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="s",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="^",
    label="F1-score"
)

plt.title("Random Forest: Decision Threshold Analysis")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nThreshold analysis chart saved successfully!")



# Step 25: Create a validation set from training data

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\n--- Training and Validation Split ---")
print("Training subset:", X_train_sub.shape)
print("Validation set:", X_val.shape)
print("Final test set:", X_test.shape)

print("\nTraining subset class percentages:")
print(
    y_train_sub.value_counts(normalize=True)
    .sort_index().mul(100).round(2)
)

print("\nValidation class percentages:")
print(
    y_val.value_counts(normalize=True)
    .sort_index().mul(100).round(2)
)



"""
Research-Grade ML: Handling Imbalanced Data
Dataset: UCI Default of Credit Card Clients
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without opening separate windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project setup
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
    }


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("\n[1/6] Downloading UCI dataset...")

dataset = fetch_ucirepo(id=350)

X = dataset.data.features
y = dataset.data.targets.iloc[:, 0].astype(int)

print("Dataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts().sort_index())
print("\nClass percentages:")
print((y.value_counts(normalize=True).sort_index() * 100).round(2))


# --------------------------------------------------
# 3. Create train, validation and test sets
# --------------------------------------------------

print("\n[2/6] Creating stratified data splits...")

# The test set remains untouched during model selection.
X_development, X_test, y_development, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)

X_train, X_val, y_train, y_val = train_test_split(
    X_development,
    y_development,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_development,
)

print("Training subset:", X_train.shape)
print("Validation set:", X_val.shape)
print("Final test set:", X_test.shape)


# --------------------------------------------------
# 4. Train and compare models on validation data
# --------------------------------------------------

print("\n[3/6] Training candidate models...")

candidates = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ]),

    "Weighted Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Random Forest + SMOTE": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

validation_rows = []
fitted_models = {}
validation_probabilities = {}

for name, model in candidates.items():
    print(f"Training: {name}")

    if name == "Random Forest + SMOTE":
        # Resample training data only, never validation or test data.
        sampler = SMOTE(random_state=RANDOM_STATE)
        X_fit, y_fit = sampler.fit_resample(X_train, y_train)
    else:
        X_fit, y_fit = X_train, y_train

    model.fit(X_fit, y_fit)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.50).astype(int)

    row = {"Model": name}
    row.update(calculate_metrics(y_val, val_pred, val_prob))
    validation_rows.append(row)

    fitted_models[name] = model
    validation_probabilities[name] = val_prob

validation_results = pd.DataFrame(validation_rows)
validation_results = validation_results.sort_values(
    "PR-AUC", ascending=False
)

validation_results.to_csv(
    RESULTS_DIR / "validation_model_comparison.csv",
    index=False,
)

print("\nValidation comparison:")
print(validation_results.round(4).to_string(index=False))


# --------------------------------------------------
# 5. Select model and threshold using validation data
# --------------------------------------------------

print("\n[4/6] Selecting model and decision threshold...")

# Select the model by validation average precision (PR-AUC).
# The test set is not used for this decision.
best_name = validation_results.iloc[0]["Model"]
best_model = fitted_models[best_name]
best_val_prob = validation_probabilities[best_name]

print("Selected model:", best_name)

threshold_rows = []

# Explore thresholds on validation data only.
thresholds = np.round(np.arange(0.20, 0.601, 0.025), 3)

for threshold in thresholds:
    val_pred = (best_val_prob >= threshold).astype(int)

    threshold_rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_pred, zero_division=0
        ),
        "F1": f1_score(
            y_val, val_pred, zero_division=0
        ),
    })

threshold_results = pd.DataFrame(threshold_rows)

# Choose the threshold with the highest validation F1-score.
best_row = threshold_results.loc[
    threshold_results["F1"].idxmax()
]
best_threshold = float(best_row["Threshold"])

threshold_results.to_csv(
    RESULTS_DIR / "validation_threshold_comparison.csv",
    index=False,
)

print(f"Selected threshold: {best_threshold:.3f}")
print("Validation metrics at selected threshold:")
print(best_row.to_string())


# --------------------------------------------------
# 6. Final evaluation on the held-out test set
# --------------------------------------------------

print("\n[5/6] Evaluating selected model on test data...")

# Use the model trained on the training subset and the
# threshold selected on validation data. Do not tune on test.
test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

final_metrics = calculate_metrics(y_test, test_pred, test_prob)

print("\nSelected model:", best_name)
print(f"Decision threshold: {best_threshold:.3f}")
print("\nFinal test metrics:")

for metric, value in final_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nFinal classification report:")
print(classification_report(
    y_test, test_pred, zero_division=0
))

cm = confusion_matrix(y_test, test_pred)
print("Final confusion matrix:")
print(cm)

final_results = pd.DataFrame([{
    "Selected Model": best_name,
    "Selected Threshold": best_threshold,
    **final_metrics,
}])

final_results.to_csv(
    RESULTS_DIR / "final_test_metrics.csv",
    index=False,
)

pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"],
).to_csv(RESULTS_DIR / "final_confusion_matrix.csv")


# --------------------------------------------------
# 7. Save final visualizations
# --------------------------------------------------

print("\n[6/6] Saving charts and reports...")

# Class distribution
fig, ax = plt.subplots(figsize=(7, 5))
y.value_counts().sort_index().plot(
    kind="bar",
    ax=ax,
    color=["seagreen", "coral"],
)
ax.set_title("Credit Card Default Class Distribution")
ax.set_xlabel("Class (0 = No Default, 1 = Default)")
ax.set_ylabel("Number of Clients")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=300)
plt.close(fig)

# Validation model comparison
chart_metrics = ["Precision", "Recall", "F1", "PR-AUC"]
chart_data = validation_results.set_index("Model")[chart_metrics]

fig, ax = plt.subplots(figsize=(10, 6))
chart_data.plot(kind="bar", ax=ax)
ax.set_title("Candidate Model Comparison on Validation Data")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=300)
plt.close(fig)

# Validation threshold analysis
fig, ax = plt.subplots(figsize=(8, 5))

for metric in ["Precision", "Recall", "F1"]:
    ax.plot(
        threshold_results["Threshold"],
        threshold_results[metric],
        marker="o",
        label=metric,
    )

ax.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected threshold = {best_threshold:.3f}",
)
ax.set_title("Validation Decision Threshold Analysis")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "threshold_analysis.png", dpi=300)
plt.close(fig)

# Final test confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"],
).plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Final Test Confusion Matrix")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "final_confusion_matrix.png", dpi=300)
plt.close(fig)

# Final test precision-recall curve
precision_curve, recall_curve, _ = precision_recall_curve(
    y_test, test_prob
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall_curve, precision_curve)
ax.set_title("Final Test Precision-Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=300)
plt.close(fig)

print("\nExperiment completed successfully!")
print("Results folder:", RESULTS_DIR)
print("Saved CSV reports and PNG charts.")
