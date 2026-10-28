
# Research-Grade ML: Handling Imbalanced Data

A reproducible machine-learning study comparing methods for
credit-card default classification on imbalanced data.

## Research Question

Can we detect more default cases while balancing precision
and the number of false-positive predictions?

## Dataset

UCI Default of Credit Card Clients:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains 30,000 records and 23 input features.

## Methods

- Logistic Regression
- Class-weighted Logistic Regression
- Random Forest
- Random Forest with SMOTE
- Validation-based model and threshold selection

SMOTE is applied to the training subset only.

## Evaluation

- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision (PR-AUC)
- Confusion matrix

The final model is selected using validation data. The
held-out test set is used for the final evaluation.

## Setup

Create and activate a virtual environment, then install
the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the experiment:

```bash
python main.py
```

The program downloads the dataset and saves metrics,
reports, and visualizations in the `results/` folder.

## Limitations

Results depend on the dataset, split, model settings,
and classification threshold. The experiment is educational
and is not a validated credit-risk decision system.

## Future Work

- Interactive Streamlit dashboard
- Cross-validation and hyperparameter tuning
- Threshold selection based on explicit business costs
- Model interpretation and feature importance
