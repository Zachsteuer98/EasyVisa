"""Training and prediction helpers for the campaign response demo."""

from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
SELECTED_MODEL_NAME = "Cost complexity pruned decision tree"
SELECTION_METRIC = "precision"
SELECTED_CCP_ALPHA = 0.003

RAW_FEATURES = [
    "Age",
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Securities_Account",
    "CD_Account",
    "Online",
    "CreditCard",
]
CATEGORICAL_FEATURES = [
    "Education",
    "Securities_Account",
    "CD_Account",
    "Online",
    "CreditCard",
    "Has_Mortgage",
]


def prepare_features(raw: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Apply the same feature engineering and one-hot encoding as model selection."""
    features = raw[RAW_FEATURES].copy()
    features["Has_Mortgage"] = (features["Mortgage"] > 0).astype(int)
    features = pd.get_dummies(
        features,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    return features.reindex(columns=feature_columns, fill_value=0).astype(float)


def train_selected_model(csv_path: str | Path) -> tuple[DecisionTreeClassifier, list[str]]:
    """Fit the selected precision-focused, cost-complexity-pruned tree on all rows."""
    data = pd.read_csv(csv_path)
    raw = data[RAW_FEATURES].copy()
    X = raw.copy()
    X["Has_Mortgage"] = (X["Mortgage"] > 0).astype(int)
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    y = data["Personal_Loan"].astype(int)
    model = DecisionTreeClassifier(
        ccp_alpha=SELECTED_CCP_ALPHA,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model, list(X.columns)
