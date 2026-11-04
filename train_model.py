"""Train and save the campaign response model selected in the notebook."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from model import (
    RANDOM_STATE,
    SELECTED_CCP_ALPHA,
    SELECTED_MODEL_NAME,
    SELECTION_METRIC,
    train_selected_model,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Loan_Modelling(1).csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "campaign_response_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "campaign_response_model.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    model, feature_columns = train_selected_model(DATA_PATH)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_columns},
        MODEL_PATH,
    )
    metadata = {
        "model_name": SELECTED_MODEL_NAME,
        "selection_metric": SELECTION_METRIC,
        "ccp_alpha": SELECTED_CCP_ALPHA,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(pd.read_csv(DATA_PATH))),
        "sklearn_version": sklearn.__version__,
        "target": "Personal_Loan: acceptance of a historical campaign offer",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metadata: {METADATA_PATH.relative_to(ROOT)}")
    print(f"Selected model: {SELECTED_MODEL_NAME} (by {SELECTION_METRIC})")


if __name__ == "__main__":
    main()
