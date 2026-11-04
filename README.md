---
title: Personal Loan Campaign
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.51.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Personal Loan Campaign

A static web application hosted on Hugging Face Spaces.

# Personal Loan Campaign Response

A project analyzing historical bank customer data and comparing decision tree models to estimate which existing customers are more likely to accept a personal loan campaign offer, helping prioritize marketing outreach.

## Project question
Which existing bank customers are more likely to accept a personal loan campaign offer, and how can a response prediction help prioritize future outreach?

**The model predicts historical campaign acceptance. It does not determine creditworthiness, loan repayment risk, or loan approval.**

## Run the notebook

1. Create and activate a Python environment.
2. Install dependencies: `python -m pip install -r requirements-notebook.txt`
3. Place `Loan_Modelling(1).csv` beside `personal_loan_campaign_model_selection.ipynb`.
4. Open the notebook and run all cells from top to bottom.

The notebook uses stratified cross-validation on the training split to compare baseline, precision-focused, recall-focused, and cost complexity pruned decision trees. Set `SELECTION_METRIC` to `precision`, `recall`, or `f1` to choose what “best” means for your campaign. The held-out test split is reserved for final evaluation after model selection.

## Files

- `personal_loan_campaign_model_selection.ipynb` — data checks, exploration, model selection, evaluation, and limitations.
- `app.py` — public-facing Streamlit demo interface.
- `model.py` — shared feature preparation and selected model definition.
- `train_model.py` — retrains and saves the selected model artifact.
- `artifacts/campaign_response_model.joblib` — fitted model used by the web app.
- `artifacts/campaign_response_model.json` — model selection and training metadata.
- `Loan_Modelling(1).csv` — dataset used by the notebook.
- `requirements.txt` — pinned dependencies for the web app.
- `requirements-notebook.txt` — extra notebook and plotting dependencies.

## Run the website locally

1. Install dependencies with `python -m pip install -r requirements.txt`.
2. Build the selected model artifact with `python train_model.py`.
3. Launch the web app with `streamlit run app.py`.

The app uses the cost complexity pruned decision tree selected by cross-validated precision in the notebook. It refits that selected model configuration on all rows in the historical dataset for the demo. The form predicts whether an existing customer is likely to accept a personal loan offer, supporting campaign outreach prioritization; it does not make a lending decision.

## Publish the website

The app is structured for Streamlit Community Cloud. Push this project to a GitHub repository with `app.py`, `requirements.txt`, and `artifacts/campaign_response_model.joblib` included. In Community Cloud, create an app from that repository and set the entrypoint to `app.py`. Regenerate the model artifact from the bundled CSV with `python train_model.py` before publishing if you retrain it.

## Limitations

The data describe a historical campaign and may not reflect future customers or campaigns. Model metrics should not be interpreted as evidence of loan repayment performance or causal customer behavior. Validate against a later campaign and evaluate campaign lift and contact economics before operational use.
