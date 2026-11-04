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

# Personal Loan Campaign Response

An applied machine learning project developed through Great Learning. It uses historical bank customer data to study which existing customers accepted a personal loan campaign offer, compares decision tree models, and presents a response estimate in an interactive Streamlit application.

> **Scope:** The model estimates response to a historical marketing offer. It does not assess creditworthiness, repayment risk, or loan eligibility, and it should not be used to make lending decisions.

## Project links

- **Live application:** [Personal Loan Campaign on Hugging Face Spaces](https://huggingface.co/spaces/zachsteuer98/PersonalLoanCampaign)
- **Source code:** [Zachsteuer98/EasyVisa on GitHub](https://github.com/Zachsteuer98/EasyVisa)
- **Analysis notebook:** [Open the notebook in Google Colab](https://colab.research.google.com/github/Zachsteuer98/EasyVisa/blob/main/personal_loan_campaign_model_selection.ipynb)

## Project background

The work began as a guided data science learning project in Google Colab and was then organized to run in a local Python environment. The analysis explores a binary classification problem: predict whether an existing bank customer accepted a prior personal loan offer. The resulting model is demonstrated through a web form so users can explore how customer attributes affect a campaign response estimate.

The dataset contains 5,000 customer records. The target, `Personal_Loan`, records acceptance of the historical offer (`1`) or non-acceptance (`0`). The project is focused on supervised learning, model selection, and communicating the limits of a prediction in a business context.

## What the project covers

- Inspecting data quality, class balance, and relationships among customer attributes.
- Exploring historical response patterns while distinguishing association from causation.
- Preparing predictors and excluding identifiers and redundant fields from modeling.
- Comparing baseline, pre-pruned, and cost-complexity-pruned decision trees.
- Using stratified five-fold cross-validation on the training data to select a model by precision, recall, or F1.
- Reserving a stratified holdout set for final evaluation after model selection.
- Packaging the selected model for an interactive Streamlit application.

The notebook's default selection criterion is precision, which favors a smaller predicted outreach list with a higher share of historical responders. Recall and F1 are also considered because the most useful metric depends on campaign goals and contact costs.

## Learning outcomes

Reviewing this project demonstrates a practical workflow for:

- Framing a business question as a binary classification task.
- Inspecting a dataset and recognizing the effects of class imbalance.
- Preparing features consistently for training and prediction.
- Comparing evaluation metrics and separating model selection from final testing.
- Interpreting tree-based feature importance cautiously.
- Turning a trained model into a small web application and documenting its intended use.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `personal_loan_campaign_model_selection.ipynb` | Data checks, exploratory analysis, cross-validated model selection, holdout evaluation, and limitations. |
| `Loan_Modelling(1).csv` | Historical customer campaign dataset used by the notebook and training script. |
| `app.py` | Streamlit interface for entering a customer profile and viewing a campaign response estimate. |
| `model.py` | Shared feature preparation and selected decision tree configuration. |
| `train_model.py` | Fits the selected model on the bundled dataset and saves its artifacts. |
| `artifacts/campaign_response_model.joblib` | Serialized model bundle loaded by the application. |
| `artifacts/campaign_response_model.json` | Training and model selection metadata. |
| `requirements.txt` | Dependencies for the Streamlit application. |
| `requirements-notebook.txt` | Additional dependencies for notebook analysis and visualizations. |

## Run the notebook

The notebook was developed for Google Colab and is also available to run locally.

1. Open the notebook using the Colab link above, or clone this repository and open `personal_loan_campaign_model_selection.ipynb` in Jupyter.
2. Ensure `Loan_Modelling(1).csv` is available alongside the notebook. In Colab, upload the CSV into the notebook session or mount storage containing it.
3. For a local environment, install the notebook dependencies with `python -m pip install -r requirements-notebook.txt`.
4. Run the notebook cells from top to bottom.

## Run the application locally

```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The application opens at the local URL printed by Streamlit, usually `http://localhost:8501`. The training script reads the bundled CSV and generates the model bundle and metadata in `artifacts/`.

## Deployment

The live demo is hosted as a Streamlit Space on Hugging Face. Its Space configuration is provided by the YAML block at the top of this README; it identifies the Streamlit SDK, Python version, and `app.py` entry point. The app also requires `requirements.txt` and the trained model at `artifacts/campaign_response_model.joblib`.

The GitHub repository includes an Actions workflow intended to sync changes from its `main` branch to the Hugging Face Space. The workflow uses the `HF_TOKEN` GitHub Actions secret for authentication. Keep the Hugging Face Space configuration and required model artifact in place when updating or deploying the app.

## Limitations and responsible use

The records describe responses to a past campaign; behavior, customer populations, and campaign design may differ in future settings. Associations and feature importance do not establish why a customer accepted an offer. Cross-validation and holdout scores estimate performance on data similar to this dataset and are not a guarantee of future results. Before considering operational use, validate with newer campaign data and account for outreach costs, fairness, privacy, and business outcomes. This demo is for educational exploration and campaign response analysis only.
