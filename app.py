"""Streamlit demo for the selected personal loan campaign response model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model import RAW_FEATURES, prepare_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "campaign_response_model.joblib"

st.set_page_config(
    page_title="Personal Loan Campaign Response Prediction",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        st.error("Model artifact is missing. Run `python train_model.py` before starting the app.")
        st.stop()
    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()
model = bundle["model"]
feature_columns = bundle["feature_columns"]

st.title("Personal Loan Campaign Response Prediction")
st.write(
    "Estimate whether an existing bank customer is likely to accept a personal loan offer "
    "based on patterns in a historical campaign. Use the estimate to help prioritize "
    "marketing outreach."
)
st.info(
    "This demo predicts campaign response only. It does not assess creditworthiness, "
    "loan repayment risk, or loan approval."
)

with st.form("customer_profile"):
    st.subheader("Customer profile")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=40, step=1)
        income = st.number_input(
            "Annual income (thousands of dollars)",
            min_value=0,
            max_value=500,
            value=75,
            step=1,
            help="For example, enter 75 for $75,000 per year.",
        )
        family = st.selectbox("Family size", options=[1, 2, 3, 4], index=1)
        education_label = st.selectbox(
            "Education",
            options=["Undergraduate", "Graduate", "Advanced / professional"],
        )
        education = {
            "Undergraduate": 1,
            "Graduate": 2,
            "Advanced / professional": 3,
        }[education_label]
    with right:
        ccavg = st.number_input(
            "Average monthly credit card spending (thousands of dollars)",
            min_value=0.0,
            max_value=50.0,
            value=1.5,
            step=0.1,
            help="For example, enter 1.5 for $1,500 per month.",
        )
        mortgage = st.number_input(
            "Mortgage balance (thousands of dollars)",
            min_value=0,
            max_value=700,
            value=0,
            step=1,
            help="Enter 0 if the customer has no mortgage.",
        )
        securities = st.selectbox("Has a securities account", ["No", "Yes"])
        cd_account = st.selectbox("Has a CD account", ["No", "Yes"])
        online = st.selectbox("Uses online banking", ["No", "Yes"])
        credit_card = st.selectbox("Has a credit card from another bank", ["No", "Yes"])

    submitted = st.form_submit_button("Predict campaign response", type="primary")

if submitted:
    applicant = pd.DataFrame(
        [{
            "Age": age,
            "Income": income,
            "Family": family,
            "CCAvg": ccavg,
            "Education": education,
            "Mortgage": mortgage,
            "Securities_Account": int(securities == "Yes"),
            "CD_Account": int(cd_account == "Yes"),
            "Online": int(online == "Yes"),
            "CreditCard": int(credit_card == "Yes"),
        }],
        columns=RAW_FEATURES,
    )
    prepared = prepare_features(applicant, feature_columns)
    prediction = int(model.predict(prepared)[0])
    probability = float(model.predict_proba(prepared)[0, 1])

    st.subheader("Model estimate")
    st.metric("Estimated chance of accepting the offer", f"{probability:.1%}")
    if prediction:
        st.success("The model predicts this customer is likely to accept a personal loan campaign offer.")
    else:
        st.warning("The model does not predict this customer as likely to accept a personal loan campaign offer.")
    st.caption(
        "This is a campaign response estimate from historical data, not a guarantee and "
        "not a measure of loan approval, creditworthiness, or repayment risk."
    )

with st.expander("About this model"):
    st.write(
        "This demo uses the cost complexity pruned decision tree selected by "
        "cross-validated precision in the companion model selection notebook. "
        "The final estimator is refit on the full historical dataset for this demo."
    )
