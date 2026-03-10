import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline
pipeline = joblib.load('telecom_churn_pipeline.pkl')

# Page config
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Telecom Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they are likely to churn.")
st.divider()

# Input sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Info")
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

with col2:
    st.subheader("Subscription Details")
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Billing")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

with col4:
    st.subheader("Add-on Services")
    st.markdown("How many of these does the customer have?")
    st.markdown("*(Online Security, Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies)*")
    num_addons = st.slider("Number of Add-ons", min_value=0, max_value=6, value=0)

st.divider()

# Predict button
if st.button("🔍 Predict Churn", use_container_width=True):

    # Build input dataframe
    input_data = pd.DataFrame({
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'num_addons': [num_addons]
    })

    # Make prediction
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.divider()

    # Display results
    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**")
        st.metric(label="Churn Probability", value=f"{probability:.0%}")
        st.markdown("**Recommended Action:** Reach out to this customer with a retention offer — a contract upgrade discount or loyalty reward could help keep them.")
    else:
        st.success(f"✅ This customer is **unlikely to churn**")
        st.metric(label="Churn Probability", value=f"{probability:.0%}")
        st.markdown("**Recommended Action:** This customer appears stable. Continue monitoring their usage and satisfaction.")

    # Show probability bar
    st.progress(probability)
    st.caption(f"Churn probability: {probability:.2%}")