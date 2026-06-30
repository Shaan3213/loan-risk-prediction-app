import streamlit as st
import pandas as pd
import joblib

# Load saved model files
model = joblib.load("loan_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Configure page
st.set_page_config(
    page_title="Loan Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Loan Risk Prediction System")

st.write(
    "This application predicts whether a customer belongs to the **Low Risk** or **High Risk** loan category using the trained Logistic Regression model."
)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("🏦 Banking Analytics Project")

st.sidebar.markdown("---")

st.sidebar.subheader("Project Information")

st.sidebar.write("""
**Model:** Logistic Regression

**Prediction:** Loan Risk Classification

**Dataset:** Czechoslovakia Banking Dataset

**Developer:** Mohd Shaan Saifi
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Prediction Classes")

st.sidebar.success("🟢 Low Risk")

st.sidebar.error("🔴 High Risk")

st.sidebar.markdown("---")

st.sidebar.info(
    "This application predicts the likelihood of a customer belonging to the High Risk or Low Risk loan category based on customer, account, and transaction information."
)

# ==========================================
# Loan Information
# ==========================================

with st.expander("📄 Loan Information", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=100000,
            step=1000
        )

    with col2:
        loan_duration = st.number_input(
            "Loan Duration (Months)",
            min_value=1,
            value=36,
            step=1
        )

    with col3:
        monthly_payment = st.number_input(
            "Monthly Payment",
            min_value=0.0,
            value=3000.0,
            step=100.0
        )

   # ==========================================
# Customer Information
# ==========================================

with st.expander("👤 Customer Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

    with col2:
        customer_gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

# ==========================================
# Account Information
# ==========================================

with st.expander("🏦 Account Information", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        statement_frequency = st.selectbox(
            "Statement Frequency",
            ["Weekly Issuance", "Monthly Issuance"]
        )

    with col2:
        account_type = st.selectbox(
            "Account Type",
            ["Salary Account", "Savings Account"]
        )

    with col3:
        account_age_days = st.number_input(
            "Account Age (Days)",
            min_value=0,
            value=365
        )

    loan_year = st.number_input(
        "Loan Year",
        min_value=1993,
        max_value=2025,
        value=1993
    )

# ==========================================
# Transaction Information
# ==========================================

with st.expander("📊 Transaction Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        total_transactions = st.number_input(
            "Total Transactions",
            min_value=0,
            value=150
        )

        total_transaction_amount = st.number_input(
            "Total Transaction Amount",
            min_value=0.0,
            value=250000.0
        )

        average_transaction_amount = st.number_input(
            "Average Transaction Amount",
            min_value=0.0,
            value=1800.0
        )

    with col2:

        average_balance = st.number_input(
            "Average Balance",
            min_value=0.0,
            value=25000.0
        )

        maximum_balance = st.number_input(
            "Maximum Balance",
            min_value=0.0,
            value=45000.0
        )

        minimum_balance = st.number_input(
            "Minimum Balance",
            min_value=0.0,
            value=5000.0
        )

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Loan Risk",
    use_container_width=True
)

if predict_button:

    # Encode categorical variables
    statement_frequency = 1 if statement_frequency == "Monthly Issuance" else 0
    account_type = 1 if account_type == "Savings Account" else 0
    customer_gender = 1 if customer_gender == "Female" else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        "loan_amount": [loan_amount],
        "loan_duration": [loan_duration],
        "monthly_payment": [monthly_payment],
        "statement_frequency": [statement_frequency],
        "Account_type": [account_type],
        "customer_gender": [customer_gender],
        "total_transactions": [total_transactions],
        "total_transaction_amount": [total_transaction_amount],
        "average_transaction_amount": [average_transaction_amount],
        "average_balance": [average_balance],
        "maximum_balance": [maximum_balance],
        "minimum_balance": [minimum_balance],
        "customer_age": [customer_age],
        "account_age_days": [account_age_days],
        "loan_year": [loan_year]
    })

    # Arrange columns in the same order used during training
    input_data = input_data[feature_columns]

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict class
    prediction = model.predict(input_scaled)[0]

    # Predict probability
    probability = model.predict_proba(input_scaled)[0]

    st.write("Prediction Probabilities:", probability)
  
    confidence = max(probability) * 100



    st.markdown("---")

    st.header("🎯 Prediction Result")

    if prediction == 0:

        st.success("🟢 Low Risk Customer")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.info("""
### Recommendation

✅ Eligible for standard loan approval.

The customer exhibits a low probability of loan default based on historical banking behaviour.
""")

    else:

        st.error("🔴 High Risk Customer")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.warning("""
### Recommendation

⚠️ Manual credit review recommended.

The customer demonstrates characteristics associated with higher loan repayment risk.
""")