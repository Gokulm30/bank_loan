import streamlit as st
import pandas as pd
import joblib

st.title("🏦 Bank Loan Predictor")

# Load scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('Bank_logistic.joblib')

st.subheader("Enter customer details:")

# Input form
age = st.number_input("Age", 18, 100, 30)
experience = st.number_input("Experience (years)", 0, 80, 5)
income = st.number_input("Income (in $1000s)", 0, 500, 60)
family = st.selectbox("Family Members", [1, 2, 3, 4])
ccavg = st.number_input("CCAvg (avg credit card spend)", 0.0, 20.0, 1.5)
education = st.selectbox("Education Level", [1, 2, 3])  # 1=UG, 2=Grad, 3=Adv
mortgage = st.number_input("Mortgage", 0, 1000, 0)
securities = st.selectbox("Securities Account", [0, 1])
cd_account = st.selectbox("CD Account", [0, 1])
online = st.selectbox("Online Banking", [0, 1])
creditcard = st.selectbox("Credit Card", [0, 1])

if st.button("Predict Loan Approval"):
    input_df = pd.DataFrame([[
        age, experience, income, family, ccavg, education,
        mortgage, securities, cd_account, online, creditcard
    ]], columns=['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
                 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard'])
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.success(f"✅ Loan Approved with probability {proba:.2%}")
    else:
        st.error(f"❌ Loan Not Approved with probability {1 - proba:.2%}")
