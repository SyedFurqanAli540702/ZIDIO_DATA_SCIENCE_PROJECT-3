import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

model = joblib.load('fraud_model.pkl')

st.title("🔍 Real-Time Financial Fraud Detection Dashboard")

# Simulate a transaction stream
def simulate_transaction():
    return {
        "amount": np.random.uniform(10, 1000),
        "user_age": np.random.randint(18, 70),
        "transaction_hour": np.random.randint(0, 24),
        "is_international": np.random.randint(0, 2),
        "num_prev_transactions": np.random.randint(1, 100)
    }

# Predict on new transaction
def predict_transaction(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return "Fraud" if prediction == 1 else "Legit"

# Stream updates
placeholder = st.empty()
for _ in range(100):
    transaction = simulate_transaction()
    result = predict_transaction(transaction)
    with placeholder.container():
        st.subheader("🚨 New Transaction Analyzed")
        st.json(transaction)
        st.markdown(f"### Prediction: :red[{result}]")
    time.sleep(5)
