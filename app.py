import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("rf_fraud_model.pkl")

# App title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
This app lets you:
- 📥 Predict fraud from individual transaction input
- 📂 Upload CSVs for batch prediction
- 📊 View a live dashboard of prediction stats
""")

# Sidebar navigation
mode = st.sidebar.selectbox("Choose Mode", ["🔍 Single Transaction", "📂 Batch Prediction", "📊 Live Dashboard"])


# Single Transaction

if mode == "🔍 Single Transaction":
    st.header("🔍 Single Transaction Fraud Check")
    with st.form("fraud_form"):
        V_features = {}
        for i in range(1, 29):
            V_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
        amount = st.number_input("Amount", value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[*V_features.values(), amount]], columns=[*V_features.keys(), "Amount"])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("🔎 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
        else:
            st.success(f"✅ Not Fraud. Probability: {probability:.2f}")


# Batch Prediction

elif mode == "📂 Batch Prediction":
    st.header("📂 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file with V1–V28 and Amount columns", type=["csv"])

    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview", df_uploaded.head())

        # Drop unnecessary columns if present
        for col in ['Time', 'Class']:
            if col in df_uploaded.columns:
                df_uploaded.drop(columns=[col], inplace=True)

        # Align column order
        expected_features = [f"V{i}" for i in range(1, 29)] + ['Amount']
        df_uploaded = df_uploaded[expected_features]

        # Predict
        predictions = model.predict(df_uploaded)
        probabilities = model.predict_proba(df_uploaded)[:, 1]

        df_uploaded['Fraud_Predicted'] = predictions
        df_uploaded['Fraud_Probability'] = probabilities

        st.success("✅ Prediction Complete")
        st.write(df_uploaded.head())

        # Download
        csv = df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, file_name="fraud_predictions.csv")


#  Live Monitoring

elif mode == "📊 Live Dashboard":
    st.header("📊 Live Fraud Monitoring Dashboard (Default is for the Demo Dataset Trained Model)")

    # Load demo dataset (or real-time data)
    df = pd.read_csv("creditcard.csv")
    df = df.drop_duplicates()
    normal = df[df['Class'] == 0]
    fraud = df[df['Class'] == 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("🔍 Total Transactions", len(df))
    col2.metric("⚠️ Fraud Cases", len(fraud))
    col3.metric("✅ Normal Cases", len(normal))

    st.subheader("📈 Fraud Distribution Over Time")
    df['Hour'] = df['Time'] / 3600
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x='Hour', hue='Class', bins=48, palette={0: 'green', 1: 'red'}, ax=ax1)
    plt.xlabel("Hour of Transaction")
    st.pyplot(fig1)

    st.subheader("💵 Transaction Amount Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df, x='Class', y='Amount', palette='Set2', ax=ax2)
    plt.xticks([0, 1], ["Not Fraud", "Fraud"])
    st.pyplot(fig2)
