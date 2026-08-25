import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Vessel Delay & Congestion Dashboard")
st.title("🚢 Vessel Delay Risk & Port Congestion Dashboard")
st.caption("A dual-model system combining supervised classification and deep learning forecasting")

# Load models
reg_model = joblib.load("regression_model.pkl")
lstm_model = load_model("lstm_model(1).h5", compile=False)
scaler = joblib.load("scaler.pkl")

tab1, tab2, tab3 = st.tabs(["📍 Risk Prediction", "📈 Congestion Forecast", "📊 Overview"])

# ---------------- TAB 1: Risk Prediction ----------------
with tab1:
    st.header("Current Vessel Delay Risk")

    c1, c2 = st.columns([1, 2])

    with c1:
        sog = st.number_input("Speed Over Ground (SOG)", min_value=0.0, max_value=40.0, value=10.0)
        heading = st.number_input("Heading (degrees)", min_value=0.0, max_value=360.0, value=90.0)
        predict_clicked = st.button("Predict Risk", type="primary")

    with c2:
        if predict_clicked:
            input_data = pd.DataFrame([[sog, heading]], columns=['SOG', 'Heading'])
            risk = reg_model.predict(input_data)[0]

            probs = None
            if hasattr(reg_model, "predict_proba"):
                probs = reg_model.predict_proba(input_data)[0]
                classes = reg_model.classes_

            st.success(f"Predicted Delay Risk: **{risk}**")

            colA, colB = st.columns(2)

            # Pie chart of model confidence across classes
            with colA:
                if probs is not None:
                    fig1, ax1 = plt.subplots()
                    colors_map = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
                    pie_colors = [colors_map.get(c, "#999999") for c in classes]
                    ax1.pie(probs, labels=classes, autopct='%1.1f%%', colors=pie_colors, startangle=90)
                    ax1.set_title("Model Confidence by Risk Category")
                    st.pyplot(fig1)
                else:
                    st.info("Confidence breakdown not available for this model type.")

            # Bar chart highlighting predicted class
            with colB:
                categories = ['Low', 'Medium', 'High']
                bar_vals = [1 if c == risk else 0.3 for c in categories]
                bar_colors = ['#4CAF50' if c == risk else '#D3D3D3' for c in categories]
                fig2, ax2 = plt.subplots()
                ax2.bar(categories, bar_vals, color=bar_colors)
                ax2.set_title("Predicted Category")
                ax2.set_ylim(0, 1.2)
                st.pyplot(fig2)
        else:
            st.info("Enter vessel speed and heading, then click Predict Risk.")

# ---------------- TAB 2: Congestion Forecast ----------------
with tab2:
    st.header("Port Congestion Forecast")
    st.write("Upload a CSV containing an `ETA_hours` column (at least 24 rows) to forecast the next trend value.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        values = data['ETA_hours'].values.reshape(-1, 1)
        scaled = scaler.transform(values)

        window = 24
        if len(scaled) >= window:
            last_seq = scaled[-window:].reshape(1, window, 1)
            pred_scaled = lstm_model.predict(last_seq)
            pred = scaler.inverse_transform(pred_scaled)
            forecast_value = pred[0][0]

            st.success(f"Forecasted next ETA trend value: **{forecast_value:.2f} hours**")

            past_values = data['ETA_hours'].values[-window:]
            steps = list(range(1, window + 1))

            colA, colB = st.columns(2)

            # Line chart: trend + forecast point
            with colA:
                fig3, ax3 = plt.subplots()
                ax3.plot(steps, past_values, marker='o', label="Past ETA", color="#2196F3")
                ax3.plot([steps[-1], steps[-1] + 1], [past_values[-1], forecast_value],
                         marker='o', linestyle='--', color="#F44336", label="Forecast")
                ax3.set_title("ETA Trend Over Time")
                ax3.set_xlabel("Time Step")
                ax3.set_ylabel("ETA (hours)")
                ax3.legend()
                st.pyplot(fig3)

            # Area-style chart for a different visual (cumulative view)
            with colB:
                fig4, ax4 = plt.subplots()
                ax4.fill_between(steps, past_values, color="#90CAF9", alpha=0.5)
                ax4.plot(steps, past_values, color="#1565C0")
                ax4.set_title("ETA Trend (Area View)")
                ax4.set_xlabel("Time Step")
                ax4.set_ylabel("ETA (hours)")
                st.pyplot(fig4)
        else:
            st.warning(f"Need at least {window} rows of ETA data to forecast.")
    else:
        st.info("Upload a CSV file to see the congestion forecast and trend charts.")

# ---------------- TAB 3: Overview ----------------
with tab3:
    st.header("Project Overview")
    st.write(
        "This dashboard integrates two predictive components: a supervised regression model "
        "that classifies real-time vessel delay risk, and an LSTM deep learning model that "
        "forecasts short-horizon port congestion trends from historical ETA sequences."
    )

    colA, colB, colC = st.columns(3)
    colA.metric("Model 1", "Logistic Regression", "Delay Risk Classification")
    colB.metric("Model 2", "LSTM", "Congestion Forecasting")
    colC.metric("Dataset", "AIS Vessel Data", "1M+ records")

    st.markdown("---")
    st.write("Built as part of a final-year research project on predictive maritime logistics analytics.")
