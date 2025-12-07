import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ai import train_model
from ai.dataset_generator import generate_dataset
from config import (
    IRRIGATION_LOG_FILE,
    PUMP_FLOW_RATE_LPM,
    CROPS,
    DEFAULT_AREA_BY_CROP,
)

st.set_page_config(
    page_title="Smart AgroSense – AI Irrigation (No Growth Stage)",
    layout="wide",
    page_icon="🌾",
)

# ---------- Helper functions ----------

def log_recommendation(record: dict):
    os.makedirs(os.path.dirname(IRRIGATION_LOG_FILE), exist_ok=True)
    df_row = pd.DataFrame([record])
    try:
        if os.path.exists(IRRIGATION_LOG_FILE):
            df_row.to_csv(IRRIGATION_LOG_FILE, mode="a", header=False, index=False)
        else:
            df_row.to_csv(IRRIGATION_LOG_FILE, mode="w", header=True, index=False)
    except Exception as e:
        st.error(f"Failed to save log: {e}")


def load_logs() -> pd.DataFrame:
    if not os.path.exists(IRRIGATION_LOG_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(IRRIGATION_LOG_FILE)
    except Exception:
        return pd.DataFrame()


# ---------- Sidebar actions ----------

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Train the model and then request irrigation advice.")

    if st.button("🔁 Train / Retrain Irrigation Model", use_container_width=True):
        try:
            with st.spinner("Training model on realistic crop data..."):
                info = train_model.train_and_save()
            st.session_state["model_info"] = info
            st.success(f"Model trained. R² score: {info['r2']:.3f}")
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.markdown("---")
    st.caption("Smart AgroSense • AI-driven irrigation advisor")


# ---------- Tabs layout ----------

tab_overview, tab_train, tab_advisor, tab_multi, tab_history = st.tabs(
    [
        "📘 Overview",
        "🧠 Train & Explain AI",
        "💧 Irrigation Advisor",
        "🧩 Multi-Plot Field Planner",
        "📊 History & Trends",
    ]
)

# ---------- Overview tab ----------

with tab_overview:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## 🌾 Smart AgroSense – AI-Driven Smart Irrigation")
        st.markdown(
            """

            **Smart AgroSense** is an AI-powered irrigation advisor.

            It predicts:

            - How much water crops need *(litres per square meter)*  
            - Total water needed for a field  
            - How long to keep the irrigation pump ON  

            It focuses on:
            - 💧 Saving water  
            - 🌍 Sustainable agriculture  
            - 🧠 Explainable AI (which factors affect water demand)

            **Input boundaries used:**
            - Soil moisture: 5–95 %
            - Temperature: 10–45 °C
            - Humidity: 20–100 %
            """
        )
    with col2:
        st.metric("Pump flow rate", f"{PUMP_FLOW_RATE_LPM:.0f} L/min")
        st.metric("Supported crops", len(CROPS))
        st.info("Train the model from the **Train & Explain AI** tab to get started.")


# ---------- Train & Explain AI tab ----------

with tab_train:
    st.markdown("## 🧠 Train & Explain the Irrigation Model")

    if "model_info" not in st.session_state:
        st.info("No trained model in this session yet. Use the sidebar to train it.")
    else:
        info = st.session_state["model_info"]
        st.success(f"Current model R² on test data: {info['r2']:.3f}")

        st.markdown("### 📄 Sample of training dataset")
        try:
            df_sample = generate_dataset(n_samples=30)
            st.dataframe(df_sample, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to generate sample dataset: {e}")

        st.markdown("### 🔍 Which parameters influence water requirement the most?")
        try:
            feature_names = info["feature_names"]
            importances = info["importances"]
            order = np.argsort(importances)[::-1]

            fig, ax = plt.subplots()
            ax.barh(
                [feature_names[i] for i in order][::-1],
                importances[order][::-1],
            )
            ax.set_xlabel("Relative importance")
            ax.set_ylabel("Feature")
            ax.set_title("Feature importance (Random Forest)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not display feature importances: {e}")


# ---------- Single-plot Irrigation Advisor tab ----------

with tab_advisor:
    st.markdown("## 💧 AI Irrigation Advisor (Single Plot)")

    if "model_info" not in st.session_state:
        st.warning("Please train the model first using the sidebar.")
    else:
        info = st.session_state["model_info"]
        pipe = info["pipeline"]

        col_left, col_right = st.columns(2)

        with col_left:
            crop = st.selectbox("Crop type", CROPS, index=0)
            area = st.number_input(
                "Field area (m²)",
                min_value=10.0,
                max_value=5000.0,
                value=DEFAULT_AREA_BY_CROP.get(crop, 100.0),
                step=10.0,
            )

        with col_right:
            soil_m = st.slider("Soil moisture (%)", 5.0, 95.0, 30.0, 1.0)
            temp = st.slider("Temperature (°C)", 10.0, 45.0, 32.0, 0.5)
            hum = st.slider("Relative humidity (%)", 20.0, 100.0, 50.0, 1.0)

        if st.button("✨ Get AI Recommendation", type="primary"):
            X = pd.DataFrame(
                [{
                    "crop": crop,
                    "soil_moisture": soil_m,
                    "temperature": temp,
                    "humidity": hum,
                }]
            )
            try:
                water_lpm2 = float(pipe.predict(X)[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                water_lpm2 = 0.0

            water_lpm2 = max(0.0, water_lpm2)

            total_liters = water_lpm2 * area
            pump_minutes = total_liters / PUMP_FLOW_RATE_LPM if PUMP_FLOW_RATE_LPM > 0 else 0.0

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Water needed", f"{water_lpm2:.2f} L/m²")
            with col_b:
                st.metric("Total water", f"{total_liters:.0f} L")
            with col_c:
                st.metric("Pump ON time", f"{pump_minutes:.1f} min")

            if water_lpm2 == 0:
                st.success("Soil seems sufficiently wet; no irrigation required.")
            elif water_lpm2 < 4:
                st.info("Light irrigation suggested.")
            elif water_lpm2 < 8:
                st.warning("Moderate irrigation suggested.")
            else:
                st.error("Heavy irrigation required. Soil is likely very dry/hot.")

            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "crop": crop,
                "area_m2": area,
                "soil_moisture": soil_m,
                "temperature": temp,
                "humidity": hum,
                "water_lpm2": water_lpm2,
                "total_liters": total_liters,
                "pump_minutes": pump_minutes,
            }
            log_recommendation(record)
            st.caption("This recommendation has been saved to the history log.")


# ---------- Multi-Plot Field Planner tab ----------

with tab_multi:
    st.markdown("## 🧩 Multi-Plot Field Planner (Multiple Crops in One Land)")

    if "model_info" not in st.session_state:
        st.warning("Please train the model first using the sidebar.")
    else:
        info = st.session_state["model_info"]
        pipe = info["pipeline"]

        st.markdown(
            "Plan a field that is divided into multiple sections. "
            "For each section, pick a crop and area. "
            "The system will automatically simulate realistic soil conditions "
            "and compute irrigation for each section."
        )

        n_sections = st.slider(
            "How many crop sections in your field?",
            min_value=1,
            max_value=4,
            value=3,
            step=1,
        )

        sections = []
        for i in range(n_sections):
            st.markdown(f"### Section {i + 1}")
            col1, col2 = st.columns(2)
            with col1:
                crop_i = st.selectbox(
                    f"Crop in section {i + 1}",
                    CROPS,
                    index=min(i, len(CROPS) - 1),
                    key=f"multi_crop_{i}",
                )
            with col2:
                area_i = st.number_input(
                    f"Area of section {i + 1} (m²)",
                    min_value=10.0,
                    max_value=5000.0,
                    value=DEFAULT_AREA_BY_CROP.get(crop_i, 100.0),
                    step=10.0,
                    key=f"multi_area_{i}",
                )
            sections.append({"crop": crop_i, "area": area_i})

        if st.button("🚜 Run Multi-Section Simulation", type="primary"):
            rng = np.random.default_rng()
            rows_for_model = []
            sim_details = []

            for idx, sec in enumerate(sections):
                soil_m = float(rng.uniform(20, 80))
                temp = float(rng.uniform(18, 38))
                hum = float(rng.uniform(30, 90))

                rows_for_model.append(
                    {
                        "crop": sec["crop"],
                        "soil_moisture": soil_m,
                        "temperature": temp,
                        "humidity": hum,
                    }
                )
                sim_details.append(
                    {
                        "section": idx + 1,
                        "crop": sec["crop"],
                        "area_m2": sec["area"],
                        "soil_moisture": soil_m,
                        "temperature": temp,
                        "humidity": hum,
                    }
                )

            X_multi = pd.DataFrame(rows_for_model)
            try:
                water_lpm2_all = pipe.predict(X_multi)
            except Exception as e:
                st.error(f"Prediction failed for multi-section: {e}")
                water_lpm2_all = [0.0] * len(sim_details)

            results = []
            for detail, w_lpm2 in zip(sim_details, water_lpm2_all):
                w_lpm2 = max(0.0, float(w_lpm2))
                total_liters = w_lpm2 * detail["area_m2"]
                pump_minutes = (
                    total_liters / PUMP_FLOW_RATE_LPM if PUMP_FLOW_RATE_LPM > 0 else 0.0
                )

                row = {
                    **detail,
                    "water_lpm2": w_lpm2,
                    "total_liters": total_liters,
                    "pump_minutes": pump_minutes,
                }
                results.append(row)

                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **row,
                }
                log_recommendation(record)

            df_results = pd.DataFrame(results)
            st.markdown("### 🔢 Irrigation plan for each section")
            st.dataframe(df_results, use_container_width=True)

            st.markdown("### 💧 Total water requirement per section")
            st.bar_chart(
                df_results.set_index("section")["total_liters"],
            )

            total_field_water = df_results["total_liters"].sum()
            st.success(f"Total water required for entire field: {total_field_water:.0f} L")


# ---------- History & Trends tab ----------

with tab_history:
    st.markdown("## 📊 History & Trends")

    logs = load_logs()
    if logs.empty:
        st.info("No irrigation history yet. Generate some recommendations first.")
    else:
        st.markdown("### Recent recommendations")
        st.dataframe(logs.tail(50), use_container_width=True)

        if "soil_moisture" in logs.columns and "crop" in logs.columns:
            st.markdown("### Average soil moisture by crop")
            avg_moisture = logs.groupby("crop")["soil_moisture"].mean().reset_index()
            st.bar_chart(avg_moisture.set_index("crop"))
        else:
            st.info("Not enough data to compute average soil moisture.")

        if "timestamp" in logs.columns and "total_liters" in logs.columns:
            st.markdown("### Water usage over time")
            logs_sorted = logs.sort_values("timestamp")
            fig2, ax2 = plt.subplots()
            ax2.plot(logs_sorted["timestamp"], logs_sorted["total_liters"], marker="o")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Total water (L)")
            ax2.set_title("Irrigation water usage over time")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig2)
        else:
            st.info("Not enough data to show water usage over time.")
