# Smart AgroSense – No Growth Stage, Robust & Enhanced

This project is an end-to-end **AI-based irrigation advisor** with:

- 🌐 A modern **Streamlit** dashboard
- 🧠 A **RandomForest regression** model trained on realistic synthetic crop–soil–weather data  
  (features: crop, soil moisture, temperature, humidity)
- 🧩 **Multi-Plot Field Planner** for multiple crops in one field
- 🧪 **Pytest** tests for CI
- 🤖 **GitHub Actions CI/CD** pipeline
- 🐳 **Docker** support for containerised deployment
- ⚠️ Built-in **error handling** with `try/except` to avoid crashes

## Input boundaries

- Soil moisture: **5–95 %**
- Temperature: **10–45 °C**
- Humidity: **20–100 %**

## Quick start (local)

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
pip install -r requirements.txt
streamlit run app.py
```

Tabs in the UI:

- **Overview** – project summary, boundaries, and key parameters
- **Train & Explain AI** – train model, view sample dataset, feature importance
- **Irrigation Advisor** – single-plot recommendation with detailed metrics
- **Multi-Plot Field Planner** – divide land into 1–4 sections, choose crops and areas, auto-simulated conditions, and irrigation per section
- **History & Trends** – log of all recommendations and visualised trends

## Docker

```bash
docker build -t smart-agrosense-nogrowth:latest .
docker run -p 8501:8501 smart-agrosense-nogrowth:latest
```

Open http://localhost:8501 to use the app from inside Docker.
