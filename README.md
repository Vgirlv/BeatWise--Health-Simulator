# 🩺 Beatwise: AI-Powered Health Input Analyzer

**Beatwise** is an AI-driven web app that analyzes basic human vitals like heart rate, glucose, oxygen levels, temperature, and step count. It uses a pre-trained XGBoost model to predict health status and offers nano-level insight simulations through a skin sensor visualizer. Users can also download a personalized health sticker PDF for offline tracking.

---

## 🚀 Features

- 📥 **Manual Input** of Heart Rate, Glucose, Temperature, Oxygen, and Steps  
- 🤖 **Health Risk Prediction** using a trained XGBoost model  
- 🧬 **Nano Sensor Simulation** with interactive skin layer visualization  
- 📊 **Summary of Abnormal Vitals** with interpretability  
- 🪪 **Personalized Health Sticker PDF** generation & download  
- 🌐 **Streamlit-based UI** for fast and beautiful deployment

---

## 🧠 Tech Stack

- **Frontend & UI**: Python + Streamlit + Plotly (for visualization)  
- **Backend & ML**: XGBoost, Scikit-learn, NumPy  
- **PDF Generation**: FPDF  
- **Data Handling**: joblib (for loading `.pkl` models)

---

## 🧠 Who Can Use This?

- **Academic Use**:  
  Ideal for students and educators in health-tech, bioengineering, or data science. VitalSense can be used to demonstrate applied machine learning in healthcare, wearable sensors, and human physiology visualization.
  
- **General Public**:  
  Anyone interested in tracking their vitals in a meaningful way. Upload your readings manually and get instant feedback, explanations, and a downloadable “Nano Health Sticker” report for reference or sharing with a doctor.

---

## 📁 Folder Structure

├── main.py
├── xgboost_health_model.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md



