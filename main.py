import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF

# ========== Load ML Model with Error Handling ==========
try:
    model = joblib.load("xgboost_health_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.stop()

# ========== Streamlit UI ==========
st.set_page_config(page_title="BeatWise - Know Your Health", layout="wide")
st.title("🧬BeatWise - Know Your Health ")
st.markdown(" Nano-Wearable Health Input Analyzer : Enter your vitals to view nano-detection logic, alerts, and AI prediction.")

# ========== Initialize Session State ==========
if "last_input" not in st.session_state:
    st.session_state.last_input = {}
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "show_sticker_ui" not in st.session_state:
    st.session_state.show_sticker_ui = False

# ========== Vitals Input ==========
st.subheader("📥 Enter Your Vitals")
heart_rate = st.number_input("💓 Heart Rate (bpm)", 30, 200, 80)
glucose = st.number_input("🧁 Glucose Level (mg/dL)", 40.0, 300.0, 110.0)
temperature = st.number_input("🌡️ Body Temperature (°C)", 30.0, 45.0, 36.8)
oxygen = st.number_input("🫁 Oxygen Level (%)", 70, 100, 98)
steps = st.number_input("👣 Step Count", 0, 30000, 6000)

# ========== Helper Functions ==========
def assess_risk(vital, value):
    if vital == "Heart Rate":
        if value < 60: return "Low", "⚠️", "yellow"
        elif 60 <= value <= 100: return "Normal", "✅", "green"
        else: return "High", "🚨", "red"
    elif vital == "Glucose":
        if value < 70: return "Low", "⚠️", "orange"
        elif 70 <= value <= 140: return "Normal", "✅", "green"
        else: return "High", "🚨", "red"
    elif vital == "Temperature":
        if value < 36.1: return "Low", "❄️", "blue"
        elif 36.1 <= value <= 37.5: return "Normal", "✅", "green"
        else: return "High", "🔥", "red"
    elif vital == "Oxygen":
        if value >= 95: return "Normal", "✅", "green"
        elif 90 <= value < 95: return "Slightly Low", "⚠️", "orange"
        else: return "Low", "🚨", "red"
    elif vital == "Steps":
        if value < 3000: return "Low", "📉", "orange"
        elif 3000 <= value <= 8000: return "Normal", "✅", "green"
        else: return "Active", "🏃‍♀️", "blue"

nano_explain = {
    "Heart Rate": "Your heart beats create tiny pressure changes detected by piezoelectric nanosensors in capillaries.",
    "Glucose": "Glucose is detected through sweat analysis in epidermis layer and enzyme-based sensors in sweat glands.",
    "Temperature": "Nano-thermistors in the epidermis layer measure your body temperature through heat conductivity.",
    "Oxygen": "Nano-optodes in the dermis layer measure blood oxygen levels using light absorption patterns.",
    "Steps": "Motion is detected by nano-accelerometers distributed throughout all skin layers."
}

def predict_risk_with_xgboost(hr, glucose, temp, oxy, steps):
    features = [[hr, glucose, temp, oxy, steps]]
    proba = model.predict_proba(features)[0]
    pred_encoded = np.argmax(proba)
    confidence = proba[pred_encoded]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    return pred_label, confidence

def all_vitals_normal(hr, glucose, temp, oxy, steps):
    return (
        60 <= hr <= 100 and
        70 <= glucose <= 140 and
        36.1 <= temp <= 37.5 and
        oxy >= 95 and
        3000 <= steps <= 8000
    )

def create_skin_diagram(selected_layer=None):
    layers = {
        "Epidermis": {"color": "#FFA07A", "depth": 1, "sensors": ["Temperature", "Glucose"]},
        "Dermis": {"color": "#E9967A", "depth": 2, "sensors": ["Oxygen"]},
        "Sweat Glands": {"color": "#CD5C5C", "depth": 3, "sensors": ["Glucose", "Hydration"]},
        "Capillaries": {"color": "#B22222", "depth": 4, "sensors": ["Heart Rate"]}
    }

    fig = go.Figure(go.Treemap(
        labels=["Skin"] + list(layers.keys()),
        parents=[""] + ["Skin"]*len(layers),
        marker_colors=["#F5D393"] + [layers[layer]["color"] for layer in layers],
        values=[10] + [5-layers[layer]["depth"] for layer in layers],
        textinfo="label",
        hoverinfo="text",
        hovertext=["<b>Skin Cross-Section</b>"] + [
            f"<b>{layer}</b><br>Sensors: {', '.join(layers[layer]['sensors'])}" 
            for layer in layers
        ],
        branchvalues="total"
    ))

    if selected_layer:
        highlight_colors = []
        for layer in ["Skin"] + list(layers.keys()):
            if layer == selected_layer:
                highlight_colors.append("red")
            else:
                highlight_colors.append(layers[layer]["color"] if layer in layers else "#F5D393")

        fig.update_traces(
            marker=dict(
                colors=highlight_colors,
                line=dict(color="white", width=2)
            )
        )

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        height=350,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

# ========== Analyze Button ==========
if st.button("🩺 Analyze My Health"):
    current_input = {
        "Heart Rate": heart_rate,
        "Glucose": glucose,
        "Temperature": temperature,
        "Oxygen": oxygen,
        "Steps": steps
    }

    if all_vitals_normal(heart_rate, glucose, temperature, oxygen, steps):
        prediction = "Healthy"
        confidence = 1.0
    else:
        prediction, confidence = predict_risk_with_xgboost(heart_rate, glucose, temperature, oxygen, steps)

    st.session_state.last_input = current_input
    st.session_state.prediction = prediction
    st.session_state.confidence = confidence
    st.session_state.show_sticker_ui = True

# ========== Render Results if Available ==========
if st.session_state.prediction:
    current_input = st.session_state.last_input
    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    left_col, right_col = st.columns([2, 1.5])

    with left_col:
        st.subheader("📊 Health Risk Prediction")
        if prediction == "Healthy":
            st.success(f"🟢 You are likely *Healthy* (Confidence: {confidence:.0%}). Great job!")
        elif prediction == "Needs Checkup":
            st.warning(f"🟠 You may *Need a Checkup* (Confidence: {confidence:.0%}). Monitor your health.")
        else:
            st.error(f"🔴 *Critical Alert!* (Confidence: {confidence:.0%}) Please consult a doctor immediately.")

        st.subheader("🔬 Nano Sensor Analysis")
        abnormal_summaries = []
        for vital, value in current_input.items():
            status, icon, color = assess_risk(vital, value)
            st.markdown(f"### {icon} {vital}: {value}")
            st.markdown(f"**Status:** `{status}`")
            st.info(f"🧬 **How Nano-sensors Detect It:** {nano_explain[vital]}")
            if vital != "Steps" and status != "Normal":
                abnormal_summaries.append(f"{vital} is {status.lower()} while other metrics are normal.")

        if abnormal_summaries:
            st.subheader("🔎 Summary of Abnormal Vitals")
            for item in abnormal_summaries:
                st.warning(f"⚠️ {item}")

    with right_col:
        st.subheader("🧪 Interactive Skin Analysis")

        layer_info = {
            "Epidermis": "📍 *Epidermis Layer*: Nano-thermistors and sweat-analyzing graphene sensors detect **temperature** and **glucose** from surface sweat.",
            "Dermis": "📍 *Dermis Layer*: Nano-optodes measure **blood oxygen** through near-infrared spectroscopy of capillaries.",
            "Sweat Glands": "📍 *Sweat Glands*: Enzyme-based nanosensors track **glucose** and **electrolytes** in secreted sweat.",
            "Capillaries": "📍 *Capillaries*: Piezoelectric nanosensors detect **heart rate** via arterial pulse vibrations."
        }

        default_layer = "Epidermis"
        for vital, value in current_input.items():
            status, icon, _ = assess_risk(vital, value)
            if status != "Normal":
                if vital == "Temperature":
                    default_layer = "Epidermis"
                elif vital == "Glucose":
                    default_layer = "Sweat Glands"
                elif vital == "Oxygen":
                    default_layer = "Dermis"
                elif vital == "Heart Rate":
                    default_layer = "Capillaries"
                break

        selected_layer = st.radio(
            "🔍 Select a Skin Layer to Explore",
            options=list(layer_info.keys()),
            index=list(layer_info.keys()).index(default_layer),
            horizontal=True,
            key="skin_layer_radio"
        )

        st.plotly_chart(create_skin_diagram(selected_layer), use_container_width=True)
        st.info(layer_info[selected_layer])

# ========== Sticker UI ==========
if st.session_state.show_sticker_ui:
    st.subheader("📥 Get Your Nano Health Report")
    with st.expander("🪪 Generate Your Personalized Report"):
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])

        if st.button("Download "):
            if not name or not age or not gender:
                st.warning("Please fill in all details.")
            else:
                def vital_line(vital, value, normal_range):
                    return f"{vital}: {value} (Normal: {normal_range})"

                def safe_text(text):
                    return text.replace("–", "-").replace("°", " degrees").encode('latin-1', 'ignore').decode('latin-1')

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt="Nano-Wearable Health Report", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=safe_text(f"Name: {name}"), ln=True)
                pdf.cell(200, 10, txt=safe_text(f"Age: {int(age)}"), ln=True)
                pdf.cell(200, 10, txt=safe_text(f"Gender: {gender}"), ln=True)
                pdf.ln(5)
                pdf.cell(200, 10, txt="Vitals:", ln=True)
                pdf.cell(200, 10, txt=safe_text(vital_line("Heart Rate", heart_rate, "60-100 bpm")), ln=True)
                pdf.cell(200, 10, txt=safe_text(vital_line("Glucose", glucose, "70-140 mg/dL")), ln=True)
                pdf.cell(200, 10, txt=safe_text(vital_line("Temperature", temperature, "36.1-37.5 C")), ln=True)
                pdf.cell(200, 10, txt=safe_text(vital_line("Oxygen", oxygen, "95-100%")), ln=True)
                pdf.cell(200, 10, txt=safe_text(vital_line("Steps", steps, "3000-8000/day")), ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt=safe_text(f"AI Prediction: {prediction} ({confidence:.0%} confidence)"), ln=True)

                pdf_data = pdf.output(dest='S').encode('latin-1')

                st.download_button(
                    label="⬇️ Download  PDF",
                    data=pdf_data,
                    file_name=f"{name}_Nano_Report.pdf",
                    mime="application/pdf"
                )

# ========== Footer ==========
st.markdown("---")
