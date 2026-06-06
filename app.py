import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI CNC Digital Twin",
    page_icon="⚙️",
    layout="wide"
)

# ==========================
# SCI-FI CSS
# ==========================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#020617,#081428,#0B1D33);
    color:white;
}

h1,h2,h3{
    color:#00E5FF;
}

div[data-testid="stMetric"]{
    background:#081428;
    border:1px solid #00E5FF;
    padding:15px;
    border-radius:15px;
    box-shadow:0 0 15px rgba(0,229,255,0.25);
}

.block-container{
    padding-top:1rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================

st.title("⚙️ AI CNC DRILLING MACHINE DIGITAL TWIN")
st.caption("Industrial Predictive Maintenance Command Center")

# ==========================
# LOAD DATA
# ==========================

try:
    df = pd.read_csv("industrial_drilling_dataset.csv")
except:
    st.error("industrial_drilling_dataset.csv not found")
    st.stop()

# ==========================
# LOAD MODEL
# ==========================

try:
    model = joblib.load("failure_prediction_model.pkl")
except:
    st.warning("Model file not found")
    model = None

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Machine Inputs")

temperature = st.sidebar.slider("Temperature (°C)",20,120,65)
vibration = st.sidebar.slider("Vibration (mm/s)",0.0,10.0,2.5)
rpm = st.sidebar.slider("RPM",1000,3000,1500)
torque = st.sidebar.slider("Torque",50,300,150)
tool_wear = st.sidebar.slider("Tool Wear (%)",0,100,20)
pressure = st.sidebar.slider("Pressure",1.0,15.0,8.0)
power = st.sidebar.slider("Power Consumption",1.0,30.0,15.0)

# ==========================
# AI PREDICTION
# ==========================

prediction = None

if model is not None:

    input_df = pd.DataFrame({
        "Temperature":[temperature],
        "Vibration":[vibration],
        "RPM":[rpm],
        "Torque":[torque],
        "Tool_Wear":[tool_wear],
        "Pressure":[pressure],
        "Power":[power]
    })

    prediction = model.predict(input_df)[0]

# ==========================
# HEALTH SCORE
# ==========================

health_score = max(
    0,
    100 - (
        tool_wear*0.4 +
        vibration*8 +
        max(0,temperature-60)*0.5
    )
)

# ==========================
# TOP METRICS
# ==========================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("🌡 Temperature", f"{temperature:.1f} °C")

with c2:
    st.metric("📳 Vibration", f"{vibration:.2f} mm/s")

with c3:
    st.metric("⚙ RPM", f"{rpm}")

with c4:
    st.metric("💚 Health", f"{health_score:.1f}%")

st.divider()

# ==========================
# MACHINE STATUS
# ==========================

if prediction is not None:

    if prediction == 0:
        st.success("🟢 MACHINE STATUS : HEALTHY")

    else:
        st.error("🔴 MACHINE STATUS : FAILURE RISK DETECTED")

# ==========================
# GAUGE
# ==========================

col1,col2 = st.columns([1,2])

with col1:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text':"Machine Health"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':'cyan'}
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

# ==========================
# TELEMETRY GRAPH
# ==========================

with col2:

    fig = px.line(
        df.head(300),
        y=["Temperature","Vibration"],
        title="Live Telemetry"
    )

    st.plotly_chart(fig,use_container_width=True)

# ==========================
# ANALYTICS
# ==========================

st.subheader("📈 Operational Analytics")

left,right = st.columns(2)

with left:

    fig = px.scatter(
        df,
        x="Temperature",
        y="Vibration",
        color="Failure",
        title="Failure Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    fig = px.histogram(
        df,
        x="Tool_Wear",
        color="Failure",
        title="Tool Wear Analysis"
    )

    st.plotly_chart(fig,use_container_width=True)

# ==========================
# DIGITAL TWIN STATUS
# ==========================

st.subheader("🤖 Digital Twin Status")

status_df = pd.DataFrame({
    "Component":[
        "Spindle",
        "Bearing",
        "Tool",
        "Hydraulic System",
        "Motor"
    ],
    "Health":[
        health_score,
        health_score-5,
        health_score-10,
        health_score-8,
        health_score-3
    ]
})

st.dataframe(
    status_df,
    use_container_width=True
)

# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    f"System Online | {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)
