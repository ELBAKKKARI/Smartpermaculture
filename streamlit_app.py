import streamlit as st
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier

# 🌍 Streamlit page config
st.set_page_config(page_title="Smart Permaculture Assistant", layout="centered")

# 🌿 Header
st.markdown("""
<h1 style='text-align: center; color: green;'>🌱 Smart Permaculture Assistant</h1>
<h4 style='text-align: center; color: gray;'>AI-powered crop & companion recommendations</h4>
<hr>
""", unsafe_allow_html=True)

# 📊 Load model and dataset
@st.cache_data
def load_model():
    df = pd.read_csv("Crop_Recommendation.csv")
    X = df.drop("Crop", axis=1)
    y = df["Crop"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, features = load_model()
# 🌿 Load companion data
with open("plants.json") as f:
    plant_data = json.load(f)

# 👨‍🌾 Choose data input mode
mode = st.radio("📥 How would you like to provide data?", ["Manual Input", "Use IoT Sensors (CPS Mode)"])

st.markdown("---")

# ✏️ MANUAL INPUT
if mode == "Manual Input":
    st.subheader("📝 Enter Your Soil and Climate Values")
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("🧪 Nitrogen (N)", 0, 140, 90)
        P = st.slider("🧪 Phosphorus (P)", 0, 145, 42)
        K = st.slider("🧪 Potassium (K)", 0, 205, 43)
    with col2:
        temperature = st.slider("🌡️ Temperature (°C)", 10, 45, 25)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 80)
        ph = st.slider("🌍 pH level", 3.0, 10.0, 6.5)
        rainfall = st.slider("🌧️ Rainfall (mm)", 20, 300, 120)

    if st.button("🔍 Recommend Crop"):
        input_data = pd.DataFrame([{
            "N": N, "P": P, "K": K, "temperature": temperature,
            "humidity": humidity, "ph": ph, "rainfall": rainfall
        }], columns=features)

        predicted_crop = model.predict(input_data)[0]
        st.success(f"✅ Recommended Crop: **{predicted_crop}**")

        for plant in plant_data:
            if plant["plant"].lower() == predicted_crop.lower():
                st.markdown("##### 🌿 Companion Plants:")
                st.markdown(", ".join(plant["companions"]))
                st.markdown("##### ⚠️ Avoid Planting Near:")
                st.markdown(", ".join(plant["avoid"]))
                st.markdown("##### 📱 Suggested Sensors:")
                st.markdown(", ".join(plant["sensors"]))
                st.info(f"📖 Explanation: {plant['notes']}")
                break
        else:
            st.warning("ℹ️ No companion data found in knowledge base.")

# 🚁 CPS SENSOR MODE
elif mode == "Use IoT Sensors (CPS Mode)":
    st.subheader("📡 Using Simulated IoT Sensor Values")

    sensor_data = {
        "N": 85, "P": 40, "K": 60,
        "temperature": 24, "humidity": 75,
        "ph": 6.3, "rainfall": 90
    }

    st.json(sensor_data)

    if st.button("🔍 Recommend Based on Sensors"):
        input_data = pd.DataFrame([sensor_data], columns=features)
        predicted_crop = model.predict(input_data)[0]
        st.success(f"✅ Recommended Crop: **{predicted_crop}**")

        for plant in plant_data:
            if plant["plant"].lower() == predicted_crop.lower():
                st.markdown("##### 🌿 Companion Plants:")
                st.markdown(", ".join(plant["companions"]))
                st.markdown("##### ⚠️ Avoid Planting Near:")
                st.markdown(", ".join(plant["avoid"]))
                st.markdown("##### 📱 Suggested Sensors:")
                st.markdown(", ".join(plant["sensors"]))
                st.info(f"📖 Explanation: {plant['notes']}")
                break
        else:
            st.warning("ℹ️ No companion data found in knowledge base.")

# 🌺 Footer
st.markdown("<hr><center><small>Built with ❤️ using Streamlit, AI, and Permaculture principles</small></center>", unsafe_allow_html=True)
