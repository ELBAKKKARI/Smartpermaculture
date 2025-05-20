import streamlit as st
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
import lime.lime_tabular
import os
from PIL import Image

# Directory paths
DATA_PATH = "data/"
IMAGE_PATH = "assets/images/"

# Streamlit page config
st.set_page_config(page_title="XCrop - Smart Permaculture Assistant", layout="wide", initial_sidebar_state="expanded")

# Language selection
language = st.sidebar.selectbox("🌍 Language", ["Français", "English"])
lang_key = "French" if language == "Français" else "English"

# Translation dictionary
translations = {
    "Français": {
        "title": "XCrop : Assistant intelligent de permaculture",
        "subtitle": "Recommandations de culture basées sur l'IA et la permaculture",
        "input_mode": "📥 Mode de saisie des données",
        "manual_input": "Entrée manuelle",
        "sensor_input": "Capteurs IoT (CPS Mode)",
        "soil_input": "🖍️ Entrez vos valeurs de sol et de climat",
        "recommend": "🔍 Recommander une culture",
        "recommended_crop": "✅ Culture recommandée :",
        "reasons": "🤖 Raisons de cette recommandation",
        "companions": "🌿 Plantes compagnes :",
        "avoid": "⚠️ À éviter à proximité :",
        "sensors": "📱 Capteurs suggérés :",
        "note": "📖 Note :",
        "not_found": "ℹ️ Aucune donnée compagne trouvée dans la base de connaissances.",
        "footer": "Créé avec ❤️ par XCrop - Intelligence & Permaculture"
    },
    "English": {
        "title": "XCrop: Smart Permaculture Assistant",
        "subtitle": "AI-based intelligent crop & companion planning",
        "input_mode": "📥 Input mode",
        "manual_input": "Manual Input",
        "sensor_input": "IoT Sensors (CPS Mode)",
        "soil_input": "🖍️ Enter your soil and climate values",
        "recommend": "🔍 Recommend a crop",
        "recommended_crop": "✅ Recommended Crop:",
        "reasons": "🤖 Explanation for this recommendation",
        "companions": "🌿 Companion Plants:",
        "avoid": "⚠️ Avoid Planting Near:",
        "sensors": "📱 Suggested Sensors:",
        "note": "📖 Explanation:",
        "not_found": "ℹ️ No companion data found in the knowledge base.",
        "footer": "Built with ❤️ by XCrop - AI & Permaculture"
    }
}
T = translations[language]

# Header
st.markdown(f"<h1 style='text-align:center; color:green;'>🌱 {T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='text-align:center; color:gray;'>{T['subtitle']}</h4><hr>", unsafe_allow_html=True)

# Load model
@st.cache_data
def load_model():
    df = pd.read_csv(DATA_PATH + "Crop_Recommendation.csv")
    X = df.drop("Crop", axis=1)
    y = df["Crop"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X, y, X.columns.tolist()

model, X_train, y_train, features = load_model()

# Load plant images
@st.cache_data
def load_images():
    images = {}
    for file in os.listdir(IMAGE_PATH):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(file)[0].lower()
            images[name] = os.path.join(IMAGE_PATH, file)
    return images

plant_images = load_images()

# Load explanations
@st.cache_data
def load_explanations():
    with open(DATA_PATH + "plant_explanations.json") as f:
        return json.load(f)

# Load plant companions
@st.cache_data
def load_companions():
    with open(DATA_PATH + "plants_full.json") as f:
        return json.load(f)

plant_justifications = load_explanations()
plant_data = load_companions()

# LIME explanation formatting
def human_readable_explanation(lime_list):
    friendly = {
        "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
        "temperature": "Temperature", "humidity": "Humidity",
        "ph": "pH", "rainfall": "Rainfall"
    }
    output = []
    for feature, weight in lime_list:
        for key, val in friendly.items():
            feature = feature.replace(key, val)
        output.append(f"🔎 {feature} {'helps' if weight > 0 else 'hinders'} this crop.")
    return output

# Sidebar input mode
mode = st.sidebar.radio(T['input_mode'], [T['manual_input'], T['sensor_input']])

# Manual input UI
if mode == T['manual_input']:
    st.markdown(f"### {T['soil_input']}")
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("🌿 Nitrogen (N)", 0, 140, 90)
        P = st.slider("🌿 Phosphorus (P)", 0, 145, 42)
        K = st.slider("🌿 Potassium", 0, 205, 43)
    with col2:
        temperature = st.slider("🌡️ Temperature (°C)", 10, 45, 25)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 80)
        ph = st.slider("🌍 pH level", 3.0, 10.0, 6.5)
        rainfall = st.slider("🌧️ Rainfall (mm)", 20, 300, 120)

    if st.button(T['recommend']):
        input_data = pd.DataFrame([{
            "N": N, "P": P, "K": K, "temperature": temperature,
            "humidity": humidity, "ph": ph, "rainfall": rainfall
        }], columns=features)

        predicted_crop = model.predict(input_data)[0]
        crop_key = predicted_crop.lower()

        st.success(f"{T['recommended_crop']} **{predicted_crop}**")

        if crop_key in plant_images:
            image = Image.open(plant_images[crop_key])
            st.image(image, caption=predicted_crop,width=400)


        # LIME explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=features,
            class_names=model.classes_,
            mode='classification'
        )
        explanation = explainer.explain_instance(input_data.iloc[0].values, model.predict_proba)

        st.markdown(f"### {T['reasons']}")
        for line in human_readable_explanation(explanation.as_list()):
            st.markdown(f"- {line}")

        # Justification
        if crop_key in plant_justifications:
            text = plant_justifications[crop_key].get(lang_key) or plant_justifications[crop_key].get("English")
            if text:
                st.info(f"📌 {text}")

        # Companions
        for plant in plant_data:
            if plant["plant"].lower() == crop_key:
                st.markdown(f"##### {T['companions']}")
                st.markdown(", ".join(plant["companions"]))
                st.markdown(f"##### {T['avoid']}")
                st.markdown(", ".join(plant["avoid"]))
                st.markdown(f"##### {T['sensors']}")
                st.markdown(", ".join(plant["sensors"]))
                st.info(f"{T['note']} {plant['notes']}")
                break
        else:
            st.warning(T['not_found'])

st.markdown(f"<hr><div style='text-align:center;'>{T['footer']}</div>", unsafe_allow_html=True)
