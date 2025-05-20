import streamlit as st
import json

st.set_page_config(page_title="Plant Chooser", layout="wide")

# Load plant companion data
@st.cache_data
def load_plant_data():
    with open("data/plants_full.json") as f:
        return json.load(f)

plant_data = load_plant_data()
available_plants = sorted([p["plant"].capitalize() for p in plant_data])

st.title("🪴 Smart Plant Chooser")
st.markdown("Select the plants you want to grow. We'll help you avoid bad combinations and suggest smart companions.")

# Let the user choose plants
selected = st.multiselect("🌱 Choose crops you'd like to plant:", available_plants)

# Show results
if selected:
    st.subheader("🌿 Compatibility Check")

    companions = set()
    avoid_near = set()
    notes = []

    for plant in plant_data:
        if plant["plant"].capitalize() in selected:
            companions.update(plant["companions"])
            avoid_near.update(plant["avoid"])
            notes.append(f"**{plant['plant'].capitalize()}**: {plant['notes']}")

    # Filter out the plants already selected
    companions.difference_update([p.lower() for p in selected])
    avoid_near.intersection_update([p.lower() for p in selected])

    if avoid_near:
        st.error(f"⚠️ Conflict detected! These plants shouldn't be planted together: {', '.join(avoid_near)}")
    else:
        st.success("✅ No conflicts found between your selected plants.")

    if companions:
        st.info(f"👍 Good companion options to consider: {', '.join(companions)}")

    st.markdown("📝 **Plant notes:**")
    for note in notes:
        st.markdown(note)
from PIL import Image
import os

# Load plant images
def load_plant_images():
    image_map = {}
    img_dir = "assets/images"
    if os.path.exists(img_dir):
        for file in os.listdir(img_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                crop_name = os.path.splitext(file)[0].lower()
                image_map[crop_name] = os.path.join(img_dir, file)
    return image_map

images = load_plant_images()

# Show plant cards
st.subheader("🖼️ Selected Plant Cards")
cols = st.columns(len(selected)) if selected else []

for i, plant in enumerate(selected):
    with cols[i]:
        key = plant.lower()
        st.markdown(f"**{plant}**")
        if key in images:
            st.image(images[key], use_container_width=True,width=200)
        else:
            st.info("📷 No image found.")
