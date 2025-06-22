
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import gc

st.set_page_config(page_title="💄 Segmentare Buze - Optimizat", layout="centered")

lipstick_df = pd.read_csv("avon_lipsticks.csv")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def find_closest_avon_lipstick(detected_rgb):
    min_dist = float('inf')
    closest = None
    for _, row in lipstick_df.iterrows():
        ref_rgb = hex_to_rgb(row["hex"])
        dist = color_distance(detected_rgb, ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest = row
    return closest

def classify_lip_color(rgb):
    h, s, v = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    if s < 64 and v > 200:
        return "nude"
    elif h < 10 or h > 170:
        return "roșu"
    elif 10 <= h <= 25:
        return "corai"
    elif 26 <= h <= 35:
        return "piersică"
    elif 36 <= h <= 70:
        return "auriu / galben"
    elif 71 <= h <= 160:
        return "mov / prună"
    elif 161 <= h <= 169:
        return "roz / fucsia"
    else:
        return "neclasificat"

def segment_lips_with_roboflow(image, api_key, model_url):
    image.save("/tmp/temp_image.jpg")
    with open("/tmp/temp_image.jpg", "rb") as image_file:
        response = requests.post(
            model_url,
            files={"file": image_file},
            headers={"Authorization": f"Bearer {api_key}"}
        )
    if response.status_code == 200:
        mask_url = response.json()["predictions"][0]["mask"]
        mask_img = Image.open(BytesIO(requests.get(mask_url).content)).convert("RGB")
        return mask_img
    else:
        return None

st.title("💄 Segmentare Buze cu Roboflow (Optimizat)")

uploaded_file = st.file_uploader("📤 Încarcă o imagine JPG/PNG", type=["jpg", "jpeg", "png"])
ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"] if "ROBOFLOW_API_KEY" in st.secrets else st.text_input("🔑 Introdu API Key Roboflow")
ROBOFLOW_MODEL_URL = "https://infer.roboflow.com/lips-segmentation-dqqxf/1"

if uploaded_file and ROBOFLOW_API_KEY:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((512, 512))
    st.image(image, caption="Imagine redimensionată", use_container_width=True)

    st.info("🔄 Se trimite imaginea către Roboflow...")
    mask = segment_lips_with_roboflow(image, ROBOFLOW_API_KEY, ROBOFLOW_MODEL_URL)

    if mask:
        st.image(mask, caption="🧠 Mască buze de la Roboflow", use_container_width=True)
        mask_np = np.array(mask)
        lips_pixels = mask_np.reshape(-1, 3)
        lips_pixels = lips_pixels[(lips_pixels != [0, 0, 0]).any(axis=1)]

        if len(lips_pixels) > 0:
            Z = np.float32(lips_pixels)
            _, _, center = cv2.kmeans(Z, 1, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                10, cv2.KMEANS_RANDOM_CENTERS)
            dominant_color = center[0].astype(int)

            nuanta = classify_lip_color(dominant_color)
            ruj = find_closest_avon_lipstick(dominant_color)

            st.markdown("### 🎨 Nuanță detectată")
            st.image(np.full((50, 50, 3), dominant_color, dtype=np.uint8), width=60)
            st.write(f"Nuanță estimată: **{nuanta}**")
            st.write(f"Ruj Avon sugerat: **{ruj['name']}** ({ruj['label']})")
        else:
            st.warning("⚠️ Nu s-au detectat pixeli colorați în mască.")
    else:
        st.error("❌ Eroare la apelul Roboflow sau detecția buzelor.")

    del image
    del mask
    gc.collect()
