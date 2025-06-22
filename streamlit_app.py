
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import colorsys
import mediapipe as mp

st.set_page_config(page_title="💄 Detectare Nuanțe Lipstick", layout="centered")

lipstick_df = pd.read_csv("avon_lipsticks.csv")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))

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
    r, g, b = rgb / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = h * 360
    if s < 0.25 and v > 0.8:
        return "nude"
    elif h < 15 or h > 345:
        return "roșu"
    elif 15 <= h <= 35:
        return "corai"
    elif 36 <= h <= 50:
        return "piersică"
    elif 51 <= h <= 70:
        return "auriu / galben (neobișnuit)"
    elif 71 <= h <= 150:
        return "verzui (neuzual)"
    elif 151 <= h <= 250:
        return "mov / prună"
    elif 251 <= h <= 320:
        return "roz / fucsia"
    elif 321 <= h <= 344:
        return "roșu-roz"
    else:
        return "neclasificat"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

st.markdown("<h1 style='text-align: center; color: #d63384;'>💄 Detecție automată a nuanțelor de ruj</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Încarcă una sau mai multe imagini pentru a detecta nuanțele lipstick-urilor și sugestiile de la Avon.</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📤 Încarcă imagini JPG sau PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image_rgb.shape

        st.image(image_rgb, caption=f"📸 {uploaded_file.name}", use_container_width=True)

        results = face_mesh.process(image_rgb)

        lips_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            415, 310, 311, 312, 13, 82, 81, 42, 183, 78
        ]

        if results.multi_face_landmarks:
            st.markdown("---")
            st.subheader("🔍 Nuanțele detectate în zona buzelor:")

            for face_landmarks in results.multi_face_landmarks:
                lips_points = []
                for idx in lips_indices:
                    x = int(face_landmarks.landmark[idx].x * img_width)
                    y = int(face_landmarks.landmark[idx].y * img_height)
                    lips_points.append((x, y))

                mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(lips_points, dtype=np.int32)], 255)

                lips_area = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
                pixels = lips_area[mask == 255].reshape(-1, 3)

                kmeans = KMeans(n_clusters=3)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_.astype(int)

                for i, color in enumerate(colors):
                    nuanta = classify_lip_color(color)
                    ruj = find_closest_avon_lipstick(color)

                    st.markdown(f"<h4 style='color:#6f42c1;'>🎨 Nuanță #{i+1}</h4>", unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.image(np.full((60, 60, 3), color, dtype=np.uint8), use_container_width=True)
                    with col2:
                        st.markdown(f"""
                        <b>Nuanță estimată:</b> <span style='color:#dc3545'>{nuanta}</span><br>
                        <b>Ruj Avon:</b> <i>{ruj['name']}</i><br>
                        <b>Etichetă:</b> {ruj['label']}
                        """, unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning(f"⚠️ Fața nu a fost detectată în imaginea {uploaded_file.name}.")
