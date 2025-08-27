import cv2
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Deteksi Wajah", layout="wide")
st.title("📷 Deteksi Wajah + Prediksi dengan Model (.pkl)")

# ===============================
# 1. Load Model dari file .pkl
# ===============================
@st.cache_resource
def load_model():
    model_path = "final_data.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
st.success("✅ Model berhasil dimuat dari `final_data.pkl`")

# ===============================
# 2. Load Haar Cascade
# ===============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ===============================
# 3. Stream Kamera
# ===============================
frame_window = st.image([])
run = st.checkbox("Aktifkan Kamera")

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("❌ Gagal membuka kamera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # --- Crop wajah ---
        face_roi = gray[y:y+h, x:x+w]

        # --- Preprocessing sesuai model ---
        try:
            face_resized = cv2.resize(face_roi, (64, 64))   # ubah sesuai ukuran input model
            face_flatten = face_resized.flatten().reshape(1, -1)

            # --- Prediksi ---
            pred = model.predict(face_flatten)
            label = str(pred[0])
        except Exception as e:
            label = f"Error: {e}"

        # --- Gambar kotak + label ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

camera.release()
