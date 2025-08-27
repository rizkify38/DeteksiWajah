import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# -------------------------------
# Download model dari Google Drive
# -------------------------------
MODEL_PATH = "best.h5"
GOOGLE_DRIVE_ID = "1BP4E6jOTaFv_-b0bouAhGuYESxGHwHDB"  # <<== Ganti dengan ID file Google Drive

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Mengunduh model dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

# -------------------------------
# Daftar Label Ekspresi
# -------------------------------
class_names = [
    "Marah",       # Angry
    "Jijik",       # Disgust
    "Takut",       # Fear
    "Bahagia",     # Happy
    "Sedih",       # Sad
    "Terkejut",    # Surprise
    "Netral"       # Neutral
]

# -------------------------------
# Aplikasi Streamlit
# -------------------------------
st.title("📷 Deteksi Ekspresi Wajah dengan Kamera")
st.write("Gunakan kamera untuk mengenali ekspresi wajah secara langsung.")

# Ambil gambar dari kamera
img_file = st.camera_input("Ambil foto dengan kamera")

if img_file is not None:
    # Buka gambar
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Gambar dari Kamera", use_container_width=True)

    # -------------------------------
    # Preprocessing
    # -------------------------------
    # Sesuaikan dengan input model (cek dengan model.input_shape jika perlu)
    target_size = (48, 48)  # ubah ke (224,224) kalau model kamu dilatih di 224x224
    img_resized = ImageOps.fit(img, target_size, Image.ANTIALIAS)

    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # -------------------------------
    # Prediksi
    # -------------------------------
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    # -------------------------------
    # Hasil
    # -------------------------------
    st.subheader("Hasil Prediksi")
    st.write(f"📌 Ekspresi Terdeteksi: **{class_names[predicted_class]}**")
    st.write(f"🔎 Probabilitas: {confidence:.2f}")

    # Tampilkan semua probabilitas
    st.bar_chart(preds[0])

