import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# --------- Download Model dari Google Drive ----------
MODEL_PATH = "best.h5"
GOOGLE_DRIVE_ID = "1gq6y4B9Gek3Y8pN6d3gkOCLjupvQ0_rq"  # ID file dari link drive

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Mengunduh model dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    st.error("❌ File best.h5 tidak ditemukan. Cek link Google Drive!")
    st.stop()

# --------- Load Model ----------
@st.cache_resource
def load_my_model():
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()
    return model

model = load_my_model()

# --------- Setup Labels Ekspresi ----------
class_names = [
    "Marah",       # Angry
    "Jijik",       # Disgust
    "Takut",       # Fear
    "Bahagia",     # Happy
    "Sedih",       # Sad
    "Terkejut",    # Surprise
    "Netral"       # Neutral
]

# --------- Streamlit UI ----------
st.title("📷 Deteksi Ekspresi Wajah (Snapshot Kamera)")
st.write("Ambil snapshot dari kamera lalu aplikasi mendeteksi ekspresi wajah.")

# Ambil gambar dari kamera
img_file = st.camera_input("Ambil foto dengan kamera")

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Gambar dari Kamera", use_container_width=True)

    # Preprocessing sesuai input model
    target_size = model.input_shape[1:3]  # ambil ukuran dari model (misal 48x48 atau 224x224)
    img_resized = ImageOps.fit(img, target_size, Image.ANTIALIAS)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write(f"**Ekspresi**: {class_names[pred_idx]}")
    st.write(f"**Probabilitas**: {confidence:.2f}")
    st.bar_chart(preds[0])
