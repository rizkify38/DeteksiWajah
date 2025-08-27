import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# --------- Download Model dari Google Drive ----------
MODEL_PATH = "best.h5"
GOOGLE_DRIVE_ID = "1gq6y4B9Gek3Y8pN6d3gkOCLjupvQ0_rq"  # ID file yang kamu kasih

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

img_file = st.camera_input("Ambil foto dengan kamera")

if
