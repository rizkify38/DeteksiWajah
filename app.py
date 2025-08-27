import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# --------- Download Weights dari Google Drive ----------
MODEL_PATH = "best.h5"
GOOGLE_DRIVE_ID = "1gq6y4B9Gek3Y8pN6d3gkOCLjupvQ0_rq"  # Ganti dengan ID Google Drive-mu

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Mengunduh best.h5 dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    st.error("❌ File best.h5 tidak ditemukan. Cek link Google Drive!")
    st.stop()

# --------- Definisikan Ulang Arsitektur Model ----------
def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --------- Load Weights ----------
@st.cache_resource
def load_my_model():
    model = build_model()
    try:
        model.load_weights(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Gagal load weights: {e}")
        st.stop()
    return model

model = load_my_model()

# --------- Label Ekspresi ----------
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
    img = Image.open(img_file).convert("L")  # grayscale
    st.image(img, caption="Gambar dari Kamera", use_container_width=True)

    # Preprocessing (48x48 grayscale)
    target_size = (48, 48)
    img_resized = ImageOps.fit(img, target_size, Image.ANTIALIAS)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=-1)  # tambahkan channel grayscale
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
