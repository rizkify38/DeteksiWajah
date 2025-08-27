import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# -------------------------------
# Download model weights dari Google Drive
# -------------------------------
WEIGHTS_PATH = "best.h5"
GOOGLE_DRIVE_ID = "1BP4E6jOTaFv_-b0bouAhGuYESxGHwHDB"

if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("🔄 Mengunduh weight model dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

if not os.path.exists(WEIGHTS_PATH):
    st.error("❌ File best.h5 tidak ditemukan. Cek link Google Drive!")
    st.stop()

# -------------------------------
# Definisikan ulang arsitektur model
# (Sesuaikan dengan arsitektur saat training)
# -------------------------------
def build_model(input_shape=(48,48,3), num_classes=7):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model

# -------------------------------
# Load model + weight
# -------------------------------
@st.cache_resource
def load_my_model():
    model = build_model()
    model.load_weights(WEIGHTS_PATH)   # <-- load weight, bukan load_model
    return model

model = load_my_model()

# -------------------------------
# Daftar Label Ekspresi
# -------------------------------
class_names = ["Marah", "Jijik", "Takut", "Bahagia", "Sedih", "Terkejut", "Netral"]

# -------------------------------
# Aplikasi Streamlit
# -------------------------------
st.title("📷 Deteksi Ekspresi Wajah dengan Kamera")
st.write("Gunakan kamera untuk mengenali ekspresi wajah secara langsung.")

img_file = st.camera_input("Ambil foto dengan kamera")

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Gambar dari Kamera", use_container_width=True)

    # Preprocess
    target_size = (48, 48)
    img_resized = ImageOps.fit(img, target_size, Image.ANTIALIAS)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    # Hasil
    st.subheader("Hasil Prediksi")
    st.write(f"📌 Ekspresi: **{class_names[predicted_class]}**")
    st.write(f"🔎 Probabilitas: {confidence:.2f}")
    st.bar_chart(preds[0])
