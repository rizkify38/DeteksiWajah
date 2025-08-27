import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    model = load_model("best.h5")
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
    # Preprocessing (ubah sesuai input model)
    # -------------------------------
    # Asumsi model dilatih dengan ukuran 48x48 atau 224x224
    # Kalau error, ubah sesuai input_shape modelmu
    target_size = (48, 48)  # ubah ke (224,224) kalau model pakai itu
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
