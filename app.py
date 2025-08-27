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
GOOGLE_DRIVE_ID = "1BP4E6jOTaFv_-b0bouAhGuYESxGHwHDB"  # ID dari link yang kamu kasih

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Mengunduh model dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    st.error("❌ File best.h5 tidak ditemukan. Cek link Google Drive!")
    st.stop()

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()

model = load_my_model()

# Ambil input shape & jumlah kelas dari model
input_shape = model.input_shape[1:3]   # (height, width)
num_classes = model.output_shape[-1]

# Default class_names untuk ekspresi wajah
default_classes = ["Marah", "Jijik", "Takut", "Bahagia", "Sedih", "Terkejut", "Netral"]

# Sesuaikan panjang class_names dengan output model
if num_classes <= len(default_classes):
    class_names = default_classes[:num_classes]
else:
    class_names = [f"Kelas {i}" for i in range(num_classes)]

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
    # Preprocessing sesuai input model
    # -------------------------------
    img_resized = ImageOps.fit(img, input_shape, Image.ANTIALIAS)
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
