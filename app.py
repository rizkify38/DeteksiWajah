# app.py
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

st.set_page_config(page_title="Deteksi Ekspresi Wajah (best.h5)", page_icon="🎭", layout="wide")

st.title("🎭 Deteksi Ekspresi Wajah — Real-time")
st.caption("Model: best.h5 (letakkan file best.h5 di root repo)")

# ================
# Load model safely
# ================
@st.cache_resource(show_spinner=False)
def load_emotion_model(path="best.h5"):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        # raise so caller can show friendly error
        raise RuntimeError(f"Gagal memuat model dari '{path}': {e}")

# ================
# Helper: detect model IO shapes
# ================
def inspect_model(model):
    """
    Mengambil info input shape dan output classes dari model Keras loaded.
    Mengembalikan: input_shape (h,w,c), n_classes (int)
    """
    # Input shape: model.input_shape bisa berupa (None,h,w,c) atau (None,h,w)
    input_shape = model.input_shape  # tuple
    if len(input_shape) == 4:
        _, h, w, c = input_shape
    elif len(input_shape) == 3:
        _, h, w = input_shape
        c = 1
    else:
        # fallback
        h, w, c = (48, 48, 1)

    # Output shape: model.output_shape -> (None, n_classes)
    output_shape = model.output_shape
    try:
        n_classes = int(output_shape[-1])
    except Exception:
        # fallback
        n_classes = 1

    return (int(h) if h is not None else 48,
            int(w) if w is not None else 48,
            int(c) if c is not None else 1,
            n_classes)

# ================
# UI: Pengaturan label
# ================
st.sidebar.header("Pengaturan Model & Label")
model_path = st.sidebar.text_input("Path model Keras (.h5)", value="best.h5")

# Muat model dengan feedback
model = None
try:
    model = load_emotion_model(model_path)
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

# Ambil info model
in_h, in_w, in_c, n_classes = inspect_model(model)
st.sidebar.write(f"Input shape model: {in_h}x{in_w}x{in_c}")
st.sidebar.write(f"Jumlah kelas (output): {n_classes}")

# Default labels jika n_classes == 7 gunakan urutan umum, jika 2 -> binary, dll.
default_labels = {
    7: ["Kemarahan", "Netral", "Jijik", "Ketakutan", "Kebahagiaan", "Kesedihan", "Kejutan"],
    6: ["Marah", "Jijik", "Takut", "Bahagia", "Sedih", "Netral"],
    5: ["Marah", "Takut", "Bahagia", "Sedih", "Netral"],
    2: ["Negatif", "Positif"]
}
if n_classes in default_labels:
    suggested = default_labels[n_classes]
else:
    # buat label placeholder
    suggested = [f"Class_{i}" for i in range(n_classes)]

st.sidebar.markdown("**Atur label kelas (satu baris per label)**")
labels_text = st.sidebar.text_area(
    "Label (jika kosong, akan menggunakan default)", 
    value="\n".join(suggested), height=160
)
# parse labels
custom_labels = [s.strip() for s in labels_text.splitlines() if s.strip() != ""]
if len(custom_labels) != n_classes:
    st.sidebar.warning(f"Jumlah label ({len(custom_labels)}) tidak sama dengan jumlah kelas model ({n_classes}). Aplikasi akan tetap berjalan, tetapi peta label mungkin tidak cocok.")
# gunakan custom_labels apa adanya
EMOTION_LABELS = custom_labels if len(custom_labels) > 0 else suggested

# ================
# Face detector
# ================
@st.cache_resource(show_spinner=False)
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

detector = load_face_detector()

# ================
# Preprocess sesuai input model
# ================
def preprocess_face(face_img):
    """
    face_img: grayscale or color crop from original (numpy)
    returns array shaped (1, h, w, c_model)
    Normalisasi ke 0-1
    """
    # Jika model expects color but face_img grayscale -> convert to BGR
    img = face_img
    if len(img.shape) == 2 and in_c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(img.shape) == 3 and in_c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    if in_c == 1:
        resized = cv2.resize(img, (in_w, in_h))
        arr = resized.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))  # (1,h,w,1)
    else:
        resized = cv2.resize(img, (in_w, in_h))
        arr = resized.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1,h,w,3)

    return arr

# ================
# Streamlit UI main
# ================
col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.subheader("Pengaturan Deteksi")
    draw_box = st.checkbox("Tampilkan Bounding Box", value=True)
    show_conf = st.checkbox("Tampilkan Confidence", value=True)
    min_face = st.slider("Ukuran minimum wajah (px)", 40, 300, 80, 10)
    scaleFactor = st.slider("Haar scaleFactor", 1.05, 1.50, 1.15, 0.01)
    minNeighbors = st.slider("Haar minNeighbors", 3, 8, 5, 1)

with col1:
    st.subheader("Kamera — Real-time")
    st.info("Klik Start untuk membuka kamera (webrtc). Jika tidak berfungsi, gunakan 'Ambil Gambar' di bawah sebagai fallback.")

# ================
# Video transformer
# ================
class EmotionTransformer(VideoTransformerBase):
    def __init__(self, draw_box=True, show_conf=True, min_size=80, scaleFactor=1.15, minNeighbors=5):
        # model & detector sudah dimuat global (cache_resource)
        self.model = model
        self.detector = detector
        self.draw_box = draw_box
        self.show_conf = show_conf
        self.min_size = min_size
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.last_probs = None

    def predict(self, face_crop):
        x = preprocess_face(face_crop)
        preds = self.model.predict(x, verbose=0)
        # preds shape: (1, n_classes) or (1,) for binary/logit
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        preds = preds[0]
        # If model outputs single logit (binary), transform to two-class probs
        if preds.size == 1:
            # assume sigmoid
            p1 = 1.0 / (1.0 + np.exp(-preds[0]))
            probs = np.array([1 - p1, p1])
        else:
            probs = preds
            # if outputs logits (not softmax), try softmax
            s = np.sum(probs)
            if s <= 1.0001 and s >= 0.0:
                # could be already probabilities
                pass
            else:
                # apply softmax to be safe
                ex = np.exp(probs - np.max(probs))
                probs = ex / np.sum(ex)
        idx = int(np.argmax(probs))
        label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else f"Class_{idx}"
        conf = float(probs[idx])
        return label, conf, probs

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=(self.min_size, self.min_size)
        )

        # pilih wajah terbesar untuk panel probabilitas
        faces_sorted = sorted(list(faces), key=lambda b: b[2]*b[3], reverse=True)
        best_probs = None

        for (x, y, w, h) in faces_sorted:
            # Crop face in original color if model expects color, else in gray
            if in_c == 3:
                face_crop = img[y:y+h, x:x+w]
            else:
                face_crop = gray[y:y+h, x:x+w]

            label, conf, probs = self.predict(face_crop)
            if best_probs is None:
                best_probs = probs

            if self.draw_box:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = label if not self.show_conf else f"{label} ({conf:.2f})"
            cv2.putText(img, text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.last_probs = best_probs
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================
# Run WebRTC streamer
# ================
webrtc_ctx = webrtc_streamer(
    key="emotion-webrtc",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=lambda: EmotionTransformer(
        draw_box=draw_box,
        show_conf=show_conf,
        min_size=min_face,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    ),
    async_processing=True,
)

# ================
# Panel probabilitas & fallback camera_input
# ================
with col2:
    st.subheader("Probabilitas (wajah utama)")
    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
        vt = webrtc_ctx.video_transformer
        if vt.last_probs is not None:
            probs = vt.last_probs
            for i, p in enumerate(probs):
                label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"Class_{i}"
                st.write(f"- **{label}**: {p:.3f}")
        else:
            st.write("Menunggu wajah terdeteksi...")
    else:
        st.write("WebRTC belum berjalan. Gunakan fallback di bawah jika kamera tidak terhubung.")

st.markdown("---")
st.subheader("Fallback: Ambil Gambar (single-shot)")
img_file = st.camera_input("Ambil gambar wajah (fallback)")

if img_file is not None:
    # Baca frame
    bytes_data = img_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(min_face, min_face))
    if len(faces) == 0:
        st.warning("Tidak ada wajah terdeteksi pada gambar.")
    else:
        # pilih wajah terbesar
        faces = sorted(list(faces), key=lambda b: b[2]*b[3], reverse=True)
        x, y, w, h = faces[0]
        if in_c == 3:
            face_crop = frame[y:y+h, x:x+w]
        else:
            face_crop = gray[y:y+h, x:x+w]
        label, conf, probs = EmotionTransformer().predict(face_crop)
        # tampil
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        st.image(frame, channels="BGR")
        st.write("Probabilitas:")
        for i, p in enumerate(probs):
            lbl = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"Class_{i}"
            st.write(f"- **{lbl}**: {p:.4f}")

st.markdown("---")


