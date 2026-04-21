import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/leaf_model.h5")

classes = ['early_blight', 'healthy', 'late_blight']

# Disease info
disease_info = {
    "early_blight": "Fungal disease. Remove infected leaves and apply fungicide.",
    "late_blight": "Serious disease. Avoid moisture and apply fungicide immediately.",
    "healthy": "Plant is healthy. Maintain proper watering and sunlight."
}

# Chatbot logic
def chatbot(user_input):
    user_input = user_input.lower()

    if "early blight" in user_input:
        return "Early blight is caused by fungus. Remove infected leaves and use fungicide."

    elif "late blight" in user_input:
        return "Late blight spreads quickly. Keep leaves dry and apply fungicide."

    elif "healthy" in user_input:
        return "Your plant is healthy. Just maintain good care."

    elif "cure" in user_input:
        return "Use fungicide, remove infected parts, and ensure proper airflow."

    elif "prevent" in user_input:
        return "Avoid overwatering and inspect plants regularly."

    else:
        return "Ask about diseases, cure, or prevention 🌿"

# Page config
st.set_page_config(page_title="LeafSense AI", layout="wide")

# ---------- UI ----------
st.markdown("""
<style>
body { background-color: #f6f7fb; }
.block-container { padding-top: 2rem; }

.title { font-size: 38px; font-weight: 700; color: #111827; }
.subtitle { color: #6b7280; }

.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.06);
}

.result { font-size: 22px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🌿 LeafSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered plant disease detection</div>', unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns(2)

# ---------- LEFT ----------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📸 Input Image")

    uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    camera = st.camera_input("Take photo")

    image = None

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    elif camera:
        image = Image.open(camera)
        st.image(image, use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- RIGHT ----------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Analysis")

    if image is not None:
        img = np.array(image)
        img = cv2.resize(img, (224,224))
        img = img / 255.0
        img = np.reshape(img, (1,224,224,3))

        prediction = model.predict(img)

        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.warning(disease_info[result])

        st.markdown("### 📊 Top Predictions")

        top3 = np.argsort(prediction[0])[-3:][::-1]

        for i in top3:
            st.progress(float(prediction[0][i]))
            st.caption(f"{classes[i]} — {prediction[0][i]*100:.2f}%")

        # Model Info
        st.markdown("### 📈 Model Info")
        st.write("Model: CNN (Deep Learning)")
        st.write("Input Size: 224x224")
        st.write("Classes: Early Blight, Late Blight, Healthy")

    else:
        st.info("Upload or capture an image")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- CHATBOT ----------
st.markdown("---")
st.subheader("🤖 Plant Assistant")

user_input = st.text_input("Ask about plant disease...")

if user_input:
    response = chatbot(user_input)
    st.write("🤖:", response)

# Footer
st.markdown("---")
st.caption("LeafSense AI • Built with Deep Learning 🚀")
