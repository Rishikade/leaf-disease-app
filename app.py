import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page config
st.set_page_config(page_title="Leaf Detector", layout="centered")

# UI styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = tf.keras.models.load_model("model/leaf_model.h5")

# Classes
classes = ['early_blight', 'healthy', 'late_blight']

# Disease info
disease_info = {
    "early_blight": "🟤 Early blight detected. Remove infected leaves and use fungicide.",
    "late_blight": "⚠ Late blight detected. Avoid moisture and apply fungicide immediately.",
    "healthy": "✅ Plant is healthy. Maintain proper care."
}

# Chatbot logic
def chatbot_response(user_input):
    user_input = user_input.lower()

    if "early blight" in user_input:
        return "Early blight is a fungal disease. Remove affected leaves and use fungicide."

    elif "late blight" in user_input:
        return "Late blight spreads fast. Avoid moisture and apply fungicide immediately."

    elif "healthy" in user_input:
        return "Your plant is healthy. Maintain proper watering and sunlight."

    elif "how to cure" in user_input or "cure" in user_input:
        return "Use proper fungicides, remove infected parts, and maintain plant hygiene."

    elif "prevention" in user_input:
        return "Avoid overwatering, ensure good airflow, and inspect plants regularly."

    else:
        return "Ask me about plant diseases, cure, or prevention 🌿"

# History store
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.title("🌿 AI Leaf Disease Detector + Chatbot 🤖")
st.write("Upload or capture a leaf image and chat about plant diseases 🚀")

# Upload
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

# Camera
camera_image = st.camera_input("📸 Or take a photo")

def process_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    prediction = model.predict(img)

    top3_idx = np.argsort(prediction[0])[-3:][::-1]
    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return prediction, top3_idx, result, confidence

# Upload prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Leaf Image 🌿", use_column_width=True)
    st.markdown("### 🔍 Analysis Result")

    prediction, top3_idx, result, confidence = process_image(image)

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(disease_info[result])

    st.markdown("### 🧠 Top Predictions")
    for i in top3_idx:
        st.write(f"{classes[i]} : {prediction[0][i]*100:.2f}%")

    st.session_state.history.append(result)

# Camera prediction
if camera_image is not None:
    image = Image.open(camera_image)

    st.image(image, caption="Captured Image 📸", use_column_width=True)
    st.markdown("### 🔍 Analysis Result")

    prediction, top3_idx, result, confidence = process_image(image)

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(disease_info[result])

    st.markdown("### 🧠 Top Predictions")
    for i in top3_idx:
        st.write(f"{classes[i]} : {prediction[0][i]*100:.2f}%")

    st.session_state.history.append(result)

# History
if st.session_state.history:
    st.markdown("### 📜 Prediction History")
    for item in st.session_state.history[-5:]:
        st.write(item)

# Chatbot UI
st.markdown("## 🤖 Plant AI Chatbot")

user_input = st.text_input("Ask something about plant diseases:")

if user_input:
    response = chatbot_response(user_input)
    st.write("🤖:", response)