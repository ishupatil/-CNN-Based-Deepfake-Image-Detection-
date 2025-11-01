import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from keras.applications.mobilenet_v2 import preprocess_input

# =========================
# Load the trained model
# =========================
MODEL_PATH = "deepfake_detection_model.h5"

try:
    model = load_model(MODEL_PATH)
except:
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found! Please run train.py first.")
    st.stop()

# =========================
# Prediction function
# =========================
def predict_image(image):
    """
    Predict if the image is Fake or Real based on the model.
    Returns class label and confidence (%)
    """
    # Resize to match model input
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocess for MobileNetV2
    image_array = preprocess_input(image_array)

    # Model prediction
    prediction = model.predict(image_array, verbose=0)[0][0]  # single value output

    # Determine class and confidence
    if prediction >= 0.5:
        result = "Real"
        color = "green"
        confidence = prediction * 100
        description = (
            "✅ This image is classified as Real. "
            "The model has detected no anomalies typical in deepfake images."
        )
    else:
        result = "Fake"
        color = "red"
        confidence = (1 - prediction) * 100
        description = (
            "❌ This image is classified as Fake. "
            "The model has detected artifacts or inconsistencies typical in deepfake images."
        )

    return result, confidence, color, description

# =========================
# Streamlit App UI
# =========================
st.markdown("<h1 style='text-align: center; color: grey;'>DEEPFAKE DETECTION IN SOCIAL MEDIA CONTENT</h1>", unsafe_allow_html=True)

# Cover page
if os.path.exists("coverpage.png"):
    st.image("coverpage.png")
else:
    st.warning("⚠️ coverpage.png not found")

# About deepfakes
st.header("Understanding Deepfakes")
st.write("""
Deepfakes are synthetic media where a person in an existing image or video is replaced with someone else's likeness. 
AI models analyze subtle artifacts to distinguish real from fake content. Detecting deepfakes is crucial for 
security, privacy, and trust in digital media.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Predict
    with st.spinner("Analyzing image..."):
        result, confidence, color, description = predict_image(image)

    # Display results
    st.markdown(f"<h1 style='color:{color};'>The image is {result}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{color};'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)
    st.write(description)

# =========================
# Training results
# =========================
st.markdown("---")
st.title("📊 Model Training Results")

training_graph_path = "training_results.png"
if os.path.exists(training_graph_path):
    st.image(training_graph_path, caption="Training Accuracy & Loss")
else:
    st.warning(f"⚠️ {training_graph_path} not found. Run `train.py` to generate it.")

# =========================
# Footer
# =========================
st.markdown("""
---
**Contact Us:** [ishupatil2003@gmail.com](mailto:ishupatil2003@gmail.com)  
**Follow us on:** [Twitter](https://twitter.com) | [LinkedIn](https://linkedin.com/in/ishwaree-patil) | [Facebook](https://facebook.com)
""")
