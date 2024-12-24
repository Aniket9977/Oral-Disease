import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model = load_model("oral_disease_classifier.h5")
class_labels = ["Healthy", "Dental Caries", "Gingivitis"]  # Update with your class labels

# Streamlit App
st.title("Oral Disease Classification")
st.write("Upload an image to classify the oral disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
