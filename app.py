import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('emotion_model.h5')  # Place model in same folder

# Define class labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# App title
st.title("Facial Emotion Recognition App")
st.write("Upload a face image and get the predicted emotion!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # Convert to grayscale and resize for model
    face_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_input = face_resized.astype("float32") / 255.0
    face_input = img_to_array(face_input)
    face_input = np.expand_dims(face_input, axis=0)
    face_input = np.expand_dims(face_input, axis=-1)  # (1, 48, 48, 1)

    # Predict
    preds = model.predict(face_input)
    emotion = emotion_labels[np.argmax(preds)]

    st.subheader(f"Predicted Emotion: **{emotion}**")
