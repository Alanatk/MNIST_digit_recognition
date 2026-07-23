import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load the trained model
# Make sure the model is saved in the same directory or provide the full path
# You might need to save your trained model from the Colab notebook first
# model.save('mnist_model.h5')
try:
    model = tf.keras.models.load_model('mnist_model.h5')
except:
    st.error("Model not found. Please save your trained model as 'mnist_model.h5' and place it in the same directory as the app.py file.")
    st.stop()


st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(input_image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image_resize = cv2.resize(grayscale, (28, 28))
    input_image_resize = input_image_resize / 255.0  # Normalize the image
    image_reshaped = np.reshape(input_image_resize, [1, 28, 28])

    # Make prediction
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    st.write(f"The handwritten digit is recognized as: **{input_pred_label}**")
