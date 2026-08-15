import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("cat_dog_mobilenetv2.h5")

st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader(
    "Upload a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.image(image, caption="Uploaded Image")

    # Preprocessing
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(image, verbose=0)
        probability = prediction[0][0]
        st.write("Prediction probability:", probability)

        if probability > 0.5:
            st.success("Dog")
        else:
            st.success("Cat")