import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("brain_tumor.h5")

classes = ["glioma", "meningioma", "notumor", "pituitary"]

st.title("Brain Tumor Classification using CNN")

st.write("""
Upload a brain MRI image.

Classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary
""")

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", width=300)

    img = img.resize((128, 128))

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted Class: {classes[predicted_index]}")
    st.write(f"Confidence: {confidence:.2f}")

    st.write("Prediction Probabilities:")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {prediction[0][i]:.4f}")