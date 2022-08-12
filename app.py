from pydoc import doc
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import h5py
image = Image.open('doc.png')
st.image(image)
st.title("Breast Cancer Detection App")
st.header("This prediction was made using Fine Needle Aspirate Images of sample patient data from the UCI Wisconsin dataset")
st.text("Made by Tatap Perets (tatapperets@gmail.com)")
# Load the model
def teachable_machine_classification(img, keras_model):
    model = load_model('keras_model.h5')

    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    print(prediction)
    return np.argmax(prediction) # return position of the highest probability


uploaded_file = st.file_uploader("Upload the FNA biopsied image ...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image', use_column_width=True)
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.success("The image is most likely benign")
    else:
        st.error("The image is most likely malignant")