import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import streamlit as st 
from keras.models import load_model
import tempfile
import pathlib
import os


def get_image_path(img):
    # Create a directory and save the uploaded image.
    file_path = f"data/uploadedImages/{img.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(img.getbuffer())
    return file_path

uploaded_file = st.file_uploader("**Upload a image for classification cat vs Dog**", type= ['png', 'jpg'] )
if uploaded_file is not None:
    # Get actual image file
    bytes_data = get_image_path(uploaded_file)
    st.image(bytes_data)
    image_size=(180, 180)
    img = keras.utils.load_img(bytes_data, target_size=image_size)
   
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
    model = load_model('end.h5')

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    st.write((f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog."))






