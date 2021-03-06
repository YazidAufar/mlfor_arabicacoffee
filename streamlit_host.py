import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

# model = tf.keras.models.load_model("saved_model/mdl_wts.hdf5")
model = tf.keras.models.load_model("saved_model/MobileNetV2-Arabica Coffe-100.0.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}

indo_dict = {0: 'anjing',
             1: 'kuda',
             2: 'gajah',
             3: 'kupu-kupu',
             4: 'ayam',
             5: 'kucing',
             6: 'sapi',
             7: 'kambing',
             8: 'laba-laba',
             9: 'tupai'}

coffee_dict = {0: 'Cerscospora',
               1: 'Healthy',
               2: 'Leaf rust',
               3: 'Miner',
               4: 'Phoma'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # 128 untuk model yang di jupyter notebook Image Detection -  Arabica Coffee
    resized = cv2.resize(opencv_image,(128,128))
    # 224 untuk model yang di google colab
    # resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        # st.title('Sepertinya gambar ini adalah {}'.format(indo_dict[prediction]))
        st.title('Sepertinya gambar ini adalah {}'.format(coffee_dict[prediction]))
