#import subprocess
#subprocess.run(['pip', 'install', 'keras'])

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import joblib

# Load the pre-trained models
cnn_model = tf.keras.models.load_model('./models/cnn_model.h5')
vgg_model = tf.keras.models.load_model('./models/vgg_model.keras')
resnet_model = tf.keras.models.load_model('./models/resnet_model.keras')
rf_model = joblib.load('./models/rf_model.joblib')

# Define class mapping
class_mapping = {
    0: 'Malignant',
    1: 'Benign',
    2: 'Normal'
}

def numerical_to_classname(label):
    return class_mapping[label]

def predict_class_from_sample_image(img, cnn_model, transfer_learning_model, resnet_model, rf_model):
    # Load and preprocess the sample image
    img = image.resize((224, 224))
    #img = image.img_to_array(img)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    # Predict using CNN model
    cnn_prediction = cnn_model.predict(img)
    cnn_class = np.argmax(cnn_prediction)

    # Predict using Transfer Learning (VGG16) model
    vgg_prediction = vgg_model.predict(img)
    vgg_class = np.argmax(vgg_prediction)

    # Predict using Transfer Learning (ResNet50) model
    resnet_prediction = resnet_model.predict(img)
    resnet_class = np.argmax(resnet_prediction)

    # Predict using Random Forest model
    rf_prediction = rf_model.predict(img.reshape(1, -1))[0]

    # Return predictions
    return {
        'CNN': numerical_to_classname(cnn_class),
        'Transfer Learning (VGG16)': numerical_to_classname(vgg_class),
        'Transfer Learning (ResNet50)': numerical_to_classname(resnet_class),
        'Random Forest': numerical_to_classname(rf_prediction)
    }

# Streamlit app
def main():
    st.title("Lung Cancer Detection")
    st.write("Upload an image of a lung scan to predict the lung cancer class.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.write("")
        st.write("Classifying...")

       # predictions_cnn, predictions_vgg, predictions_resnet, predictions_rf = predict_class(image)

        predictions = predict_class_from_sample_image(image, cnn_model, vgg_model, resnet_model, rf_model)

        st.write(predictions)

if __name__ == '__main__':
    main()