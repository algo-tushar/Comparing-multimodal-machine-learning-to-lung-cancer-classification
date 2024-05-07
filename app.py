import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import joblib

vgg_model = tf.keras.models.load_model('./models/vgg_model.h5')

# Define class mapping
class_mapping = {
    0: 'Malignant',
    1: 'Benign',
    2: 'Normal'
}

def numerical_to_classname(label):
    return class_mapping[label]

def predict_class_from_sample_image(img, vgg_model):
    # Load and preprocess the sample image
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    # Predict using Transfer Learning (VGG16) model
    vgg_prediction = vgg_model.predict(img)
    vgg_class = np.argmax(vgg_prediction)
    vgg_accuracy = vgg_prediction[0][vgg_class]  # Probability of the predicted class

    # Return predictions with accuracy scores
    return {
        'class': numerical_to_classname(vgg_class),
        'accuracy': vgg_accuracy
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
        
        predictions = predict_class_from_sample_image(image, vgg_model)

        # Display the JSON data using Streamlit components
        st.title("Predicted Value:")
        st.text("Class: " + str(predictions["class"]))
        st.text("Accuracy: " + str(predictions["accuracy"]))

if __name__ == '__main__':
    main()