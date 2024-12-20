import numpy as np 
import pandas as pd 
import pickle 
import streamlit as st 
import tensorflow
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained VGG16 model
model=pickle.load(open('model.pkl','rb'))

# Define the list of emotions (match your dataset's order)
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Streamlit app title
st.title("Emotion Detection App")
st.subheader('upload a pic and our app will detect your current mood')

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
col1,col2=st.columns([0.7, 0.3],gap='small')

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with col1:
        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=False,width=400)
        
    # Preprocess the image for VGG16
    resized_img = cv2.resize(img, (150, 150))  # Resize to model's input size
    normalized_img = resized_img / 255.0  # Normalize pixel values to [0, 1]
    reshaped_img = np.expand_dims(normalized_img, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(reshaped_img)
    predicted_class = np.argmax(prediction)  # Get the index of the highest score
    emotion = EMOTIONS[predicted_class]  # Get the emotion label

    # Display the predicted emotion
    with col2:
        st.header(f"**Predicted Emotion:** {emotion}")
