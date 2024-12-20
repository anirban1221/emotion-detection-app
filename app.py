import numpy as np 
import pandas as pd 
import pickle 
import streamlit as st 
import tensorflow as tf
import streamlit as st
import numpy as np
import cv2

# Define the list of emotions (match your dataset's order)
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Load the TFLite model
@st.cache_resource
def load_tflite_model(model_path):
    with open(model_path, 'rb') as f:
        model_content = f.read()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model("emotion_detector.tflite")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit app title
st.title("Emotion Detection App")
st.subheader('Let our app predict your current mood,upload a pic here:')

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

col1,col2=st.columns([0.7,0.3],gap='small')

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=False,width=400)

    # Preprocess the image for the TFLite model
    resized_img = cv2.resize(img, (150, 150))  # Resize to model's input size
    normalized_img = resized_img / 255.0  # Normalize pixel values to [0, 1]
    input_data = np.expand_dims(normalized_img, axis=0).astype(np.float32)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the prediction
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = np.argmax(predictions)  # Get the index of the highest score
    emotion = EMOTIONS[predicted_class]  # Get the emotion label

    # Display the predicted emotion
    with col2:
        st.header(f"**Predicted Emotion:** {emotion}")

    # Optionally display probabilities for all emotions
    st.write("**Prediction Probabilities:**")
    for i, emo in enumerate(EMOTIONS):
        st.write(f"{emo}: {predictions[i]:.2f}")
