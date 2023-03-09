import streamlit as st
from PIL import Image
import time
import numpy as np
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # support vector classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from extract import final_prediction
from audio_input import create_audio_file

st.title(" :thought_balloon: Speech Emotion Recognition")

# st.write("""
# # Explore different classifier
# Which one is the best?
# """)

image = Image.open('images\smileyfacesboxes.jpg')
st.sidebar.image(image, caption='')
dataset_name = st.sidebar.selectbox("Select Dataset", ("RAVDESS", "SUBESCOO", "Wine Dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("MLPClassifier", "SVM", "Random Forest"))

# Show setting
st.sidebar.write("Options")
wave_plot = st.sidebar.checkbox("Wave Form")
spectrogram = st.sidebar.checkbox("Spectrogram")

def get_dataset(dataset_name):
    """
    Loading Dataset
    """
    if (dataset_name == "Iris"):
        data = datasets.load_iris()
    elif (dataset_name == "Breast Cancer"):
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y

def add_parameter_ui(clf_name):

    params = {}
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
    return clf

# Informations for using the web application
st.info('''
:mag_right: **Why Speech Emotion Recognition is necessary?**

Understanding human emotion by machine can help it to take better decisions
in different situations.

:page_with_curl: Steps
+ Please select an audio file from your device.
+ Or, you can also record audio in real-time and see prediction of emotion.
+ Or, you can use the test file for expreiments.
+ After choosing the audio file, press the prediction button.
''')

# 2 columns for uploading file
c1, c2 = st.columns([3, 1])
with c1:
    uploaded_file = st.file_uploader("Upload audio file for predicting it's emotion.")
    audiofile = "D:\p350_test\F_01_OISHI_S_1_ANGRY_4.wav"

    if uploaded_file:
        st.audio(uploaded_file)
        audiofile = uploaded_file.name

        if st.button("Predict Tag", type="primary", key='b1'):
            with st.spinner('Wait for prediction...'):
                time.sleep(4)
        
            predictions = final_prediction(audiofile)
            for i in range(len(predictions)):
                st.success(predictions[i])

    st.markdown("---")
    st.write(" :singer: **Record audio using microphone.** ")
    image = Image.open('D:\p350_test\images\mic1.png')
    new_image = image.resize((25, 25))
    st.image(new_image, caption='')

    # Initialize state
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False

    def callback():
        st.session_state.button_clicked = True
    
    if (st.button("Start Recording", type="primary", key='b2', on_click=callback) or st.session_state.button_clicked):

        st.info("Recording audio is started...(duration 5s)")
        create_audio_file()
        audiofile = "recording0.wav"
        st.audio(audiofile)
        st.success("Successfully recorded audio")

        if (st.button("Predict Tag", type="primary", key='b3')):
            with st.spinner('Wait for prediction...'):
                time.sleep(2)
        
            predictions = final_prediction(audiofile)
            for i in range(len(predictions)):
                st.success(predictions[i])

    st.markdown("---")


with c2:
    st.write("")
        
    
