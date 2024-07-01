import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import utils_harmful
import embeddings
import formating
import models
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
from typing import List
import pickle
import base64
import buildGraph

# nltk.download('stopwords')

# Define maximum sequence length
max_sequence_length = 50

# Function to preprocess the tweet
def preprocess_tweet(tweet):
    if not st.session_state.load_model:
        with open('tokenizer.pickle', 'rb') as handle:
            st.session_state.tokenizer = pickle.load(handle)

    tweet = pd.Series(tweet, name='tweet')
    df_tweet = formating.clean_text(tweet)
    sequences = st.session_state.tokenizer.texts_to_sequences(df_tweet)  # Convert texts to sequences
    tweet = pad_sequences(sequences, maxlen=50)
    return tweet

# Function to make predictions
def predict(tweet):
    if not st.session_state.load_model:
        st.session_state.model = tf.keras.models.load_model('model_glove.h5')

    processed_tweet = preprocess_tweet(tweet)
    prediction = st.session_state.model.predict(tf.constant(processed_tweet))
    return prediction

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image.jpg")
img2 = get_img_as_base64("social-media-background-twitte.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:social-media-background-twitte/png;base64,{img2}");
background-size: 120%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: bottom left; 
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stHeading"] {{
font-size: 100px;
font-weight: bold;
text-align: center;
color: #fffff;}}

[data-testid="stToolbar"] {{
right: 2rem;
}}


.custom-container {{
    background-color: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    margin-top: 20px;
}}

.text-font {{
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    color: #fffff;
}}

.text-font2{{
    font-size: 18px;
    font-weight: bold;
    text-align: left;
    color: #fffff;
}}

</style>
"""

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #465e9c;
    color:#ffffff;
    width: 50%;
    margin: 0 auto;
    display: block;
}
div.stButton > button:hover {
    background-color: #FF0000;
    color:##ff99ff;
    }
</style>""", unsafe_allow_html=True)

# Functie pentru a adauga CSS personalizat
def add_custom_css():
    custom_css = """
    <style>
    /* Customize the handle of the slider */
    .stSlider > div:nth-child(1) > div > div > div {
        background-color: #00000!important;  /* Change color */
        width: 30px !important;  /* Set custom width */
        height: 30px !important;  /* Set custom height */
        border-radius: 50%;  /* Make it circular */
    }
    /* Customize the track of the slider */
    .stSlider > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-bm2z3a.ea3mdgi8 > div.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.st-emotion-cache-0.e1f1d6gn0 > div > div > div:nth-child(3) > div {
        background-color: #FF5733 !important;
    }
    /* Customize the slider track */
    .stSlider > div:nth-child(1) > div {
        padding: 10px 0 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state variables
if 'tweet_text' not in st.session_state:
    st.session_state.tweet_text = ""
    
if 'load_model' not in st.session_state:
    st.session_state.load_model = False

if 'show_button' not in st.session_state:
    st.session_state.show_button = False

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.header("")

st.title('*Tweet Harmfulness Prediction and Mitigation*' )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="text-font2">First, pick an user:</div>', unsafe_allow_html=True)

value = st.selectbox("Pick a user", ["Thrilla_dondada", "managingmadrid", "delicategnf", "MandiCandi1234"], label_visibility= 'hidden')

st.session_state.tweet_text = st.text_area('#', value=st.session_state.tweet_text, placeholder="Type your tweet here...")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Use a custom container with a special design
with st.container():
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="text-font">Select prediction confidence threshold:</div>', unsafe_allow_html=True)
        # add_custom_css()
    with col2:
        threshold = st.slider(':blue[Select a value]', 0.0, 1.0, 0.4, key='threshold_slider', label_visibility = 'hidden')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.button('Predict'):
    if st.session_state.tweet_text:
        with st.spinner('Analyzing the tweet...'):
            try:
                prediction = predict(st.session_state.tweet_text)
                print(prediction)
                prediction = float(prediction[0][0])
                st.session_state.load_model = True  

                if prediction > threshold:
                    st.markdown(f'<div class="custom-container"><div class="stAlert">⚠️ The tweet is harmful.</div></div>', unsafe_allow_html=True)
                    st.session_state.show_button = True
                    st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="custom-container"><div class="stAlert">✅ The tweet is not harmful.</div></div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
        
    else:
        st.warning('ℹ️ Please enter a tweet to make a prediction.')

# st.write(pressed)
if 'show_button' in st.session_state or st.session_state.show_button:
    if st.button('Immunize!!'):
        with st.spinner('Running immunization...'):
            graph = buildGraph.buildGraph()
            simulator = buildGraph.SimpleSimulator(graph, [value])
            saved, active1, active2 = simulator.simulate(buildGraph.apply_netShield_on_reachable_subgraph(graph, [value], 2))
            st.markdown(f'<div class="custom-container"><div class="stAlert">✅ Managed to save {saved} users.</div></div>', unsafe_allow_html=True)    

st.markdown("""
    <hr>
    <p style="text-align: center;">Created by <strong>Ana Constantinescu</strong>. Powered by University Politehnica of Bucharest.</p>
""", unsafe_allow_html=True)
