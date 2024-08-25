import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from gtts import gTTS
import io
from googletrans import Translator

# Define model and tokenizer names
dialo_model_name = "microsoft/DialoGPT-medium"
trans_model_name = "csebuetnlp/banglat5_nmt_en_bn"

# Cache the DialoGPT model and tokenizer to optimize memory usage
@st.cache_resource
def load_dialo_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(dialo_model_name)
    model = AutoModelForCausalLM.from_pretrained(dialo_model_name)
    return tokenizer, model

dialo_tokenizer, dialo_model = load_dialo_model_and_tokenizer()

# Cache the translation model and tokenizer to optimize memory usage
@st.cache_resource
def load_translation_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(trans_model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)
    return tokenizer, model

trans_tokenizer, trans_model = load_translation_model_and_tokenizer()

# Translator for Bengali to English
translator = Translator()

# Function to generate a response from DialoGPT
def generate_response(prompt):
    new_user_input_ids = dialo_tokenizer.encode(prompt + dialo_tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = dialo_model.generate(new_user_input_ids, max_length=500, pad_token_id=dialo_tokenizer.eos_token_id)  # Reduce max length
    response = dialo_tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Function to translate English text to Bengali
def translate_to_bengali(text):
    input_ids = trans_tokenizer(text, return_tensors="pt").input_ids
    generated_tokens = trans_model.generate(input_ids, max_length=100)  # Reduce max length
    decoded_tokens = trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return decoded_tokens

# Function to convert text to speech in Bengali
def text_to_speech(text):
    tts = gTTS(text, lang='bn')  # 'bn' for Bengali
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

# Streamlit UI Styling
st.markdown(
    """
    <style>
    /* Styles here */
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>BHAVI <span class='small'>(prototype 1)</span></div>", unsafe_allow_html=True)  # Updated title

# Input and button arrangement
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    user_input = st.text_input("", key="text_input", placeholder="write to chitchat...", label_visibility="collapsed")
   
    col1, col2 = st.columns([0.1, 0.1])
    with col1:
        record_button = st.button(" 🗣️ ", key="record", help="Record Speech Input")  # Microphone icon button

    with col2:
        send_button = st.button("let's go")

    st.markdown("</div>", unsafe_allow_html=True)

if record_button:
    st.write("Will arrive soon...")  # Temporary message for speech input

if send_button:
    if user_input:
        response = generate_response(user_input)
        bengali_response = translate_to_bengali(response)
       
        st.markdown("<div class='response'><b>প্রতিক্রিয়া:</b></div>", unsafe_allow_html=True)
        st.write(bengali_response)
       
        audio_file = text_to_speech(bengali_response)
        st.audio(audio_file, format='audio/mp3')
    else:
        st.write("একটি বার্তা লিখুন দয়া করে..")

# Clear conversation and cache button
if st.button("Clear Memory 🧹"):
    st.cache_resource.clear()  # Clears all cached resources
    st.write("Memory cleared and resources reset.")
