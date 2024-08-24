import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from gtts import gTTS
import io
import speech_recognition as sr
from googletrans import Translator

# Define model and tokenizer names
dialo_model_name = "microsoft/DialoGPT-medium"
trans_model_name = "csebuetnlp/banglat5_nmt_en_bn"

# Load DialoGPT model and tokenizer
def load_dialo_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(dialo_model_name)
    model = AutoModelForCausalLM.from_pretrained(dialo_model_name)
    return tokenizer, model

dialo_tokenizer, dialo_model = load_dialo_model_and_tokenizer()

# Load Bengali translation model and tokenizer
def load_translation_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(trans_model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)
    return tokenizer, model

trans_tokenizer, trans_model = load_translation_model_and_tokenizer()

# Initialize chat history
chat_history_ids = torch.tensor([]).long()  # Initialize as an empty tensor

# Translator for Bengali to English
translator = Translator()

# Function to generate a response from DialoGPT
def generate_response(prompt):
    global chat_history_ids
    new_user_input_ids = dialo_tokenizer.encode(prompt + dialo_tokenizer.eos_token, return_tensors='pt')
    
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.size(0) > 0 else new_user_input_ids
    chat_history_ids = dialo_model.generate(bot_input_ids, max_length=1000, pad_token_id=dialo_tokenizer.eos_token_id)
    
    response = dialo_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Function to translate English text to Bengali
def translate_to_bengali(text):
    input_ids = trans_tokenizer(text, return_tensors="pt").input_ids
    generated_tokens = trans_model.generate(input_ids)
    decoded_tokens = trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return decoded_tokens

# Function to convert text to speech in Bengali
def text_to_speech(text):
    tts = gTTS(text, lang='bn')  # 'bn' for Bengali
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

# Function to recognize speech in Bengali
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='bn-IN')  # Bengali language
        return text
    except sr.UnknownValueError:
        return "Sorry, I did not understand that."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."

# Streamlit UI Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Sans+MS:wght@400;700&display=swap');  /* Import Comic Sans font */
    .main {
        background-color: #2E2E2E;  /* Dark grey background */
        color: #FFFFFF;  /* White text */
        font-family: 'Comic Sans MS', sans-serif;  /* Apply Comic Sans font */
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #FFFFFF;  /* White text */
        font-family: 'Comic Sans MS', sans-serif;  /* Apply Comic Sans font */
        font-weight: bold;  /* Bold title text */
    }
    .stButton button {
        background-color: #FFDB58;  /* Mustard color for send button */
        color: #000000;  /* Black text */
        font-weight: bold;  /* Bold text */
        border-radius: 20px;  /* Rounded button shape */
        padding: 10px 20px;
        font-size: 16px;
        font-family: 'Comic Sans MS', sans-serif;  /* Apply Comic Sans font */
        margin-top: 0.5cm;  /* Move down by 0.5 cm */
        border: 2px solid #000000;  /* Black border */
    }
    .stButton button[data-baseweb="button"] {
        background-color: #FFFFFF;  /* White color for record button */
        border-color: #000000;  /* Black border for record button */
        color: #000000;  /* Black text */
    }
    .message {
        font-size: 1.2em;
        color: #FFFFFF;  /* White text */
        font-weight: bold;  /* Bold message text */
    }
    .response {
        background-color: #4F4F4F;  /* Lighter grey background */
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        color: #FFFFFF;  /* White text */
        font-weight: bold;  /* Bold response text */
    }
    .small {
        font-size: 0.6em;
        color: #FFFFFF;  /* White text */
        font-weight: bold;  /* Bold small text */
    }
    .input-container {
        display: flex;
        align-items: center;
    }
    .input-container input {
        flex-grow: 1;
    }
    .input-container button {
        margin-left: 5px;  /* Reduce gap between buttons */
        margin-top: 0.5cm;  /* Move buttons down by 0.5 cm */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>BHAVI <span class='small'>(prototype 1)</span></div>", unsafe_allow_html=True)  # Updated title

# Display chat history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  # Initialize conversation history

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
    st.write(" কথা বলুন ")
    speech_input = recognize_speech()
    st.write(f"আমি যা শুনেছি: {speech_input}")
    
    # Translate Bengali speech input to English
    english_input = translator.translate(speech_input, src='bn', dest='en').text
    st.write(f"Translated Speech Input : {english_input}")
    
    if english_input:
        response = generate_response(english_input)
        bengali_response = translate_to_bengali(response)
        
        # Update conversation history
        st.session_state.conversation_history.append((speech_input, response))
        
        st.markdown("<div class='response'><b>প্রতিক্রিয়া:</b></div>", unsafe_allow_html=True)
        st.write(bengali_response)
        
        audio_file = text_to_speech(bengali_response)
        st.audio(audio_file, format='audio/mp3')

if send_button:
    if user_input:
        response = generate_response(user_input)
        bengali_response = translate_to_bengali(response)
        
        # Update conversation history
        st.session_state.conversation_history.append((user_input, response))
        
        st.markdown("<div class='response'><b>প্রতিক্রিয়া:</b></div>", unsafe_allow_html=True)
        st.write(bengali_response)
        
        audio_file = text_to_speech(bengali_response)
        st.audio(audio_file, format='audio/mp3')
    else:
        st.write("একটি বার্তা লিখুন দয়া করে..")

# Clear conversation button
if st.button("clear 🧹"):
    st.session_state.conversation_history = []  # Clear conversation history
    st.write("Conversation cleared.")  # Notify user that conversation has been cleared
