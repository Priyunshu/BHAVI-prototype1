import os
import json
import streamlit as st
from gtts import gTTS
from io import BytesIO
from mtranslate import translate
from groq import Groq

# Streamlit page configuration
st.set_page_config(
    page_title="BHAVI (Prototype 2)",
    page_icon="",
    layout="centered"
)

# Load configuration data
working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")
with open(config_file_path, "r") as config_file:
    config_data = json.load(config_file)

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="bhav.png" alt="Bengali Logo" style="height: 60px; border-radius: 10px;">
        <div style="font-family: 'Arial', sans-serif; color: #000; font-size: 1.5em; margin-left: 10px;">
            BHAV (Prototype 2)
        </div>
    </div>
""", unsafe_allow_html=True)

# Function to convert text to speech
def text_to_speech(text, lang='bn'):
    tts = gTTS(text=text, lang=lang, slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# Function to translate text
def translate_text(text, src_lang='bn', dest_lang='en'):
    return translate(text, dest_lang, src_lang)

# Handle form submission
def handle_form_submission(user_prompt):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Translate user message to English
    translated_prompt = user_prompt

    # Send translated message to LLM
    messages = [
        {"role": "system", "content": "A helpful assistant."},
        {"role": "user", "content": translated_prompt},
    ]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    assistant_response = response.choices[0].message.content

    # Translate response back to Bengali
    translated_response = translate_text(assistant_response, src_lang='en', dest_lang='bn')
    st.session_state.chat_history.append({"role": "assistant", "content": translated_response})

    # Display response
    st.markdown(f"**Assistant:** {translated_response}")

    # Convert response to speech
    audio_file = text_to_speech(translated_response, lang='bn')
    st.audio(audio_file, format='audio/mp3')

    return ""  # Clear input field

# Main layout
with st.form(key='input_form'):
    user_prompt = st.text_input("আপনার প্রশ্ন লিখুন...", key="user_prompt")
    submit_button = st.form_submit_button(label='Ask')

    # Handle form submission
    if submit_button and user_prompt:
        user_prompt = handle_form_submission(user_prompt)

# Collapsible input history
with st.expander("Input History"):
    for message in st.session_state.chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        st.markdown(f"**{role}:** {message['content']}")
ś
