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

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# Initialize the chat history as Streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "play_audio" not in st.session_state:
    st.session_state.play_audio = True  # Default to play audio

# Title and logo
st.markdown("""
    <style>
        .title-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }

        .title-container img {
            height: 80px;
            border-radius: 50%;
            margin-right: 10px;
        }

        .title-container .title-text {
            font-family: 'Arial', sans-serif;
            color: #D8BFD8;
            font-size: 1.5em;
        }
    </style>
    <div class="title-container">
        <img src="bhav.png" alt="Bengali Logo">
        <div class="title-text">BHAVI (Prototype 2)</div>
    </div>
""", unsafe_allow_html=True)

# Function to convert text to speech
def text_to_speech(text, lang='bn'):
    tts = gTTS(text=text, lang=lang, slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# Function to translate text using mtranslate
def translate_text(text, src_lang='bn', dest_lang='en'):
    return translate(text, dest_lang, src_lang)

# Function to handle form submission
def handle_form_submission(user_prompt):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Translate the user's message to English
    translated_prompt = translate_text(user_prompt, src_lang='bn', dest_lang='en')

    # Send translated message to the LLM and get a response
    messages = [
        {"role": "system", "content": "A helpful polite assistant."},
        {"role": "user", "content": translated_prompt},
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    assistant_response = response.choices[0].message.content

    # Translate the assistant's response back to Bengali
    translated_response = translate_text(assistant_response, src_lang='en', dest_lang='bn')

    st.session_state.chat_history.append({"role": "assistant", "content": translated_response})

    # Display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(translated_response)

    # Conditionally convert response to speech and play it
    if st.session_state.play_audio:
        audio_file = text_to_speech(translated_response, lang='bn')
        st.audio(audio_file, format='audio/mp3')

    # Clear the input field after submission
    return ""  # Return empty string to clear input field

# Main layout
col1, col2 = st.columns([1.0, 0.2])

with col1:
    with st.form(key='input_form'):
        user_prompt = st.text_input("আপনার প্রশ্ন লিখুন...", key="user_prompt")
        submit_button = st.form_submit_button(label='Ask')

    # Handle form submission
    if submit_button and user_prompt:
        user_prompt = handle_form_submission(user_prompt)

# Placeholder for voice input
if st.button('🎤 রেকর্ড করুন'):
    st.write("এটি শিগগিরই আসছে")

# Collapsible input history
with st.expander("ইনপুট ইতিহাস"):
    for message in st.session_state.chat_history:
        role = "ব্যবহারকারী" if message["role"] == "user" else "সহকারী"
        st.markdown(f"**{role}:** {message['content']}")

# Sidebar for additional settings
with st.sidebar:
    st.write("অডিও প্লেব্যাক:")
    st.session_state.play_audio = st.radio(
        "আপনি কি অডিও শুনতে চান?",
        options=[True, False],
        format_func=lambda x: "Yes" if x else "No"
    )

    if st.button("CLEAR"):
        st.session_state.chat_history = []
ś
