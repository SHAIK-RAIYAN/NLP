import streamlit as st
import numpy as np
import librosa
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from keras.models import load_model
import os

# Custom CSS styling
st.markdown(
    """
    <style>
    /* Full-page background image */
    body {
        background-image: url('https://github.com/SHAIK-RAIYAN/NLP/blob/main/pexels-eberhardgross-691668.jpg');
        background-size: cover;
        background-attachment: fixed;
        color: white;  /* Default text color */
    }
    
    /* Outer container with border and blur */
    .outer-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);  /* Background blur */
        background: rgba(0, 0, 0, 0.6);  /* Dark overlay with transparency */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Title styling */
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 0.5em 0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #007ACC;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 8px;
        transition: 0.3s;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #005B99;
        color: #fff;
    }
    
    /* Dropdown styling */
    .stSelectbox {
        color: white;
        font-size: 1.1em;
        font-weight: 500;
        padding: 10px 0;
    }
    
    /* Text styling */
    .section-text {
        color: white;
        font-size: 1.2em;
        font-weight: 500;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the emotion recognition model
emotion_model = load_model('emotion_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize translator and speech recognizer
recognizer = sr.Recognizer()
translator = Translator()

LANGUAGES = {
    'en': 'English', 'zh-cn': 'Mandarin Chinese', 'hi': 'Hindi', 'es': 'Spanish',
    'fr': 'French', 'ar': 'Arabic', 'it': 'Italian', 'ru': 'Russian', 'pt': 'Portuguese',
    'de': 'German', 'ja': 'Japanese'
}

# Function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, 40, 1)

# Function to predict emotion
def predict_emotion(audio_path):
    mfcc = extract_mfcc(audio_path)
    prediction = emotion_model.predict(mfcc)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

# Function to translate text and generate audio
def translate_and_generate_audio(text, target_lang):
    translated_text = translator.translate(text, dest=target_lang).text
    tts = gTTS(text=translated_text, lang=target_lang)
    audio_path = "translated_audio.mp3"
    tts.save(audio_path)
    return translated_text, audio_path

# Function to recognize speech from an audio file
def recognize_speech_from_file(file_path):
    audio = AudioSegment.from_file(file_path).set_channels(1)
    temp_wav = "temp_audio.wav"
    audio.export(temp_wav, format="wav")
    
    with sr.AudioFile(temp_wav) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    
    os.remove(temp_wav)  # Clean up temporary file
    return text

# Streamlit app
def main():
    st.markdown("<div class='outer-container'><div class='title'>Audio Translation and Emotion Detection System</div>", unsafe_allow_html=True)

    target_lang = st.selectbox("Select a target language:", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    
    st.write("<div class='section-text'>### Choose an Input Method</div>", unsafe_allow_html=True)

    # Define columns for button layout
    col1, col2 = st.columns(2)

    # "Real-Time Translation" button (currently only works locally)
    with col1:
        if st.button("Real-Time Translation"):
            st.write("Real-time translation requires local microphone input, which is not supported on Streamlit Cloud.")
            st.write("You can run this app locally to use real-time translation.")

    # "Upload an Audio File" button
    with col2:
        audio_file = st.file_uploader("Upload an audio file (wav or mp3)", type=["wav", "mp3"])
        
        if audio_file is not None:
            # Save the uploaded file temporarily
            with open("uploaded_audio.wav", "wb") as f:
                f.write(audio_file.read())

            try:
                # Recognize and display text
                detected_text = recognize_speech_from_file("uploaded_audio.wav")
                st.write("**Detected Text:**", detected_text)

                # Predict and display emotion
                emotion = predict_emotion("uploaded_audio.wav")
                st.write("**Detected Emotion:**", emotion)

                # Translate text and display translated text and audio
                translated_text, translated_audio_path = translate_and_generate_audio(detected_text, target_lang)
                st.write("**Translated Text:**", translated_text)
                st.write("**Translated Audio:**")
                st.audio(translated_audio_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                # Clean up uploaded audio after processing
                if os.path.exists("uploaded_audio.wav"):
                    os.remove("uploaded_audio.wav")

    st.markdown("</div>", unsafe_allow_html=True)  # Close the outer container div

if __name__ == "__main__":
    main()
