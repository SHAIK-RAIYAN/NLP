import streamlit as st
import numpy as np
import librosa
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from keras.models import load_model
import os

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
    st.write(f"Detected Emotion: {emotion}")
    return emotion

# Function to translate text and generate audio
def translate_and_generate_audio(text, target_lang):
    translated_text = translator.translate(text, dest=target_lang).text
    st.write(f"Translated Text: {translated_text}")
    tts = gTTS(text=translated_text, lang=target_lang)
    audio_path = "translated_audio.mp3"
    tts.save(audio_path)
    st.audio(audio_path)

# Function to recognize speech from an audio file
def recognize_and_translate(file_path, target_lang):
    audio = AudioSegment.from_file(file_path).set_channels(1)
    temp_wav = "temp_audio.wav"
    audio.export(temp_wav, format="wav")
    
    with sr.AudioFile(temp_wav) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        st.write(f"Recognized Text: {text}")
        translate_and_generate_audio(text, target_lang)

    os.remove(temp_wav)  # Clean up temporary file

# Streamlit app
def main():
    st.title("Audio Translation and Emotion Detection System")

    target_lang = st.selectbox("Select a target language:", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    
    st.write("### Choose Input Method:")
    input_method = st.radio("Select an input method", ("Upload an Audio File", "Simulated Real-Time Translation"))

    if input_method == "Upload an Audio File":
        audio_file = st.file_uploader("Upload an audio file (wav or mp3)", type=["wav", "mp3"])
        
        if audio_file is not None:
            with open("uploaded_audio.wav", "wb") as f:
                f.write(audio_file.read())

            st.write("Detecting emotion...")
            predict_emotion("uploaded_audio.wav")
            st.write("Translating recognized text...")
            recognize_and_translate("uploaded_audio.wav", target_lang)

            os.remove("uploaded_audio.wav")  # Clean up after processing

    elif input_method == "Simulated Real-Time Translation":
        st.write("Real-time translation requires local microphone input, which is not supported on Streamlit Cloud.")
        st.write("You can run this app locally to use real-time translation.")

if __name__ == "__main__":
    main()
