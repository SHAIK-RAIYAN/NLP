import streamlit as st
import numpy as np
import librosa
import os
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from keras.models import load_model
from io import BytesIO

# Load the emotion recognition model
emotion_model = load_model('emotion_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

recognizer = sr.Recognizer()
translator = Translator()

LANGUAGES = {
    'en': 'English', 'zh-cn': 'Mandarin Chinese', 'hi': 'Hindi', 'es': 'Spanish',
    'fr': 'French', 'ar': 'Arabic', 'it': 'Italian', 'ru': 'Russian', 'pt': 'Portuguese',
    'de': 'German', 'ja': 'Japanese'
}

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, 40, 1)

def predict_emotion(audio_path):
    mfcc = extract_mfcc(audio_path)
    prediction = emotion_model.predict(mfcc)
    emotion = emotion_labels[np.argmax(prediction)]
    st.write(f"Detected Emotion: {emotion}")
    return emotion

def translate_and_generate_audio(text, target_lang):
    translated_text = translator.translate(text, dest=target_lang).text
    st.write(f"Translated Text: {translated_text}")
    tts = gTTS(text=translated_text, lang=target_lang)
    audio_path = "translated_audio.mp3"
    tts.save(audio_path)
    st.audio(audio_path)

def translate_from_mp3(file_path, target_lang):
    audio = AudioSegment.from_mp3(file_path).set_channels(1)
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        st.write(f"Recognized from MP3: {text}")
        translate_and_generate_audio(text, target_lang)

def recognize_and_translate(target_lang):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        text = recognizer.recognize_google(audio)
        st.write(f"Recognized: {text}")
        translate_and_generate_audio(text, target_lang)

def main():
    st.title("Real-time and MP3 Translation System")

    target_lang = st.selectbox("Select a target language:", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

    st.write("### Choose Input Method:")
    real_time_button = st.button("Real-time Voice Translation")
    mp3_file = st.file_uploader("Upload MP3 file for translation", type="mp3")
    emotion_file = st.file_uploader("Upload Audio file for Emotion Detection", type=["wav", "mp3"])

    if real_time_button:
        st.write("Listening for real-time translation...")
        recognize_and_translate(target_lang)

    if mp3_file is not None:
        st.write("Translating from uploaded MP3 file...")
        with open("uploaded_audio.mp3", "wb") as f:
            f.write(mp3_file.read())
        translate_from_mp3("uploaded_audio.mp3", target_lang)

    if emotion_file is not None:
        st.write("Predicting emotion from uploaded audio file...")
        with open("emotion_audio.wav", "wb") as f:
            f.write(emotion_file.read())
        predict_emotion("emotion_audio.wav")

if __name__ == "__main__":
    main()
