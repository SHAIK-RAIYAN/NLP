import streamlit as st
import numpy as np
import librosa
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from keras.models import load_model

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
    return emotion

def translate_and_generate_audio(text, target_lang):
    translated_text = translator.translate(text, dest=target_lang).text
    st.write(f"Translated Text: {translated_text}")
    tts = gTTS(text=translated_text, lang=target_lang)
    audio_path = "translated_audio.mp3"
    tts.save(audio_path)
    st.audio(audio_path)
    return translated_text

def process_audio(file_path, target_lang):
    # Convert to wav for speech recognition
    audio = AudioSegment.from_mp3(file_path).set_channels(1)
    audio.export("temp.wav", format="wav")

    # Speech recognition
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        st.write(f"Recognized Text: {text}")

    # Translate and play the translated audio
    translated_text = translate_and_generate_audio(text, target_lang)

    # Emotion detection
    emotion = predict_emotion("temp.wav")
    st.write(f"Detected Emotion: {emotion}")

def main():
    st.title("Audio Translation and Emotion Detection System")

    target_lang = st.selectbox("Select a target language:", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    audio_file = st.file_uploader("Upload Audio file for Translation and Emotion Detection", type=["mp3", "wav"])

    if audio_file is not None:
        st.write("Processing uploaded audio file...")
        with open("uploaded_audio.mp3", "wb") as f:
            f.write(audio_file.read())
        process_audio("uploaded_audio.mp3", target_lang)

if __name__ == "__main__":
    main()
