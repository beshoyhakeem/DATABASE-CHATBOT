import speech_recognition as sr
import tempfile
import os

def record_and_recognize():
    """
    Records audio from the microphone and converts it to text using Google Web Speech API.
    Returns:
        str: Recognized text or an error message.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording... Speak now!")
        audio = recognizer.listen(source)
        print("Recording stopped. Processing...")

        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            return f"Recognized Text: {text}"
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an issue with the speech recognition service."