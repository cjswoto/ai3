
import speech_recognition as sr
import logging

def configure_recognizer():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    return recognizer

def get_speech_input(recognizer, source):
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        logging.error("No speech could be understood.")
        return ""
    except Exception as e:
        logging.error(f"Error in get_speech_input: {e}", exc_info=True)
        return ""
