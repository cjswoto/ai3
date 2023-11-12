import speech_recognition as sr
import logging

def get_speech_input(recognizer, source, update_status, prompt="Listening..."):
    logging.debug("Start listening for speech")
    audio = recognizer.listen(source)

    try:
        update_status("Working...")
        text = recognizer.recognize_google(audio)
        logging.debug(f"Recognized speech: {text}")
        return text
    except Exception as e:
        logging.error(f"Error recognizing speech: {e}")
        return ""
