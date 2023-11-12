import speech_recognition as sr

def get_speech_input(update_status, prompt="Listening..."):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        update_status(prompt)
        audio = recognizer.listen(source)

        try:
            update_status("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            update_status("")  # Reset status
            return text
        except Exception as e:
            print("Error:", str(e))
            update_status("")  # Reset status
            return ""

def listen_for_command(update_status, on_submit_callback):
    update_status("Command mode activated. Please speak your command.")
    command = get_speech_input(update_status, "Listening for command...")

    if command.lower() == 'submit':
        update_status("Executing submit...")
        on_submit_callback()
        return

    update_status("Command mode waiting for 'submit' to confirm...")
    confirmation = get_speech_input(update_status, "Say 'submit' to confirm...")

    if confirmation.lower() == 'submit':
        update_status("Command confirmed. Executing submit...")
        on_submit_callback()
    else:
        update_status("Command cancelled.")

def listen_for_wake_word(wake_word, on_submit_callback, update_status):
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        while True:
            try:
                update_status("Listening for wake word...")
                audio = recognizer.listen(source)
                recognized_text = recognizer.recognize_google(audio).lower()
                if wake_word.lower() in recognized_text:
                    listen_for_command(update_status, on_submit_callback)
            except sr.UnknownValueError:
                pass  # Ignore unrecognized speech
            except sr.RequestError as e:
                print("Could not request results; check your internet connection")
