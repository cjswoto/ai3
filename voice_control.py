import logging
import speech_recognition as sr
import tkinter as tk

def voice_control(update_status, status_label, window, on_submit_callback, prompt_entry):
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    mode = "sleeping"  # Modes: sleeping, listening, command

    with sr.Microphone() as source:
        while True:
            logging.debug(f"Current mode: {mode}")
            if mode == "sleeping":
                update_status("Say 'AI3' and I will listen")
                logging.debug("System is sleeping, waiting for 'AI3'")
                command = get_speech_input(recognizer, source, update_status)
                if "ai3" in command.lower():
                    mode = "listening"
                    logging.debug("Switched to listening mode")
            elif mode == "listening":
                update_status("Listening Mode: Speak your input")
                logging.debug("Listening for user input")
                command = get_speech_input(recognizer, source, update_status)
                if "ai3 command" in command.lower():
                    mode = "command"
                    update_status("Command Mode: Speak 'Submit', 'Exit', or 'Close'")
                    logging.debug("Switched to command mode")
                elif "ai3 sleep" in command.lower():
                    mode = "sleeping"
                    update_status("Say 'AI3' and I will listen")
                    logging.debug("Switched to sleeping mode")
                else:
                    prompt_entry.insert(tk.END, command + "\n")
            elif mode == "command":
                update_status("Command Mode: Speak 'Submit', 'Exit', or 'Close'")
                logging.debug("In command mode, waiting for 'Submit', 'Exit', or 'Close'")
                command = get_speech_input(recognizer, source, update_status).lower()
                if "submit" in command:
                    on_submit_callback()
                    mode = "sleeping"
                    logging.debug("Executed submit and switched to sleeping mode")
                elif "exit" in command:
                    mode = "sleeping"
                    logging.debug("Exited command mode and switched to sleeping mode")
                elif "close" in command:
                    logging.info("Closing application")
                    window.destroy()
                    break
