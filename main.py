import tkinter as tk
from tkinter import scrolledtext, StringVar, OptionMenu
import threading
import speech_recognition as sr
import torch
import logging
import time
from get_speech_input import get_speech_input
from load_model import load_model
from generate_response import generate_response
from update_status_message import update_status
from on_submit import on_submit
from voice_control import voice_control

# Setup logging to write to a file
logging.basicConfig(filename='voice_control_app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

def main():
    global prompt_entry, status_label, model_var

    window = tk.Tk()
    window.title("GPT-Neo Text Generator")

    model_label = tk.Label(window, text="Select Model:")
    model_label.pack()

    model_options = ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]
    model_var = StringVar(window)
    model_var.set(model_options[1])

    model_dropdown = OptionMenu(window, model_var, *model_options)
    model_dropdown.pack()

    prompt_label = tk.Label(window, text="Enter your prompt:")
    prompt_label.pack()
    prompt_entry = scrolledtext.ScrolledText(window, height=5)
    prompt_entry.pack()

    submit_button = tk.Button(window, text="Generate", command=lambda: on_submit(model_var, prompt_entry, update_status_message, load_model, generate_response))
    submit_button.pack()

    status_label = tk.Label(window, text="Currently sleeping. Say 'AI3' to wake up.", fg="blue")
    status_label.pack()

    logging.info("Starting voice recognition thread")
    listening_thread = threading.Thread(target=lambda: voice_control(update_status, status_label, window, on_submit, prompt_entry), daemon=True)
    listening_thread.start()

    window.mainloop()

if __name__ == "__main__":
    main()
