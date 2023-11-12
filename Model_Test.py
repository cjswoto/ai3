import tkinter as tk
from tkinter import scrolledtext, StringVar, OptionMenu
import threading
import speech_recognition as sr
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import logging
import time

# Setup logging to write to a file
logging.basicConfig(filename='voice_control_app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Voice input function
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

# Voice control function
def voice_control(update_status, window, on_submit_callback):
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

# GPT-Neo model functions
def load_model(model_name):
    logging.debug(f"Loading model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True)
    logging.debug("Model loaded successfully")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p,
                      num_return_sequences, no_repeat_ngram_size):
    logging.debug("Generating response")
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_beams=1,
        length_penalty=1.0
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    logging.debug("Response generated")
    return response

def update_status_message(message):
    logging.debug(f"Updating status message: {message}")
    status_label.config(text=message)
    status_label.update_idletasks()

def on_submit():
    logging.debug("Submit button clicked")
    model_name = model_var.get()
    prompt = prompt_entry.get("1.0", tk.END).strip()
    model, tokenizer = load_model(model_name)

    max_length = 50
    temperature = 0.7
    top_k = 50
    top_p = 0.95
    num_return_sequences = 1
    no_repeat_ngram_size = 2

    response = generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p, num_return_sequences, no_repeat_ngram_size)

    update_status_message("Response: " + response)

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

    submit_button = tk.Button(window, text="Generate", command=on_submit)
    submit_button.pack()

    status_label = tk.Label(window, text="Currently sleeping. Say 'AI3' to wake up.", fg="blue")
    status_label.pack()

    logging.info("Starting voice recognition thread")
    listening_thread = threading.Thread(target=lambda: voice_control(update_status_message, window, on_submit), daemon=True)
    listening_thread.start()

    window.mainloop()

if __name__ == "__main__":
    main()
