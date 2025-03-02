import tkinter as tk
from tkinter import scrolledtext, StringVar, OptionMenu
import pyttsx3
import speech_recognition as sr
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import threading
import torch
import webbrowser
import logging
from speech_recognition import UnknownValueError

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../app.log',
                    filemode='w')
console_logger = logging.StreamHandler()
console_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_logger.setFormatter(formatter)
logging.getLogger('').addHandler(console_logger)

# Initialize the TTS engine
tts_engine = pyttsx3.init()

def list_voices():
    try:
        voices = tts_engine.getProperty('voices')
        voice_ids = [voice.id for voice in voices]
        logging.info("Voice list retrieved successfully.")
        return voice_ids
    except Exception as e:
        logging.error(f"Error in list_voices: {e}", exc_info=True)

# Global variable to store the last spoken message
last_spoken_message = ""

# Text-to-Speech function
def speak(text):
    global last_spoken_message
    try:
        if text != last_spoken_message:
            last_spoken_message = text
            tts_engine.say(text)
            tts_engine.runAndWait()
            logging.info(f"Spoken text: {text}")
    except Exception as e:
        logging.error(f"Error in speak function: {e}", exc_info=True)

# Configure Speech Recognizer
def configure_recognizer():
    try:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5
        logging.info("Recognizer configured successfully.")
        return recognizer
    except Exception as e:
        logging.error(f"Error in configure_recognizer: {e}", exc_info=True)

def get_speech_input(recognizer, source):
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
    except UnknownValueError:
        logging.error("No speech could be understood.")
        return ""
    except Exception as e:
        logging.error(f"Error in get_speech_input: {e}", exc_info=True)
        return ""

# GPT-Neo model functions
def load_model(model_name):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True)
        logging.info(f"Model {model_name} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error in load_model: {e}", exc_info=True)

def generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p,
                      num_return_sequences, no_repeat_ngram_size):
    try:
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

        response = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        logging.info(f"Response generated successfully.")
        return response
    except Exception as e:
        logging.error(f"Error in generate_response: {e}", exc_info=True)

def update_status_message(message, status_label):
    try:
        status_label.config(text=message)
        status_label.update_idletasks()
        speak(message)
        logging.info(f"Status message updated: {message}")
    except Exception as e:
        logging.error(f"Error in update_status_message: {e}", exc_info=True)

def on_voice_change(voice_var):
    selected_voice = voice_var.get()
    tts_engine.setProperty('voice', selected_voice)
    speak("Hello")

def on_submit(model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command=False):
    try:
        model_name = model_var.get()
        selected_voice = voice_var.get()
        tts_engine.setProperty('voice', selected_voice)

        prompt = prompt_entry.get("1.0", tk.END).strip()
        model, tokenizer = load_model(model_name)

        response = generate_response(model, tokenizer, prompt, 50, 0.7, 50, 0.95, 1, 2)
        response_display.insert(tk.END, f"Prompt: {prompt}\nResponse: {response}\n\n")

        # Clear the prompt entry
        prompt_entry.delete("1.0", tk.END)

        if chatgpt_command:
            webbrowser.open("https://chat.openai.com/")
        else:
            # Say 'Working..' and update the status
            speak("Working..")
            update_status_message("Working...", status_label)
        logging.info("Submit action completed successfully.")
    except Exception as e:
        logging.error(f"Error in on_submit: {e}", exc_info=True)

def voice_control(update_status, window, on_submit_callback, prompt_entry):
    recognizer = configure_recognizer()
    mode = "sleeping"

    with sr.Microphone() as source:
        while True:
            try:
                if mode == "sleeping":
                    update_status("Say 'AI3' and I will listen.")
                    command = get_speech_input(recognizer, source)
                    if "ai3" in command.lower():
                        mode = "listening"
                        update_status("AI3 is now listening and working")
                    elif "chatgpt" in command.lower():
                        on_submit_callback(chatgpt_command=True)
                elif mode == "listening":
                    update_status("Listening Mode: Speak your input")
                    command = get_speech_input(recognizer, source)
                    if "ai3 command" in command.lower():
                        mode = "command"
                        speak("Working")
                        update_status("Working...")
                    elif "ai3 sleep" in command.lower():
                        mode = "sleeping"
                    else:
                        # Append the user's command to the prompt entry without extra characters or blank lines
                        command_text = command.strip()
                        if command_text:
                            prompt_entry.insert(tk.END, command_text + "\n")
                elif mode == "command":
                    update_status("Command Mode: Speak 'Submit', 'Exit',  'Close', 'ChatGPT'")
                    command = get_speech_input(recognizer, source).lower()
                    if "submit" in command:
                        speak("Working")
                        on_submit_callback()
                        mode = "sleeping"
                    elif "exit" in command:
                        speak("Working")
                        mode = "sleeping"
                    elif "close" in command:
                        window.destroy()
                        break
                    elif "chat" in command:
                        speak("Working")
                        on_submit_callback(chatgpt_command=True)
            except Exception as e:
                logging.error(f"Error in voice_control: {e}", exc_info=True)

def main():
    try:
        window = tk.Tk()
        window.title("GPT-Neo Text Generator")

        model_options = ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]
        model_var = StringVar(window)
        model_var.set(model_options[1])

        tk.Label(window, text="Select Model:").pack()
        OptionMenu(window, model_var, *model_options).pack()

        voice_options = list_voices()
        voice_var = StringVar(window)
        voice_var.set(voice_options[1])

        tk.Label(window, text="Select Voice:").pack()
        voice_menu = OptionMenu(window, voice_var, *voice_options, command=on_voice_change)
        voice_menu.pack()

        tk.Label(window, text="Enter your prompt:").pack()
        prompt_entry = scrolledtext.ScrolledText(window, height=5)
        prompt_entry.pack()

        tk.Button(window, text="Generate", command=lambda: on_submit(model_var, voice_var, prompt_entry, response_display, status_label)).pack()

        status_label = tk.Label(window, text="Currently sleeping. Say 'AI3' to wake up or 'ChatGPT' to open the chat.", fg="blue")
        status_label.pack()

        response_display = scrolledtext.ScrolledText(window, height=10)
        response_display.pack()

        listening_thread = threading.Thread(target=lambda: voice_control(lambda msg: update_status_message(msg, status_label), window, lambda chatgpt_command=False: on_submit(model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command), prompt_entry), daemon=True)
        listening_thread.start()

        window.mainloop()
        logging.info("Application started successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    main()
