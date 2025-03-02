import tkinter as tk
from tkinter import scrolledtext, StringVar, OptionMenu
import pyttsx3
import speech_recognition as sr
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import threading
import torch
import webbrowser
import logging
import json

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


# Load command logic from JSON file
def load_command_logic(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


command_logic = load_command_logic("../command_logic.json")


def list_voices():
    voices = tts_engine.getProperty('voices')
    return [voice.id for voice in voices]


last_spoken_message = ""


def speak(text):
    global last_spoken_message
    if text != last_spoken_message:
        last_spoken_message = text
        tts_engine.say(text)
        tts_engine.runAndWait()


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


def load_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p, num_return_sequences, no_repeat_ngram_size):
    # Check for empty prompt
    if not prompt.strip():
        logging.error("Empty prompt received.")
        return "No input provided."

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Log the shape of the input tensor
    logging.debug(f"Input tensor shape: {input_ids.shape}")

    try:
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

        return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
    except Exception as e:
        logging.error(f"Error in generate_response: {e}", exc_info=True)
        return "Error generating response."


def update_status_message(message, status_label):
    status_label.config(text=message)
    status_label.update_idletasks()
    speak(message)


def on_voice_change(voice_var):
    selected_voice = voice_var.get()
    tts_engine.setProperty('voice', selected_voice)
    speak("Hello, the voice has been changed.")


def on_submit(model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command=False):
    model_name = model_var.get()
    selected_voice = voice_var.get()
    tts_engine.setProperty('voice', selected_voice)

    prompt = prompt_entry.get("1.0", tk.END).strip()
    model, tokenizer = load_model(model_name)

    response = generate_response(model, tokenizer, prompt, 50, 0.7, 50, 0.95, 1, 2)
    response_display.insert(tk.END, f"Prompt: {prompt}\nResponse: {response}\n\n")

    prompt_entry.delete("1.0", tk.END)

    if chatgpt_command:
        webbrowser.open("https://chat.openai.com/")
    else:
        speak("Working..")
        update_status_message("Working...", status_label)


def execute_command(command_details, args):
    action = command_details.get("action")

    if action == "update_status":
        message = command_details.get("message", "")
        update_status_message(message, args["status_label"])
        return command_details.get("next_mode", args["mode"])

    if action == "chat_gpt_command":
        args["on_submit_callback"](chatgpt_command=True)
        return command_details.get("next_mode", args["mode"])

    if action == "speak":
        message = command_details.get("message", "")
        speak(message)
        return command_details.get("next_mode", args["mode"])

    if action == "submit":
        args["on_submit_callback"]()
        return command_details.get("next_mode", args["mode"])

    if action == "exit":
        return command_details.get("next_mode", args["mode"])

    if action == "close":
        args["window"].destroy()
        return ""

    return args["mode"]


def voice_control(update_status_lambda, window, on_submit_callback, prompt_entry, status_label):
    recognizer = configure_recognizer()
    mode = "sleeping"

    with sr.Microphone() as source:
        logging.info("Microphone is now active.")
        while True:
            try:
                if mode in command_logic:
                    message = f"ai3 in {mode} mode"
                    update_status_lambda(message)
                    logging.debug(f"Listening for commands in mode: {mode}")
                    command = get_speech_input(recognizer, source).lower()
                    logging.debug(f"Received command: {command} in mode:{mode}")

                    command_found = False
                    for cmd, details in command_logic.get(mode, {}).items():
                        if cmd in command:
                            logging.debug(f"Command '{cmd}' detected in mode '{mode}'")
                            mode = execute_command(details, {
                                "mode": mode,
                                "status_label": status_label,
                                "on_submit_callback": on_submit_callback,
                                "window": window
                            })
                            logging.debug(f"Switched to mode: {mode}")
                            command_found = True
                            break

                    # Handling non-command input in listening mode
                    if mode == "listening" and not command_found:
                        prompt_entry.insert(tk.END, command + "\n")
                        logging.info(f"Non-command input appended to the prompt entry: {command}")

            except Exception as e:
                logging.error(f"Error in voice_control: {e}", exc_info=True)


def main():
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
    voice_menu = OptionMenu(window, voice_var, *voice_options)
    voice_menu.pack()
    voice_var.trace("w", lambda *args: on_voice_change(voice_var))

    tk.Label(window, text="Enter your prompt:").pack()
    prompt_entry = scrolledtext.ScrolledText(window, height=5)
    prompt_entry.pack()

    tk.Button(window, text="Generate",
              command=lambda: on_submit(model_var, voice_var, prompt_entry, response_display, status_label)).pack()

    status_label = tk.Label(window, text="Currently sleeping. Say 'AI3' to wake up or 'ChatGPT' to open the chat.",
                            fg="blue")
    status_label.pack()

    response_display = scrolledtext.ScrolledText(window, height=10)
    response_display.pack()

    listening_thread = threading.Thread(
        target=lambda: voice_control(
            lambda message: update_status_message(message, status_label),
            window,
            lambda chatgpt_command=False: on_submit(
                model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command
            ),
            prompt_entry,
            status_label
        ),
        daemon=True
    )
    listening_thread.start()

    window.mainloop()


if __name__ == "__main__":
    main()
