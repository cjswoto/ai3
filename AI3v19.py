import tkinter as tk
import pyttsx3
import speech_recognition as sr
import threading
import torch
import webbrowser
import json
import logging
from tkinter import scrolledtext, StringVar, OptionMenu
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from PIL import Image, ImageTk
from playwright.sync_api import sync_playwright

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w')
console_logger = logging.StreamHandler()
console_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_logger.setFormatter(formatter)
logging.getLogger('').addHandler(console_logger)

# Initialize the TTS engine
tts_engine = pyttsx3.init()

def speak(text):
    """
    Text-to-speech function to convert text into spoken words.
    """
    global last_spoken_message
    if text != last_spoken_message:
        last_spoken_message = text
        tts_engine.say(text)
        tts_engine.runAndWait()

def update_status_message(message, status_label, log_display):
    """
    Updates the status message in the GUI, speaks the message, and writes to the log display.
    """
    status_label.config(text=message)
    status_label.update_idletasks()
    speak(message)
    if log_display is not None:
        log_display.insert(tk.END, f"Status Update: {message}\n")

# Load command logic from JSON file
def load_command_logic(file_path):
    """
    Loads command logic from a JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)

command_logic = load_command_logic("command_logic.json")

def list_voices():
    """
    Lists available voices for text-to-speech.
    """
    voices = tts_engine.getProperty('voices')
    return [voice.id for voice in voices]

last_spoken_message = ""

def configure_recognizer():
    """
    Configures and returns a speech recognition recognizer.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    return recognizer

def get_speech_input(recognizer, source):
    """
    Captures and converts speech input into text.
    Modified to use the already opened microphone source.
    """
    try:
        audio = recognizer.listen(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        logging.error("No speech could be understood.")
        return ""
    except Exception as e:
        logging.error(f"Error in get_speech_input: {e}", exc_info=True)
        return ""

def load_model(model_name):
    """
    Loads a pre-trained language model and tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True)
    return model, tokenizer

def open_url_with_playwright(url):
    with sync_playwright() as p:
        logging.info("Opening Browser")
        browser = p.chromium.launch(headless=False)
        logging.info("Opening page")
        page = browser.new_page()
        logging.info("goto")
        page.goto(url)
        # Additional interactions can be added here
        browser.close()

def handle_browsing_mode(recognizer, source, status_label, log_display):
    """
    Handles browsing mode by listening for voice commands and navigating to URLs.
    """
    while True:
        command = get_speech_input(recognizer, source).lower()
        if command == "exit":
            update_status_message("Exiting browsing mode.", status_label, log_display)
            return "monitor"
        elif command:
            url = f"https://www.{command}"
            webbrowser.open(url)
            update_status_message(f"Navigating to {url}", status_label, log_display)

def generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p, num_return_sequences, no_repeat_ngram_size):
    """
    Generates a response using a pre-trained language model.
    """
    if not prompt.strip():
        logging.error("Empty prompt received.")
        return "No input provided."

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
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

def on_voice_change(voice_var):
    """
    Changes the voice used for text-to-speech based on the selected voice option.
    """
    selected_voice = voice_var.get()
    tts_engine.setProperty('voice', selected_voice)
    speak("Hello, the voice has been changed.")

def list_commands():
    """
    Lists available commands in the current mode.
    """
    current_mode_commands = command_logic.get(mode, {})
    available_commands = "\n".join(current_mode_commands.keys())
    return f"Available commands in {mode} mode: {available_commands}"

def on_submit(model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command, log_display):
    global mode
    model_name = model_var.get()
    selected_voice = voice_var.get()
    tts_engine.setProperty('voice', selected_voice)
    prompt = prompt_entry.get("1.0", tk.END).strip()
    if mode == "browsing":
        url = f"https://www.{prompt}"
        response = f"Opening {url}"
        update_status_message(response, status_label, log_display)
        speak(response)
        open_url_with_playwright(url)
        mode = "monitor"
    else:
        model, tokenizer = load_model(model_name)
        response = generate_response(model, tokenizer, prompt, 50, 0.7, 50, 0.95, 1, 2)
        response_display.insert(tk.END, f"Prompt: {prompt}\nResponse: {response}\n\n")
    prompt_entry.delete("1.0", tk.END)
    if chatgpt_command:
        webbrowser.open("https://chat.openai.com/")
    else:
        speak("Working...")
        update_status_message("Working...", status_label, log_display)

def execute_command(command_details, args):
    action = command_details.get("action")
    logging.info(f"Executing action: {action}")
    args['log_display'].insert(tk.END, f"Executing Command: {action}\n")
    if action == "update_status":
        message = command_details.get("message", "")
        update_status_message(message, args["status_label"], args['log_display'])
        logging.info(f"Updating status: {message}")
        return command_details.get("next_mode", args["mode"])
    if action == "chat_gpt_command":
        message = "Opening ChatGPT"
        update_status_message(message, args["status_label"], args['log_display'])
        speak(message)
        logging.info("ChatGPT command executed")
        webbrowser.open("https://chat.openai.com/")
        return command_details.get("next_mode", args["mode"])
    if action == "speak":
        message = command_details.get("message", "")
        speak(message)
        update_status_message(f"Speaking: {message}", args["status_label"], args['log_display'])
        logging.info(f"Speaking: {message}")
        return command_details.get("next_mode", args["mode"])
    if action == "submit":
        message = "Submitting Prompt to LLG"
        update_status_message(message, args["status_label"], args['log_display'])
        speak(message)
        logging.info("Submitting prompt")
        args["on_submit_callback"]()
        return command_details.get("next_mode", args["mode"])
    if action == "next_voice":
        current_voice_index = args["voice_options"].index(args["voice_var"].get())
        next_voice_index = (current_voice_index + 1) % len(args["voice_options"])
        args["voice_var"].set(args["voice_options"][next_voice_index])
        on_voice_change(args["voice_var"])
        update_status_message(f"Switched to next voice: {args['voice_options'][next_voice_index]}", args["status_label"], args['log_display'])
        logging.info(f"Switched to next voice: {args['voice_options'][next_voice_index]}")
        return args["mode"]
    if action == "next_model":
        current_model_index = args["model_options"].index(args["model_var"].get())
        next_model_index = (current_model_index + 1) % len(args["model_options"])
        args["model_var"].set(args["model_options"][next_model_index])
        update_status_message(f"Switched to next model: {args['model_options'][next_model_index]}", args["status_label"], args['log_display'])
        logging.info(f"Switched to next model: {args['model_options'][next_model_index]}")
        return args["mode"]
    if action == "go_to":
        message = "Entering browsing mode/goto."
        update_status_message(message, args["status_label"], args['log_display'])
        speak(message)
        logging.info("Entered 'go_to' action")
        prompt_value = args.get("prompt_entry", "").get("1.0", "end-1c")
        logging.info(f"Retrieved prompt value: {prompt_value}")
        speak("Checking Prompt")
        speak(prompt_value)
        words = prompt_value.split()
        logging.info(f"Split prompt into words: {words}")
        if words:
            first_word = words[0]
            logging.info(f"First word in input: {first_word}")
            url = f"https://www.{first_word}"
            logging.info(f"Constructed URL: {url}")
            print(url)
        else:
            speak("No words found in the input.")
            logging.warning("No words found in the input.")
        response = f"browsing to {url}"
        update_status_message(response, args["status_label"], args['log_display'])
        logging.info(f"Updated status with response: {response}")
        webbrowser.open(url)
        logging.info("Opened URL with Playwright")
        return command_details.get("next_mode", args["mode"])
    if action == "exit":
        message = "Exiting mode."
        update_status_message(message, args["status_label"], args['log_display'])
        speak(message)
        logging.info("Exiting mode")
        return command_details.get("next_mode", args["mode"])
    if action == "close":
        message = "Closing Application."
        update_status_message(message, args["status_label"], args['log_display'])
        speak(message)
        logging.info("Closing application")
        args["window"].destroy()
        return ""
    logging.warning(f"Unknown action: {action}")
    return args["mode"]

def voice_control(update_status_lambda, window, on_submit_callback, prompt_entry, status_label, logo_label, logo_image, voice_options, model_options, voice_var, model_var, voice_menu, log_display):
    recognizer = configure_recognizer()
    global mode
    mode = "monitor"
    with sr.Microphone() as source:
        logging.info("Microphone is now active.")
        while True:
            try:
                if mode in command_logic:
                    message = f"ai3 in {mode} mode"
                    update_status_lambda(message, log_display)
                    logging.debug(f"Listening for commands in mode: {mode}")
                    command = get_speech_input(recognizer, source).lower()
                    log_display.insert(tk.END, f"User Command: {command}\n")
                    log_display.yview(tk.END)
                    command_found = False
                    for cmd, details in command_logic.get(mode, {}).items():
                        if cmd in command:
                            message = f"Command '{cmd}' detected in mode '{mode}'"
                            update_status_lambda(message, log_display)
                            logging.debug(message)
                            mode = execute_command(details, {
                                "mode": mode,
                                "status_label": status_label,
                                "on_submit_callback": on_submit_callback,
                                "window": window,
                                "voice_options": voice_options,
                                "model_options": model_options,
                                "voice_var": voice_var,
                                "model_var": model_var,
                                "voice_menu": voice_menu,
                                "log_display": log_display,
                                "prompt_entry": prompt_entry
                            })
                            command_found = True
                            break
                    if not command_found and command == "list commands":
                        response = list_commands()
                        update_status_lambda(response, log_display)
                    if mode == "listening" and not command_found:
                        prompt_entry.insert(tk.END, command + "\n")
                        logging.info(f"Non-command input appended to the prompt entry: {command}")
                    if mode == "browsing":
                        if command == "exit":
                            mode = "monitor"
                        else:
                            response = "Browsing mode activated"
                            handle_browsing_mode(recognizer, source, status_label, log_display)
                            update_status_lambda(response, log_display)
            except Exception as e:
                logging.error(f"Error in voice_control: {e}", exc_info=True)

def main():
    window = tk.Tk()
    window.title("ai3 - v12")
    background_image = Image.open('background.png')
    resized_image = background_image.resize((800, 600))
    bg_image = ImageTk.PhotoImage(resized_image)
    background_label = tk.Label(window, image=bg_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    logo_image = Image.open("ai3.png")
    logo_image = logo_image.resize((100, 100))
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(window, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.pack()
    logo_label.place(x=0, y=0)
    status_label = tk.Label(window, text="Currently sleeping. Say 'AI3' to wake up or 'ChatGPT' to open the chat.", fg="blue", anchor="w")
    status_label.pack()
    voice_options = list_voices()
    voice_var = StringVar(window)
    voice_var.set(voice_options[0])
    tk.Label(window, text="Select Voice:").pack()
    voice_menu = OptionMenu(window, voice_var, *voice_options)
    voice_menu.config(width=15)
    voice_menu.pack()
    voice_var.trace("w", lambda *args: on_voice_change(voice_var))
    model_options = ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]
    model_var = StringVar(window)
    model_var.set(model_options[0])
    tk.Label(window, text="Select Model:").pack()
    model_menu = OptionMenu(window, model_var, *model_options)
    model_menu.pack()
    tk.Label(window, text="Enter your prompt:").pack()
    prompt_entry = scrolledtext.ScrolledText(window, height=5)
    prompt_entry.pack()
    response_display = scrolledtext.ScrolledText(window, height=10)
    response_display.pack()
    log_display = scrolledtext.ScrolledText(window, height=10)
    log_display.pack()
    background_label.lower()
    listening_thread = threading.Thread(
        target=lambda: voice_control(
            lambda message, log_display: update_status_message(message, status_label, log_display),
            window,
            lambda chatgpt_command=False: on_submit(model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command, log_display),
            prompt_entry,
            status_label,
            logo_label,
            logo_image,
            voice_options,
            model_options,
            voice_var,
            model_var,
            voice_menu,
            log_display
        ),
        daemon=True
    )
    listening_thread.start()
    window.mainloop()

if __name__ == "__main__":
    main()
