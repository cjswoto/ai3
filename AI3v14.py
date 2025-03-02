
import numpy as np
import pyaudio
import tkinter as tk
import pyttsx3
import speech_recognition as sr
import threading
import torch
import webbrowser
import logging
import json
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

    Args:
        text (str): The text to be spoken.

    Global Variables:
        last_spoken_message (str): Tracks the last spoken message to avoid repeating it.

    Action:
        - Compares the provided text with the last spoken message to avoid repetition.
        - Uses the text-to-speech engine (tts_engine) to say the provided text.
        - Runs the text-to-speech engine to actually speak the text.
    """
    global last_spoken_message
    if text != last_spoken_message:
        last_spoken_message = text
        tts_engine.say(text)
        tts_engine.runAndWait()


def update_status_message(message, status_label):
    """
    Updates the status message in the GUI and speaks the message.

    Args:
        message (str): The status message to display and speak.
        status_label (Label): The GUI label for displaying the status message.
    """
    status_label.config(text=message)
    status_label.update_idletasks()
    speak(message)

# Load command logic from JSON file
def load_command_logic(file_path):
    """
    Loads command logic from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing command logic.

    Returns:
        dict: The loaded command logic as a dictionary.
    """
    with open(file_path, "r") as file:
        return json.load(file)


command_logic = load_command_logic("command_logic.json")


def list_voices():
    """
    Lists available voices for text-to-speech.

    Returns:
        list: A list of available voice IDs.
    """
    voices = tts_engine.getProperty('voices')
    return [voice.id for voice in voices]


last_spoken_message = ""


def configure_recognizer():
    """
    Configures and returns a speech recognition recognizer.

    Returns:
        Recognizer: An instance of a configured speech recognition recognizer.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    return recognizer


def get_speech_input(recognizer, source):
    """
    Captures and converts speech input into text.

    Args:
        recognizer (Recognizer): An instance of a configured speech recognition recognizer.
        source: The audio source, typically a microphone.

    Returns:
        str: The recognized speech input as text.
    """
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
    """
    Loads a pre-trained language model and tokenizer.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True)
    return model, tokenizer

def open_url_with_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        # You can add more interactions here

def handle_browsing_mode(recognizer, source, status_label):
    while True:
        command = get_speech_input(recognizer, source).lower()
        if command == "exit":
            update_status_message("Exiting browsing mode.", status_label)
            return "monitor"
        elif command:
            url = f"https://www.{command}"
            open_url_with_playwright(url)
            update_status_message(f"Navigating to {url}", status_label)
def generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p, num_return_sequences, no_repeat_ngram_size):
    """
    Generates a response using a pre-trained language model.

    Args:
        model (PreTrainedModel): The loaded language model.
        tokenizer (PreTrainedTokenizer): The loaded tokenizer.
        prompt (str): The input text prompt.
        max_length (int): The maximum length of the generated response.
        temperature (float): The temperature parameter for sampling.
        top_k (int): The top-k parameter for sampling.
        top_p (float): The top-p parameter for sampling.
        num_return_sequences (int): The number of sequences to generate.
        no_repeat_ngram_size (int): The size of n-grams to avoid repeating.

    Returns:
        str: The generated response text.
    """
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



def on_voice_change(voice_var):
    """
    Changes the voice used for text-to-speech based on the selected voice option.

    Args:
        voice_var (StringVar): The selected voice option.

    Action:
        - Sets the text-to-speech engine's voice to the selected voice.
        - Speaks a confirmation message.
    """
    selected_voice = voice_var.get()
    tts_engine.setProperty('voice', selected_voice)
    speak("Hello, the voice has been changed.")


def list_commands():
    """
    Lists available commands in the current mode.

    Returns:
        str: A string listing available commands.
    """
    current_mode_commands = command_logic.get(mode, {})
    available_commands = "\n".join(current_mode_commands.keys())
    return f"Available commands in {mode} mode: {available_commands}"


def on_submit(model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command=False):
    """
    Handles the submission of the prompt and generates a response.

    Args:
        model_var (StringVar): The selected model option.
        voice_var (StringVar): The selected voice option.
        prompt_entry (ScrolledText): The GUI text entry for the user's prompt.
        response_display (ScrolledText): The GUI text area for displaying responses.
        status_label (Label): The GUI label for displaying status messages.
        chatgpt_command (bool): Indicates whether it's a ChatGPT command.

    Action:
        - Retrieves the selected model and voice options.
        - Loads the model and tokenizer.
        - Generates a response based on the user's prompt.
        - Displays the response in the GUI.
        - Clears the prompt entry.
        - Initiates the "Working..." status message.
    """
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
import logging
def execute_command(command_details, args):
    """
    Executes a command based on command details.

    Args:
        command_details (dict): Details of the command to execute.
        args (dict): Additional arguments for command execution.

    Returns:
        str: The next mode after executing the command.
    """
    action = command_details.get("action")
    logging.info(f"Executing action: {action}")

    if action == "update_status":
        message = command_details.get("message", "")
        update_status_message(message, args["status_label"])
        logging.info(f"Updating status: {message}")
        return command_details.get("next_mode", args["mode"])

    if action == "chat_gpt_command":
        message = "Opening ChatGPT"
        update_status_message(message, args["status_label"])
        speak(message)
        logging.info("ChatGPT command executed")
        args["on_submit_callback"](chatgpt_command=True)
        return command_details.get("next_mode", args["mode"])

    if action == "speak":
        message = command_details.get("message", "")
        speak(message)
        logging.info(f"Speaking: {message}")
        return command_details.get("next_mode", args["mode"])

    if action == "submit":
        message = "Submitting Prompt to LLG"
        update_status_message(message, args["status_label"])
        speak(message)
        logging.info("Submitting prompt")
        args["on_submit_callback"]()
        return command_details.get("next_mode", args["mode"])

    if action == "next_voice":
        current_voice_index = args["voice_options"].index(args["voice_var"].get())
        next_voice_index = (current_voice_index + 1) % len(args["voice_options"])
        args["voice_var"].set(args["voice_options"][next_voice_index])
        on_voice_change(args["voice_var"])
        logging.info(f"Switched to next voice: {args['voice_options'][next_voice_index]}")
        return args["mode"]

    if action == "next_model":
        current_model_index = args["model_options"].index(args["model_var"].get())
        next_model_index = (current_model_index + 1) % len(args["model_options"])
        args["model_var"].set(args["model_options"][next_model_index])
        logging.info(f"Switched to next model: {args['model_options'][next_model_index]}")
        return args["mode"]

    if action == "go_to":
        message = "Entering browsing mode."
        update_status_message(message, args["status_label"])
        speak(message)
        return "browsing"

    if action == "exit":
        message = "Exiting mode."
        update_status_message(message, args["status_label"])
        speak(message)
        logging.info("Exiting mode")
        return command_details.get("next_mode", args["mode"])

    if action == "close":
        message = "Closing Application."
        update_status_message(message, args["status_label"])
        speak(message)
        logging.info("Closing application")
        args["window"].destroy()
        return ""

    logging.warning(f"Unknown action: {action}")
    return args["mode"]


def voice_control(update_status_lambda, window, on_submit_callback, prompt_entry, status_label, logo_label, logo_image, voice_options, model_options, voice_var, model_var, voice_menu):
    """
    Controls voice recognition and command execution.

    Args:
        update_status_lambda (function): Function for updating the status message.
        window (Tk): The main GUI window.
        on_submit_callback (function): Callback function for submitting prompts.
        prompt_entry (ScrolledText): The GUI text entry for the user's prompt.
        status_label (Label): The GUI label for displaying status messages.
        logo_label (Label): The GUI label for displaying the logo.
        logo_image (PIL.Image): The logo image.
        voice_options (list): List of available voice options.
        model_options (list): List of available model options.
        voice_var (StringVar): Variable for the selected voice.
        model_var (StringVar): Variable for the selected model.
        voice_menu (OptionMenu): The GUI option menu for voice selection.

    Global Variables:
        mode (str): Tracks the current mode.

    Action:
        - Listens for voice commands.
        - Recognizes and executes commands based on the current mode.
        - Handles special commands like "list commands" and mode transitions.
        - Appends non-command input to the prompt entry.
    """
    recognizer = configure_recognizer()
    global mode
    mode = "monitor"

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
                            message = f"Command '{cmd}' detected in mode '{mode}'"
                            update_status_lambda(message)
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
                                "voice_menu": voice_menu
                            })
                            logging.debug(f"Switched to mode: {mode}")
                            command_found = True
                            break

                    if not command_found and command == "list commands":
                        response = list_commands()
                        update_status_message(response, status_label)
                        speak(response)

                    if mode == "listening" and not command_found:
                        prompt_entry.insert(tk.END, command + "\n")
                        logging.info(f"Non-command input appended to the prompt entry: {command}")

                    # Handling browsing mode
                    if mode == "browsing":
                        if command == "exit":
                            mode = "monitor"
                        else:
                            url = f"https://www.{command}"
                            response = "browsing to " +url
                            update_status_message(response, status_label)
                            speak(response)
                            open_url_with_playwright(url)
                            mode = "monitor"

            except Exception as e:
                logging.error(f"Error in voice_control: {e}", exc_info=True)
def main():
    window = tk.Tk()
    window.title("ai3 - v12")

    # Load and display the logo image
    logo_image = Image.open("ai3.png")  # Replace with your image file path
    logo_image = logo_image.resize((100, 100))
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(window, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.pack()

    status_label = tk.Label(window, text="Currently sleeping. Say 'AI3' to wake up or 'ChatGPT' to open the chat.", fg="blue")
    status_label.pack()

    # Configure voice and model selection
    voice_options = list_voices()
    voice_var = StringVar(window)
    voice_var.set(voice_options[0])  # Default or previously saved voice

    tk.Label(window, text="Select Voice:").pack()
    voice_menu = OptionMenu(window, voice_var, *voice_options)
    voice_menu.pack()
    voice_var.trace("w", lambda *args: on_voice_change(voice_var))

    model_options = ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]
    model_var = StringVar(window)
    model_var.set(model_options[0])  # Default or previously saved model

    tk.Label(window, text="Select Model:").pack()
    model_menu = OptionMenu(window, model_var, *model_options)
    model_menu.pack()

    tk.Label(window, text="Enter your prompt:").pack()
    prompt_entry = scrolledtext.ScrolledText(window, height=5)
    prompt_entry.pack()

    response_display = scrolledtext.ScrolledText(window, height=10)
    response_display.pack()

    # Log display (optional, depending on your application needs)
    log_display = scrolledtext.ScrolledText(window, height=10)
    log_display.pack()

    # Voice control thread
    listening_thread = threading.Thread(
        target=lambda: voice_control(
            lambda message: update_status_message(message, status_label),
            window,
            lambda chatgpt_command=False: on_submit(
                model_var, voice_var, prompt_entry, response_display, status_label, chatgpt_command
            ),
            prompt_entry,
            status_label,
            logo_label,
            logo_image,
            voice_options,
            model_options,
            voice_var,
            model_var,
            voice_menu
        ),
        daemon=True
    )
    listening_thread.start()

    window.mainloop()

if __name__ == "__main__":
    main()
