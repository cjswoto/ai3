import tkinter as tk
from tkinter import scrolledtext, StringVar, OptionMenu
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

def load_model(model_name):
    """
    Load the GPT-Neo model and tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p,
                      num_return_sequences, no_repeat_ngram_size):
    """
    Generate a response from the model based on the given prompt with custom parameter values.
    """
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

    return response

def main():
    # Function to handle the submit action
    def on_submit():
        processing_label.config(text="Generating response...")
        window.update_idletasks()

        selected_model = model_var.get()
        model, tokenizer = load_model(selected_model)
        prompt = prompt_entry.get("1.0", tk.END).strip()

        # Retrieve parameter values
        max_length = int(max_length_entry.get())
        temperature = float(temperature_entry.get())
        top_k = int(top_k_entry.get())
        top_p = float(top_p_entry.get())
        num_return_sequences = int(num_return_sequences_entry.get())
        no_repeat_ngram_size = int(no_repeat_ngram_size_entry.get())

        response = generate_response(model, tokenizer, prompt, max_length, temperature, top_k, top_p, num_return_sequences, no_repeat_ngram_size)

        response_display.configure(state='normal')
        response_display.delete("1.0", tk.END)
        response_display.insert(tk.INSERT, response)
        response_display.configure(state='disabled')

        processing_label.config(text="")

    # Create the main window
    window = tk.Tk()
    window.title("GPT-Neo Text Generator")

    # Model selection dropdown
    model_label = tk.Label(window, text="Select Model:")
    model_label.pack()

    # Model options
    model_options = ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]
    model_var = StringVar(window)
    model_var.set(model_options[1])  # default value

    model_dropdown = OptionMenu(window, model_var, *model_options)
    model_dropdown.pack()

    # Create a text entry for the prompt
    prompt_label = tk.Label(window, text="Enter your prompt:")
    prompt_label.pack()
    prompt_entry = scrolledtext.ScrolledText(window, height=5)
    prompt_entry.pack()

    # Parameter input fields
    max_length_label = tk.Label(window, text="Max Length (Default 150):")
    max_length_label.pack()
    max_length_entry = tk.Entry(window)
    max_length_entry.insert(0, "150")
    max_length_entry.pack()

    temperature_label = tk.Label(window, text="Temperature (Default 0.5):")
    temperature_label.pack()
    temperature_entry = tk.Entry(window)
    temperature_entry.insert(0, "0.5")
    temperature_entry.pack()

    top_k_label = tk.Label(window, text="Top K (Default 20):")
    top_k_label.pack()
    top_k_entry = tk.Entry(window)
    top_k_entry.insert(0, "20")
    top_k_entry.pack()

    top_p_label = tk.Label(window, text="Top P (Default 0.7):")
    top_p_label.pack()
    top_p_entry = tk.Entry(window)
    top_p_entry.insert(0, "0.7")
    top_p_entry.pack()

    num_return_sequences_label = tk.Label(window, text="Number of Return Sequences (Default 3):")
    num_return_sequences_label.pack()
    num_return_sequences_entry = tk.Entry(window)
    num_return_sequences_entry.insert(0, "3")
    num_return_sequences_entry.pack()

    no_repeat_ngram_size_label = tk.Label(window, text="No Repeat N-Gram Size (Default 4):")
    no_repeat_ngram_size_label.pack()
    no_repeat_ngram_size_entry = tk.Entry(window)
    no_repeat_ngram_size_entry.insert(0, "4")
    no_repeat_ngram_size_entry.pack()

    # Create a submit button
    submit_button = tk.Button(window, text="Generate", command=on_submit)
    submit_button.pack()

    # Create a processing label
    processing_label = tk.Label(window, text="")
    processing_label.pack()

    # Create a text display for the response
    response_display = scrolledtext.ScrolledText(window, height=15, state='disabled')
    response_display.pack()

    # Start the GUI event loop
    window.mainloop()

if __name__ == "__main__":
    main()
