import logging

def on_submit(model_var, prompt_entry, update_status_message, load_model, generate_response):
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
