def write_responses_to_file(prompt, model_name, responses, filename="responses.txt"):
    """
    Write responses and their parameters to a file, including the prompt and model name.
    """
    with open(filename, "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Prompt: {prompt}\n\n")
        for response, params in responses:
            file.write("Parameters:\n")
            file.write(f"Max Length: {params['max_length']}, Temperature: {params['temperature']}, ")
            file.write(f"Top K: {params['top_k']}, Top P: {params['top_p']}, ")
            file.write(f"No Repeat N-Gram Size: {params['no_repeat_ngram_size']}\n")
            file.write("Response:\n")
            file.write(response + "\n\n")
