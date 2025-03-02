import torch
import logging

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
