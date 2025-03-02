from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import logging

def load_model(model_name):
    logging.debug(f"Loading model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True)
    logging.debug("Model loaded successfully")
    return model, tokenizer
