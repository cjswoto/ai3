import random

def generate_parameter_sets(prompt, num_sets):
    """
    Generate a list of random parameter sets along with the prompt.
    """
    parameter_sets = []
    for _ in range(num_sets):
        params = {
            "prompt": prompt,
            "max_length": random.randint(50, 200),
            "temperature": random.uniform(0.5, 1.0),
            "top_k": random.randint(10, 50),
            "top_p": random.uniform(0.5, 1.0),
            "num_return_sequences": 1,
            "no_repeat_ngram_size": random.randint(2, 5)
        }
        parameter_sets.append(params)
    return parameter_sets
