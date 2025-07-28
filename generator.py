# generator/generator.py
from transformers import pipeline

def generate_fake_news(prompt):
    generator = pipeline('text-generation', model='gpt2')
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]['generated_text']
