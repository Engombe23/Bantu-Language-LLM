import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

checkpoint_en_to_sw = os.path.abspath("../output/run-300/checkpoint-300")
checkpoint_sw_to_en = os.path.abspath("../output/checkpoint-1000")

model_en_to_sw = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_en_to_sw)
tokenizer_en_to_sw = AutoTokenizer.from_pretrained(checkpoint_en_to_sw, use_fast=False)

model_sw_to_en = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_sw_to_en)
tokenizer_sw_to_en = AutoTokenizer.from_pretrained(checkpoint_sw_to_en, use_fast=False)

def translate_en_to_sw(text):
    inputs = tokenizer_en_to_sw(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model_en_to_sw.generate(**inputs)
    return tokenizer_en_to_sw.batch_decode(translated, skip_special_tokens=True)[0]

def translate_sw_to_en(text):
    inputs = tokenizer_sw_to_en(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model_sw_to_en.generate(**inputs)
    return tokenizer_sw_to_en.batch_decode(translated, skip_special_tokens=True)[0]

# Example use
english_text = "My name is Engombe Wedi Lokanga. I am 22 years old."
swahili_text = "Jina langu ni Engombe Wedi Lokanga. Nina umri wa miaka ishirini na miwili."

swahili_translation = translate_en_to_sw(english_text)
english_translation = translate_sw_to_en(swahili_text)

print("EN → SW:")
print(f"English: {english_text}")
print(f"Swahili: {swahili_translation}\n")

print("SW → EN:")
print(f"Swahili: {swahili_text}")
print(f"English: {english_translation}")