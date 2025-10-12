from transformers import MarianMTModel, MarianTokenizer
import os

MODEL_PATH = os.path.abspath("../output/run-300/checkpoint-300")
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
model = MarianMTModel.from_pretrained(MODEL_PATH)

def translate_text(text: str):
  inputs = tokenizer(text, return_tensors="pt", padding=True)
  outputs = model.generate(**inputs)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)