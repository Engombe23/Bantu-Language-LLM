# Convert safetensors to pytorch_model.bin

from safetensors.torch import load_file
import torch
from transformers import MarianTokenizer

safetensors_path = "../output/run-300/checkpoint-300/model.safetensors"
state_dict = load_file(safetensors_path)

torch.save(state_dict, "../output/run-300/checkpoint-300/pytorch_model.bin")

# Save tokenizer

checkpoint_path = "../output/run-300/checkpoint-300"
# Load the original tokenizer from the pretrained model
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sw")  
tokenizer.save_pretrained(checkpoint_path)