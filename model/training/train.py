import yaml
import torch
import os
import psutil
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset_loader import BantuTranslationDataset

# Function to monitor memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  

# Print initial memory usage
print(f"Initial memory usage: {memory_usage()} MB")


# Load config from YAML
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load model and tokenizer
model_name = config["model"]["name"]
tokenizer_name = model_name

# ✅ Use MarianTokenizer for MarianMT
tokenizer = MarianTokenizer.from_pretrained(tokenizer_name, use_fast=False, padding=False)
model = MarianMTModel.from_pretrained(model_name)

# Load dataset
dataset_loader = BantuTranslationDataset(tokenizer_path=tokenizer_name, 
    data_path=config["dataset"]["train_path"],
    src_lang=config["dataset"]["source_lang"],
    tgt_lang=config["dataset"]["target_lang"]
)
train_dataset, val_dataset, test_dataset = dataset_loader.get_tokenized_datasets()

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=config["output"]["save_dir"],
    logging_dir=config["output"]["log_dir"],
    per_device_train_batch_size=config["training"]["batch_size"],
    per_device_eval_batch_size=config["training"]["batch_size"],
    learning_rate=float(config["training"]["learning_rate"]),
    warmup_steps=config["training"]["warmup_steps"],
    weight_decay=config["training"]["weight_decay"],
    num_train_epochs=config["training"]["max_epochs"],
    save_steps=config["training"]["save_steps"],
    logging_steps=config["training"]["logging_steps"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    fp16=False,  # No fp16 for CPU
    save_total_limit=2,
    predict_with_generate=True  # Needed for seq2seq models like MarianMT
)

# Ensure the training happens on CPU
device = torch.device("cpu")
model.to(device)

# Use gradient checkpointing to save memory if needed
from torch.utils.checkpoint import checkpoint

def checkpointed_model(*inputs):
    return checkpoint(model.forward, *inputs)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    #train_dataset = train_dataset.select(range(min(200, len(train_dataset)))),
    train_dataset = train_dataset.select(range(min(1000, len(train_dataset)))),
    eval_dataset=val_dataset
)

# Start training and monitor memory usage
print(f"Starting training...")

for epoch in range(training_args.num_train_epochs):
    print(f"Epoch {epoch+1}/{training_args.num_train_epochs}...")
    
    # Monitor memory usage before each epoch
    print(f"Memory usage before epoch: {memory_usage()} MB")
    
    trainer.train()
    
    # Monitor memory usage after each epoch
    print(f"Memory usage after epoch: {memory_usage()} MB")

# Monitor memory usage at the end of training
print(f"Final memory usage: {memory_usage()} MB")