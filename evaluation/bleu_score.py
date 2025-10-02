import os
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint_path = os.path.abspath("../output/run-300/checkpoint-300")

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
model.eval()

# Swahili & English test pairs
test_data = [
    ("My name is Engombe Wedi Lokanga. I am 22 years old.", "Jina langu ni Engombe Wedi Lokanga. Nina umri wa miaka 22."),
    ("I love programming.", "Ninapenda programu."),
    ("How are you?", "Habari gani?"),
    ("What is your name?", "Jina lako nani?"),
    ("Where do you live?", "Unaishi wapi?"),
]

sw_sentences, reference_translations = zip(*test_data)

# Translate Swahili to English
def translate_sw_to_en(sentences, model, tokenizer, max_length=96):
    inputs = tokenizer(list(sentences), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=max_length)
    translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translations

# Generate translations
predicted_translations = translate_sw_to_en(sw_sentences, model, tokenizer)

bleu = sacrebleu.corpus_bleu(predicted_translations, [reference_translations])
print(f"BLEU score: {bleu.score:.2f}")