import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu
import torch
import os

# Load both translation models
checkpoint_en_to_sw = os.path.abspath("output/run-300/checkpoint-300")
checkpoint_sw_to_en = os.path.abspath("output/checkpoint-1000")

model_en_to_sw = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_en_to_sw)
tokenizer_en_to_sw = AutoTokenizer.from_pretrained(checkpoint_en_to_sw, use_fast=False)
model_en_to_sw.eval()

model_sw_to_en = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_sw_to_en)
tokenizer_sw_to_en = AutoTokenizer.from_pretrained(checkpoint_sw_to_en, use_fast=False)
model_sw_to_en.eval()

# Streamlit UI
st.set_page_config(page_title="Bantu Translator", page_icon="🌍")
st.title("🌍 Swahili ↔ English Translator")
st.caption("Powered by MarianMT")

# Direction toggle
direction = st.radio("Translation Direction", ["English → Swahili", "Swahili → English"])
text_input = st.text_area("Enter your sentence:")

# BLEU Score function
def calculate_bleu(reference, prediction):
    bleu = sacrebleu.corpus_bleu([prediction], [[reference]])
    return bleu.score

# Translation functions
def translate_en_to_sw(text):
    inputs = tokenizer_en_to_sw([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model_en_to_sw.generate(**inputs)
    return tokenizer_en_to_sw.decode(output[0], skip_special_tokens=True)

def translate_sw_to_en(text):
    inputs = tokenizer_sw_to_en([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model_sw_to_en.generate(**inputs)
    return tokenizer_sw_to_en.decode(output[0], skip_special_tokens=True)

# Translation logic
if st.button("Translate"):
    if not text_input.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        if direction == "English → Swahili":
            translated = translate_en_to_sw(text_input)
        else:
            translated = translate_sw_to_en(text_input)

        st.markdown("### 🔁 Translated Text:")
        st.success(translated)

        # BLEU option
        if st.checkbox("Show BLEU Score (reference comparison)"):
            reference_text = st.text_input("Enter reference translation:")
            if reference_text:
                bleu = calculate_bleu(reference_text, translated)
                st.metric("BLEU Score", f"{bleu:.2f}")