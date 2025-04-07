import sentencepiece as spm
import glob
import os

os.makedirs("../data/tokenized", exist_ok=True)

def train_model(input_folder, model_prefix, vocab_size=20000):
  text_files = glob.glob(os.path.join(input_folder, "*.txt"))
  corpus = "../data/tokenized/eng_swa_corpus.txt"

  with open(corpus, "w", encoding="utf-8") as outfile:
    for file in text_files:
      with open(file, "r", encoding="utf-8") as infile:
        outfile.write(infile.read() + "\n")
  
  spm.SentencePieceTrainer.train(
    input=corpus, 
    model_prefix=f"../data/tokenized/{model_prefix}", 
    vocab_size=vocab_size, 
    character_coverage=0.9995,
    model_type="unigram"
  )

def tokenize_text(model_prefix):
    sp = spm.SentencePieceProcessor(model_file=f"../data/tokenized/{model_prefix}.model")
    
    input_en = "../data/cleaned/Tanzil.en-sw.en"
    input_sw = "../data/cleaned/Tanzil.en-sw.sw"
    output_en = "../data/tokenized/train-en.sp"
    output_sw = "../data/tokenized/train-sw.sp"

    # Open files and tokenize data
    with open(output_en, "w", encoding="utf-8") as eng_out, open(output_sw, "w", encoding="utf-8") as swa_out:
      with open(input_en, "r", encoding="utf-8") as eng_in, open(input_sw, "r", encoding="utf-8") as swa_in:
        for english_line, swahili_line in zip(eng_in, swa_in):
          eng_out.write(" ".join(sp.encode(english_line.strip(), out_type=str)) + "\n")
          swa_out.write(" ".join(sp.encode(swahili_line.strip(), out_type=str)) + "\n")

train_model("../data/cleaned", "eng_swa")
tokenize_text("eng_swa")
print("Tokenized text saved in 'tokenized/' folder.")