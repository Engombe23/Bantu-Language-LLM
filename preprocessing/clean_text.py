import re
import html
import os
from unidecode import unidecode

def clean_text(text):
  text = html.unescape(text)
  text = re.sub(r'<.*?>', '', text)
  text = re.sub(r'[^a-zA-z0-9\s]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  # Normalise text
  return unidecode(text.lower())

def process_folder(input_folder, output_folder):
  os.makedirs(output_folder, exist_ok=True)

  for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    if filename.endswith(".txt") or filename.endswith(".en") or filename.endswith(".sw"):
      with open(input_path, "r", encoding="utf-8") as file, open(output_path, "w", encoding="utf-8") as outfile:
        cleaned_lines = [clean_text(line) for line in file]
        outfile.write("\n".join(cleaned_lines))

  print(f"Processed files saved in {output_folder}")

process_folder("../data/raw", "../data/cleaned")