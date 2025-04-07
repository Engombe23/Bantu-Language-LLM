import os
import json

os.makedirs("../data/dataset", exist_ok=True)

english_file = "../data/tokenized/train-en.sp"
swahili_file = "../data/tokenized/train-sw.sp"
output_file = "../data/dataset/en-sw-dataset.json"

dataset = []

with open(english_file, "r", encoding="utf-8") as en_file, open(swahili_file, "r", encoding="utf-8") as sw_file:
  for en_line, sw_line in zip(en_file, sw_file):
    en_text = en_line.strip()
    sw_text = sw_line.strip()

    dataset.append({"en": en_text, "sw": sw_text})

with open(output_file, "w", encoding="utf_8") as out_file:
  json.dump(dataset, out_file, indent=4, ensure_ascii=False)