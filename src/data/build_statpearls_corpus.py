import os
import tqdm

chunk_folder = r"corpus/statpearl/chunk"
output_file = r"corpus/statpearl/statpearls_corpus.jsonl"

os.makedirs("corpus/statpearl", exist_ok=True)

files = os.listdir(chunk_folder)

with open(output_file, "w", encoding="utf-8") as outfile:

    for file in tqdm.tqdm(files, desc="Merging StatPearls chunks"):

        path = os.path.join(chunk_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                outfile.write(line)

print("StatPearls corpus created:", output_file)