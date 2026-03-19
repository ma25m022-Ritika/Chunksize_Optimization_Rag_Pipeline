import os
import tqdm

chunk_folder = r"corpus/pubmed/chunk"
output_file = r"corpus/pubmed/pubmed_corpus.jsonl"

os.makedirs("corpus/pubmed", exist_ok=True)

files = os.listdir(chunk_folder)

with open(output_file, "w", encoding="utf-8") as outfile:

    for file in tqdm.tqdm(files, desc="Merging PubMed chunks"):

        path = os.path.join(chunk_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                outfile.write(line)

print("PubMed corpus created:", output_file)