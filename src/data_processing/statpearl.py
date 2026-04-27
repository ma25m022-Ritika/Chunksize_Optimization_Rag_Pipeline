import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

import os
import json
import tqdm

input_folder = os.path.join(BASE_DIR, "corpus", "statpearl", "cleaned_articles")
output_folder = os.path.join(BASE_DIR, "corpus", "statpearl", "chunk")

os.makedirs(output_folder, exist_ok=True)

chunk_size = 900

files = os.listdir(input_folder)

for file in tqdm.tqdm(files):

    path = os.path.join(input_folder, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data["doc_id"]
    text = data["text"]

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    out_file = os.path.join(output_folder, file.replace(".json", ".jsonl"))

    with open(out_file, "w", encoding="utf-8") as f:

        for i, chunk in enumerate(chunks):

            record = {
                "id": f"{title}_{i}",
                "title": title,
                "content": chunk,
                "contents": f"{title}. {chunk}"
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")