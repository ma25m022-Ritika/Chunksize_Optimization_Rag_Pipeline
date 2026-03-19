import os
import tqdm
import json
import regex as re
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()


if __name__ == "__main__":

    # Load Wikipedia dataset from HuggingFace
    dat = load_dataset(
        "wikipedia",
        "20220301.en",
        cache_dir="./corpus/wikipedia",
        trust_remote_code=True
    )

    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Output directory
    chunk_dir = "corpus/wikipedia/chunk"

    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)

    batch_size = 10000
    len_just = len(str(len(dat['train']) // batch_size + 1))

    saved_text = []

    for i in tqdm.tqdm(range(len(dat['train'])), desc="Processing Wikipedia articles"):

        save_id = i // batch_size

        out_path = f"{chunk_dir}/wiki20220301en{str(save_id).rjust(len_just,'0')}.jsonl"

        if os.path.exists(out_path):
            continue

        article = dat['train'][i]

        title = article['title']
        article_id = article['id']
        text = article['text'].strip()

        # Split article into chunks
        texts = text_splitter.split_text(text)

        curr_text = []

        for j, t in enumerate(texts):

            t = re.sub("\s+", " ", t)

            record = {
                "id": f"{article_id}_{j}",
                "title": title,
                "content": t,
                "contents": concat(title, t)
            }

            curr_text.append(json.dumps(record))

        saved_text.extend(curr_text)

        if (i + 1) % batch_size == 0:

            with open(out_path, 'w', encoding="utf-8") as f:
                f.write('\n'.join(saved_text))

            saved_text = []

    if len(saved_text) > 0:

        with open(out_path, 'w', encoding="utf-8") as f:
            f.write('\n'.join(saved_text))

    print("Wikipedia chunk generation complete.")