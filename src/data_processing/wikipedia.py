import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


import os
import tqdm
import json
import regex as re
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()


if __name__ == "__main__":


    MAX_ARTICLES = 20000

    dat = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True
    )

    # Chunking config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Output directory
    output_dir = os.path.join(BASE_DIR, "corpus", "wikipedia", "chunk")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = 2000
    len_just = 5  # fixed padding since length is unknown in streaming

    saved_text = []

    for i, article in enumerate(tqdm.tqdm(dat, total=MAX_ARTICLES)):

        # STOP after required articles
        if i >= MAX_ARTICLES:
            break

        save_id = i // batch_size

        texts = text_splitter.split_text(article['text'].strip())

        curr_text = [
            json.dumps({
                "id": '_'.join([article['id'], str(j)]),
                "title": article['title'],
                "content": re.sub(r"\s+", " ", t),
                "contents": concat(article['title'], re.sub(r"\s+", " ", t))
            })
            for j, t in enumerate(texts)
        ]

        saved_text.extend(curr_text)

        # Save batch
        if (i + 1) % batch_size == 0:
            output_file = os.path.join(
                output_dir,
                f"wiki20220301en{str(save_id).rjust(len_just, '0')}.jsonl"
            )

            with open(output_file, 'w') as f:
                f.write('\n'.join(saved_text))

            saved_text = []

    # Save remaining data
    if len(saved_text) > 0:
        output_file = os.path.join(
            output_dir,
            f"wiki20220301en{str(save_id).rjust(len_just, '0')}.jsonl"
        )

        with open(output_file, 'w') as f:
            f.write('\n'.join(saved_text))
