import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import config


DB_MOE_DIR = Path(BASE_DIR) / config["db_moe_dir"]
CACHE_DIR = config["cache_dir"]
MEDRAG_PATH = Path(config["medrag_path"])

DATASET_PATHS = {
    "medmcqa": str(Path(BASE_DIR) / config["medmcqa_path"]),
    "bioasq": str(Path(BASE_DIR) / config["bioasq_path"]),
    "pubmedqa": str(Path(BASE_DIR) / config["pubmedqa_path"]),
    "medqa": str(Path(BASE_DIR) / config["medqa_path"]),
    "mmlu": str(Path(BASE_DIR) / config["mmlu_path"]),
}

TOP_K = 3
SOURCE_PREFIXES = ["statpearl", "wikipedia", "pubmed"]

OUTPUT_DIR = MEDRAG_PATH / "retrieval_similarity_per_level"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_eval_questions(dataset_name: str) -> set:
    eval_data_path = MEDRAG_PATH / "qa_datasets_rawdata" / "eval_data.json"
    with open(eval_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {data[dataset_name][k]["question"] for k in data[dataset_name].keys()}


def load_training_questions_and_answers(dataset_name: str, dataset_path: str) -> Tuple[List[str], List[str]]:
    questions, answer_texts = [], []
    eval_questions = get_eval_questions(dataset_name)

    if dataset_name == "medmcqa":
        file_path = os.path.join(dataset_path, "data", "dev.json")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["question"] in eval_questions:
                    continue
                questions.append(item["question"])
                explanation = item["exp"] if item["exp"] is not None else ""
                if item["cop"] == 1:
                    answer_texts.append(explanation + item["opa"])
                elif item["cop"] == 2:
                    answer_texts.append(explanation + item["opb"])
                elif item["cop"] == 3:
                    answer_texts.append(explanation + item["opc"])
                elif item["cop"] == 4:
                    answer_texts.append(explanation + item["opd"])

    elif dataset_name == "bioasq":
        file_path = os.path.join(dataset_path, "all_dataset.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for q in data["questions"]:
            if q["body"] in eval_questions:
                continue
            questions.append(q["body"])
            answer_texts.append(" ".join(snippet["text"] for snippet in q["snippets"]))

    elif dataset_name == "pubmedqa":
        file_path = os.path.join(dataset_path, "data", "test_set.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for _, q in data.items():
            if q["QUESTION"] in eval_questions:
                continue
            questions.append(q["QUESTION"])
            answer_texts.append(".".join(q["CONTEXTS"]))

    elif dataset_name == "medqa":
        file_path = os.path.join(
            dataset_path, "data_clean", "questions", "US", "4_options", "phrases_no_exclude_test.jsonl"
        )
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["question"] in eval_questions:
                    continue
                questions.append(item["question"])
                answer_texts.append("Q:" + item["question"] + "; A:" + item["answer"])

    elif dataset_name == "mmlu":
        file_path = os.path.join(dataset_path, "data", "dev.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for _, q in data.items():
            if q["question"] in eval_questions:
                continue
            questions.append(q["question"])
            options = q["options"]
            answer_idx = q["answer"]
            answer_texts.append("Q:" + q["question"] + "; A:" + options[answer_idx])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return questions, answer_texts


def read_jsonl(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_one_level(level: str) -> Path:
    merged_chunk_dir = DB_MOE_DIR / f"merged_{level}" / "chunk"
    merged_chunk_dir.mkdir(parents=True, exist_ok=True)
    merged_file = merged_chunk_dir / "corpus.jsonl"

    if merged_file.exists():
        print(f"[INFO] Merged file already exists: {merged_file}")
        return merged_chunk_dir

    seen_ids = set()
    total_written = 0

    with open(merged_file, "w", encoding="utf-8") as out_f:
        for source in SOURCE_PREFIXES:
            src_dir = DB_MOE_DIR / f"{source}_{level}" / "chunk"

            if not src_dir.exists():
                raise FileNotFoundError(f"Missing source chunk dir: {src_dir}")

            files = sorted(src_dir.glob("*.jsonl"))
            print(f"[INFO] Reading {len(files)} files from {src_dir}")

            for file in files:
                for rec in read_jsonl(file):
                    new_id = f"{source}_{level}_{rec['id']}"
                    if new_id in seen_ids:
                        continue
                    seen_ids.add(new_id)

                    rec["id"] = new_id
                    rec["source"] = source
                    rec["granularity"] = level

                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_written += 1

    print(f"[DONE] merged_{level}: wrote {total_written} docs to {merged_file}")
    return merged_chunk_dir


def ensure_bm25_index_for_level(level: str) -> Path:
    merged_chunk_dir = DB_MOE_DIR / f"merged_{level}" / "chunk"
    index_dir = DB_MOE_DIR / f"merged_{level}" / "index" / "bm25"

    if not merged_chunk_dir.exists():
        merge_one_level(level)

    if not index_dir.exists():
        print(f"[INFO] Building BM25 index for merged_{level}")
        index_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(merged_chunk_dir),
            "--index", str(index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "8",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        subprocess.run(cmd, check=True)

    return index_dir


def get_searcher_for_level(level: str) -> LuceneSearcher:
    index_dir = ensure_bm25_index_for_level(level)
    return LuceneSearcher(str(index_dir))


class SimilarityScorer:
    def __init__(self, model_name: str = "stsb-roberta-large"):
        self.encoder = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
        self.answer_cache = {}

    def encode_text(self, text: str):
        return self.encoder.encode(text, convert_to_numpy=True)

    def get_answer_embedding(self, answer_text: str):
        if answer_text not in self.answer_cache:
            self.answer_cache[answer_text] = self.encode_text(answer_text)
        return self.answer_cache[answer_text]

    def compute_similarity(self, snippet_text: str, answer_text: str) -> float:
        snippet_emb = self.encode_text(snippet_text).reshape(1, -1)
        answer_emb = self.get_answer_embedding(answer_text).reshape(1, -1)
        return float(cosine_similarity(snippet_emb, answer_emb)[0][0])


def retrieve_top_k_with_similarity(
    question: str,
    answer_text: str,
    searcher: LuceneSearcher,
    scorer: SimilarityScorer,
    top_k: int = TOP_K,
) -> Dict:
    hits = searcher.search(question, k=top_k)

    top_docs = []
    similarity_scores = []

    if len(hits) == 0:
        return {
            "question": question,
            "answer_text": answer_text,
            "top_docs": [],
            "similarity_scores": [],
            "avg_similarity_score": 0.0,
        }

    for h in hits:
        raw_doc = searcher.doc(h.docid).raw()
        doc = json.loads(raw_doc)

        contents = doc.get("contents", doc.get("content", ""))
        sim_score = scorer.compute_similarity(contents, answer_text) if contents else 0.0

        top_docs.append({
            "id": h.docid,
            "title": doc.get("title", ""),
            "contents": contents,
            "source": doc.get("source", ""),
            "granularity": doc.get("granularity", ""),
            "bm25_score": float(h.score),
        })

        similarity_scores.append(float(sim_score))

    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    return {
        "question": question,
        "answer_text": answer_text,
        "top_docs": top_docs,
        "similarity_scores": similarity_scores,
        "avg_similarity_score": float(avg_similarity_score),
    }


def run_for_one_level(level: str):
    print(f"\n========== Running level: {level} ==========")

    merge_one_level(level)
    searcher = get_searcher_for_level(level)
    scorer = SimilarityScorer(model_name="stsb-roberta-large")

    for dataset_name, dataset_path in DATASET_PATHS.items():
        print(f"\n[INFO] Dataset: {dataset_name}")
        questions, answer_texts = load_training_questions_and_answers(dataset_name, dataset_path)

        output = []

        for question, answer_text in tqdm(
            zip(questions, answer_texts),
            total=len(questions),
            desc=f"{dataset_name} | merged_{level}"
        ):
            item = retrieve_top_k_with_similarity(
                question=question,
                answer_text=answer_text,
                searcher=searcher,
                scorer=scorer,
                top_k=TOP_K,
            )

            item["dataset_name"] = dataset_name
            item["level"] = level
            item["corpus_name"] = f"merged_{level}"

            output.append(item)

        out_file = OUTPUT_DIR / f"{dataset_name}_merged_{level}_top{TOP_K}_with_similarity.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[DONE] Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["half", "1", "2", "4", "8"],
        help="Granularity level to run"
    )
    args = parser.parse_args()

    run_for_one_level(args.level)