import os
import json
import random
import subprocess
from typing import List, Dict, Tuple

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import config


# =========================
# Fixed settings
# =========================
DB_MOE_DIR = config["db_moe_dir"]
CACHE_DIR = config["cache_dir"]

DATASET_PATHS = {
    "medmcqa": config["medmcqa_path"],
    "bioasq": config["bioasq_path"],
    "pubmedqa": config["pubmedqa_path"],
    "medqa": config["medqa_path"],
    "mmlu": config["mmlu_path"],
}

# Your current folder naming
CORPUS_GROUPS = {
    "statpearl": ["statpearl_half", "statpearl_1", "statpearl_2", "statpearl_4", "statpearl_8"],
    "pubmed": ["pubmed_half", "pubmed_1", "pubmed_2", "pubmed_4", "pubmed_8"],
    "wikipedia": ["wikipedia_half", "wikipedia_1", "wikipedia_2", "wikipedia_4", "wikipedia_8"],
}

TOP_K = 3
SIM_OPTION = "roberta"
OUTPUT_DIR = os.path.join(config["medrag_path"], "soft_labels")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Dataset loading
# =========================
def get_eval_questions(dataset_name: str) -> set:
    eval_data_path = os.path.join(config["medrag_path"], "qa_datasets_rawdata", "eval_data.json")
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


# =========================
# BM25 helpers
# =========================
def ensure_bm25_index(corpus_name: str) -> str:
    chunk_dir = os.path.join(DB_MOE_DIR, corpus_name, "chunk")
    index_dir = os.path.join(DB_MOE_DIR, corpus_name, "index", "bm25")

    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    if not os.path.isdir(index_dir):
        print(f"[INFO] Building BM25 index for {corpus_name}")
        os.makedirs(os.path.dirname(index_dir), exist_ok=True)
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", chunk_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "8",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        subprocess.run(cmd, check=True)

    return index_dir


def get_searcher(corpus_name: str) -> LuceneSearcher:
    index_dir = ensure_bm25_index(corpus_name)
    return LuceneSearcher(index_dir)


# =========================
# Similarity model
# =========================
class SimilarityScorer:
    def __init__(self, sim_option: str = "roberta"):
        self.sim_option = sim_option.lower()
        self.encoder = SentenceTransformer("stsb-roberta-large", cache_folder=CACHE_DIR)

    def compute(self, snippet_text: str, answer_text: str) -> float:
        snippet_embedding = self.encoder.encode(snippet_text).reshape(1, -1)
        answer_embedding = self.encoder.encode(answer_text).reshape(1, -1)
        return float(cosine_similarity(snippet_embedding, answer_embedding)[0][0])


def transform_similarity_scores(similarity_scores: List[float]) -> List[float]:
    if len(similarity_scores) < 2:
        raise ValueError("Need at least 2 granularities to build soft labels.")

    if all(score == 0 for score in similarity_scores):
        chosen = random.sample(list(range(len(similarity_scores))), 2)
        best_idx, second_idx = chosen[0], chosen[1]
    else:
        best_idx = similarity_scores.index(max(similarity_scores))
        remaining = [(i, s) for i, s in enumerate(similarity_scores) if i != best_idx]
        second_idx = max(remaining, key=lambda x: x[1])[0]

    soft_label = [0.0] * len(similarity_scores)
    soft_label[best_idx] = 0.8
    soft_label[second_idx] = 0.2
    return soft_label


# =========================
# Retrieval across 5 levels
# =========================
def retrieve_for_all_levels(question: str, searchers: List[LuceneSearcher]) -> Tuple[List, List]:
    all_level_snippets = []
    all_level_scores = []

    for searcher in searchers:
        hits = searcher.search(question, k=TOP_K)

        level_snippets = []
        level_scores = []

        if len(hits) == 0:
            level_snippets.append("NO_TEXT_RETRIEVED")
            level_scores.append(0.0)
        else:
            for h in hits:
                raw_doc = searcher.doc(h.docid).raw()
                doc = json.loads(raw_doc)
                level_snippets.append({
                    "id": h.docid,
                    "title": doc.get("title", ""),
                    "contents": doc.get("contents", doc.get("content", "")),
                })
                level_scores.append(h.score)

        all_level_snippets.append(level_snippets)
        all_level_scores.append(level_scores)

    # Keep same nesting style expected by your moe.py:
    # snippets[0][0] -> list over granularities
    retrieved_snippets = [[all_level_snippets]]
    scores = [[all_level_scores]]

    return retrieved_snippets, scores


# =========================
# Build soft labels end-to-end
# =========================
def build_soft_labels_for_group(group_name: str, corpus_names: List[str], scorer: SimilarityScorer):
    print(f"\n[INFO] Processing corpus group: {group_name}")
    searchers = [get_searcher(corpus_name) for corpus_name in corpus_names]

    combined_output = []

    for dataset_name, dataset_path in DATASET_PATHS.items():
        print(f"[INFO] Loading dataset: {dataset_name}")
        questions, answer_texts = load_training_questions_and_answers(dataset_name, dataset_path)

        for question, answer_text in tqdm(
            list(zip(questions, answer_texts)),
            total=len(questions),
            desc=f"{group_name} | {dataset_name}"
        ):
            retrieved_snippets, scores = retrieve_for_all_levels(question, searchers)

            level_snippets = retrieved_snippets[0][0]
            top_snippets = []
            for level_result in level_snippets:
                if level_result == "NO_TEXT_RETRIEVED":
                    top_snippets.append("NO_TEXT_RETRIEVED")
                else:
                    top_snippets.append(level_result[0])

            similarity_scores = [0.0] * len(top_snippets)
            for i, top_item in enumerate(top_snippets):
                if top_item != "NO_TEXT_RETRIEVED":
                    snippet_text = top_item["contents"]
                    similarity_scores[i] = scorer.compute(snippet_text, answer_text)

            soft_label = transform_similarity_scores(similarity_scores)

            combined_output.append({
                "dataset_name": dataset_name,
                "question": question,
                "answer_text": answer_text,
                "retrieved_snippets": retrieved_snippets,
                "scores": scores,
                "soft_label": soft_label,
            })

    output_path = os.path.join(OUTPUT_DIR, f"cache_soft_labels_{SIM_OPTION}_{group_name}_5.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_output, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved soft labels to:\n{output_path}")


if __name__ == "__main__":
    scorer = SimilarityScorer(sim_option=SIM_OPTION)

    for group_name, corpus_names in CORPUS_GROUPS.items():
        build_soft_labels_for_group(group_name, corpus_names, scorer)