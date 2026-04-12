import json
import os
import random
from typing import Dict, List

from config import config

LEVELS = ["half", "1", "2", "4", "8"]
DATASETS = ["medmcqa", "bioasq", "pubmedqa", "medqa", "mmlu"]
SIM_OPTION = "roberta"

INPUT_DIR = os.path.join(config["medrag_path"], "retrieval_similarity_per_level")
OUTPUT_DIR = os.path.join(config["medrag_path"], "soft_labels")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_level_results(dataset_name: str, level: str) -> List[Dict]:
    file_name = f"{dataset_name}_merged_{level}_top3_with_similarity.json"
    file_path = os.path.join(INPUT_DIR, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Retrieval result file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {file_path}")

    return data


def transform_similarity_scores(similarity_scores: List[float]) -> List[float]:
    if len(similarity_scores) < 2:
        raise ValueError("Need at least 2 granularities to build soft labels.")

    if all(score == 0 for score in similarity_scores):
        best_idx, second_idx = random.sample(list(range(len(similarity_scores))), 2)
    else:
        best_idx = similarity_scores.index(max(similarity_scores))
        remaining = [(i, s) for i, s in enumerate(similarity_scores) if i != best_idx]
        second_idx = max(remaining, key=lambda item: item[1])[0]

    soft_label = [0.0] * len(similarity_scores)
    soft_label[best_idx] = 0.8
    soft_label[second_idx] = 0.2
    return soft_label


def build_record(level_records: List[Dict], dataset_name: str) -> Dict:
    base_record = level_records[0]
    question = base_record["question"]
    answer_text = base_record["answer_text"]

    for record, level in zip(level_records, LEVELS):
        if record["question"] != question:
            raise ValueError(
                f"Question mismatch for {dataset_name} at level {level}: "
                f"{record['question']!r} != {question!r}"
            )
        if record["answer_text"] != answer_text:
            raise ValueError(
                f"Answer text mismatch for {dataset_name} at level {level} and question {question!r}"
            )

    retrieved_snippets = [[[record.get("top_docs", []) for record in level_records]]]
    scores = [[[[
        float(doc.get("bm25_score", 0.0)) for doc in record.get("top_docs", [])
    ] for record in level_records]]]
    per_level_similarity_scores = []

    for record in level_records:
        sim_list = record.get("similarity_scores", [])

        if len(sim_list) == 0:
            best_score = 0.0
        else:
            best_score = max(sim_list)

        per_level_similarity_scores.append(float(best_score))
    per_doc_similarity_scores = [[[record.get("similarity_scores", []) for record in level_records]]]

    return {
        "dataset_name": dataset_name,
        "question": question,
        "answer_text": answer_text,
        "levels": LEVELS,
        "retrieved_snippets": retrieved_snippets,
        "scores": scores,
        "per_level_similarity_scores": per_level_similarity_scores,
        "per_doc_similarity_scores": per_doc_similarity_scores,
        "soft_label": transform_similarity_scores(per_level_similarity_scores),
    }


def build_soft_labels_for_dataset(dataset_name: str) -> None:
    print(f"[INFO] Building soft labels for {dataset_name}")

    level_to_records = {level: load_level_results(dataset_name, level) for level in LEVELS}
    expected_count = len(level_to_records[LEVELS[0]])

    for level in LEVELS[1:]:
        current_count = len(level_to_records[level])
        if current_count != expected_count:
            raise ValueError(
                f"Record count mismatch for {dataset_name}: "
                f"{LEVELS[0]} has {expected_count}, {level} has {current_count}"
            )

    combined_output = []
    for idx in range(expected_count):
        level_records = [level_to_records[level][idx] for level in LEVELS]
        combined_output.append(build_record(level_records, dataset_name))

    output_path = os.path.join(
        OUTPUT_DIR,
        f"cache_soft_labels_{SIM_OPTION}_{dataset_name}_merged_5_top.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_output, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved soft labels to: {output_path}")


if __name__ == "__main__":
    for dataset_name in DATASETS:
        build_soft_labels_for_dataset(dataset_name)
