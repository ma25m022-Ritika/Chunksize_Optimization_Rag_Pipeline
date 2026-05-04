import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Run evaluation for:
- CoT (LLM only)
- MedRAG (BM25 retrieval)
- MoG (BM25 + Router)

Supports open-source LLMs only.
Produces accuracy results per dataset.
"""

# =========================
# IMPORTS
# =========================
import os
import json
import torch
from tqdm import tqdm

from src.pipeline.medrag import MedRAG
from src.models.router import Router
from src.config import config


# =========================
# CONFIG
# =========================
LLMS = ["internlm", "llama3", "qwen"]  # open-source only
METHODS = ["cot", "medrag", "mog"]
DATASETS = ["medmcqa", "bioasq", "pubmedqa", "medqa", "mmlu"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROUTER_CHECKPOINT = os.path.join(
    config["router_checkpoint_path"],
    "best_model_fold_5.pt"
)


# =========================
# LOAD DATASET
# =========================
def load_dataset(dataset_name):
    dataset_file = config["benchmark_dataset_json"]

    with open(dataset_file, "r") as f:
        data = json.load(f)

    questions, answers, options = [], [], []

    for key in data[dataset_name]:
        item = data[dataset_name][key]

        questions.append(item["question"])
        answers.append(item["answer"])

        if "options" in item:
            options.append(item["options"])
        else:
            options.append(None)

    return questions, answers, options


# =========================
# LOAD ROUTER
# =========================
def load_router():
    print(f"[INFO] Loading router from: {ROUTER_CHECKPOINT}")

    router = Router(input_dim=1024, output_dim=5).to(DEVICE)

    state = torch.load(ROUTER_CHECKPOINT, map_location=DEVICE)
    router.load_state_dict(state)

    router.eval()
    return router


# =========================
# ACCURACY
# =========================
def compute_accuracy(preds, labels):
    correct = 0

    for p, l in zip(preds, labels):
        if str(p).strip().lower() == str(l).strip().lower():
            correct += 1

    return correct / len(preds)


# =========================
# CoT GENERATION
# =========================
def run_cot(medrag, question, options):
    prompt = f"Answer step by step:\n\nQuestion: {question}\n"

    if options:
        for k, v in options.items():
            prompt += f"{k}: {v}\n"

    messages = [{"role": "user", "content": prompt}]
    return medrag.generate(messages)


# =========================
# RUN ONE CONFIG
# =========================
def run_one(llm_name, method, dataset_name, router=None):

    print(f"\n=== Running {llm_name} | {method} | {dataset_name} ===")

    questions, answers, options = load_dataset(dataset_name)

    # Initialize MedRAG
    if method == "cot":
        medrag = MedRAG(
            rag=False,
            llm_name=llm_name
        )

    elif method == "medrag":
        medrag = MedRAG(
            rag=True,
            pred_with_router=False,
            llm_name=llm_name
        )

    elif method == "mog":
        medrag = MedRAG(
            rag=True,
            pred_with_router=True,
            router_model=router,
            llm_name=llm_name
        )

    predictions = []

    for q, opt in tqdm(zip(questions, options), total=len(questions)):
        try:
            if method == "cot":
                ans = run_cot(medrag, q, opt)
            else:
                ans, _, _ = medrag.answer(q, options=opt)

        except Exception as e:
            print(f"[ERROR] {e}")
            ans = ""

        predictions.append(ans)

    acc = compute_accuracy(predictions, answers)

    print(f"[RESULT] {llm_name} | {method} | {dataset_name} → {acc:.4f}")

    return acc


# =========================
# MAIN LOOP
# =========================
def main():

    results = {}

    # Load router once
    router = load_router()

    for llm in LLMS:

        print("\n==============================")
        print(f"Running for LLM: {llm}")
        print("==============================")

        results[llm] = {}

        for method in METHODS:

            results[llm][method] = {}

            for dataset in DATASETS:

                if method == "mog":
                    acc = run_one(llm, method, dataset, router)
                else:
                    acc = run_one(llm, method, dataset)

                results[llm][method][dataset] = acc

    # Save results
    output_path = "final_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()