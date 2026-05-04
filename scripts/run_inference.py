import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import argparse
from tqdm import tqdm

from src.pipeline.medrag import MedRAG
from src.models.router import Router
from src.evaluation.dataset_loader import QADataset, locate_answer
from src.config import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROUTER_CHECKPOINT = os.path.join(
    config["router_checkpoint_path"],
    "best_model_fold_5.pt"
)


def load_router():
    router = Router(input_dim=1024, output_dim=5).to(DEVICE)
    state = torch.load(ROUTER_CHECKPOINT, map_location=DEVICE)
    router.load_state_dict(state)
    router.eval()
    return router


def run_inference(dataset_name: str, method: str, llm_name: str, output_path: str):
    print(f"Loading dataset: {dataset_name}...")
    dataset = QADataset(dataset_name)
    print("dataset loaded:", dataset_name, len(dataset))

    router = None
    if method == "mog":
        print("Loading router model...")
        router = load_router()
        print("router model loaded.")

    medrag = MedRAG(
        rag=(method != "cot"),
        pred_with_router=(method == "mog"),
        router_model=router,
        llm_name=llm_name
    )
    print("MedRAG pipeline initialized.")

    results = []
    correct = 0
    print("started loop")

    for i in tqdm(range(len(dataset)), desc=f"{method} | {dataset_name}"):       # change to range(len(dataset)) for full run
        item = dataset[i]
        question = item["question"]
        answer_gt = item["answer"]
        options = item.get("options", None)
        print("question:", question)

        try:
            ans, retrieved_docs, router_probs = medrag.answer(question, options=options)

            # Split explanation and answer letter
            raw = ans.strip()
            if "Answer:" in raw:
                explanation = raw[:raw.rfind("Answer:")].strip()
                answer_line = raw[raw.rfind("Answer:"):].strip()
                pred = locate_answer(answer_line)
            else:
                explanation = raw
                answer_line = ""
                pred = locate_answer(raw)

        except Exception as e:
            print(f"[ERROR] idx={i}: {e}")
            explanation = ""
            answer_line = ""
            pred = "No"
            retrieved_docs = []
            router_probs = None

        is_correct = pred.strip().upper() == str(answer_gt).strip().upper()
        if is_correct:
            correct += 1

        results.append({
            "idx": i,
            "question": question,
            "ground_truth": answer_gt,
            "prediction": pred,
            "llm_explanation": explanation,        # full paragraph from LLM
            "llm_answer_line": answer_line,        # "Answer: B"
            "correct": is_correct,
            "router_probs": {k: float(v) for k, v in router_probs.items()} if isinstance(router_probs, dict) else None,
        })

    accuracy = correct / len(dataset)
    print(f"\n[RESULT] {method} | {dataset_name} | {llm_name} → Accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

    print(f"[SAVED] {output_path}")
    return accuracy

if __name__ == "__main__":
    print("started inference...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["medqa", "medmcqa", "bioasq", "pubmedqa", "mmlu"])
    parser.add_argument("--method", type=str, required=True, choices=["cot", "medrag", "mog"])
    parser.add_argument("--llm", type=str, default="llama3")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    print(f"Arguments: {args}")
    output_path = args.output or os.path.join(
        config["prediction_folder"],
        f"{args.method}_{args.dataset}_{args.llm}.json"
    )
    print("runing inference")
    run_inference(args.dataset, args.method, args.llm, output_path)