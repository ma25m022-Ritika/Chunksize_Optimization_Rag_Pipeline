import os
import torch
from src.pipeline.medrag import MedRAG
from src.models.router import Router
from src.config import config
from src.evaluation.dataset_loader import locate_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_router = None
_medrag_instances = {}


def get_router():
    global _router
    if _router is None:
        checkpoint = os.path.join(config["router_checkpoint_path"], "best_model_fold_5.pt")
        _router = Router(input_dim=1024, output_dim=5).to(DEVICE)
        state = torch.load(checkpoint, map_location=DEVICE)
        _router.load_state_dict(state)
        _router.eval()
    return _router


def get_medrag(method: str) -> MedRAG:
    global _medrag_instances
    if method not in _medrag_instances:
        router = get_router() if method == "mog" else None
        _medrag_instances[method] = MedRAG(
            rag=(method != "cot"),
            pred_with_router=(method == "mog"),
            router_model=router,
        )
    return _medrag_instances[method]


def answer_question(question: str, options=None, method: str = "mog", top_k: int = 5):
    medrag = get_medrag(method)
    medrag.top_k = top_k

    answer, retrieved_docs, router_probs = medrag.answer(question, options=options)

    # Split LLM explanation and answer letter
    raw = answer.strip()
    if "Answer:" in raw:
        explanation = raw[:raw.rfind("Answer:")].strip()
        answer_line = raw[raw.rfind("Answer:"):].strip()
        prediction = locate_answer(answer_line)
    else:
        explanation = raw
        answer_line = ""
        prediction = locate_answer(raw)

    return {
        "question": question,
        "answer": explanation,                          # full LLM paragraph
        "answer_line": answer_line,                     # "Answer: B"
        "prediction": prediction,                       # "B"
        "method": method,
        "retrieved_docs": [
            {
                "title": d.get("title", ""),
                "content": d.get("contents", "")[:300]  # fixed: contents not content
            }
            for d in retrieved_docs
        ],
        "router_probs": {k: float(v) for k, v in router_probs.items()} if isinstance(router_probs, dict) else None,
    }