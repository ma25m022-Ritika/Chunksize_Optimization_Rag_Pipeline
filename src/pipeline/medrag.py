import torch
import numpy as np

from src.retrieval.retrieval_system import RetrievalSystem
from src.models.router import Router


class MedRAG:
    def __init__(
        self,
        rag: bool = True,
        pred_with_router: bool = False,
        router_model: Router = None,
        llm_name: str = "llama3",
        top_k: int = 3
    ):
        self.rag = rag
        self.pred_with_router = pred_with_router
        self.router = router_model
        self.llm_name = llm_name
        self.top_k = top_k

        if self.rag:
            self.retriever = RetrievalSystem()

    # =========================
    # Dummy LLM (replace later)
    # =========================
    def generate(self, messages):
        # Simple placeholder for now
        return "Generated answer (LLM output)"

    # =========================
    # Router scoring
    # =========================
    def route_query(self, query: str):
        """
        Converts query → embedding → router prediction
        Currently using random vector (you must replace later)
        """

        # TODO: replace with SentenceTransformer
        embedding = torch.randn(1024)

        with torch.no_grad():
            logits = self.router(embedding.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return probs  # shape: [5]

    # =========================
    # Core QA function
    # =========================
    def answer(self, question: str, options=None):
        """
        Returns:
            answer_text
            retrieved_docs
            router_probs
        """

        # ----------------------------------
        # Case 1: CoT (no retrieval)
        # ----------------------------------
        if not self.rag:
            prompt = f"Question: {question}\nAnswer:"
            messages = [{"role": "user", "content": prompt}]
            return self.generate(messages), [], None

        # ----------------------------------
        # Case 2: Retrieval
        # ----------------------------------
        retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)

        # ----------------------------------
        # Case 3: Router (MoG)
        # ----------------------------------
        router_probs = None

        if self.pred_with_router and self.router is not None:
            router_probs = self.route_query(question)

            # NOTE:
            # Currently NOT modifying retrieval based on router
            # This is where real MoG logic should be applied

        # ----------------------------------
        # Build context
        # ----------------------------------
        context = "\n".join([
            doc.get("contents", "") for doc in retrieved_docs
        ])

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        messages = [{"role": "user", "content": prompt}]

        answer = self.generate(messages)

        return answer, retrieved_docs, router_probs
