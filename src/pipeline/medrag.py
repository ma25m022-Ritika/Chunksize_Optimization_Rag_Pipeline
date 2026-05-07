from xmlrpc import client

import torch
from src.retrieval.retrieval_system import RetrievalSystem
from src.models.router import Router
from src.config import config
from groq import Groq


class MedRAG:
    def __init__(
        self,
        rag: bool = True,
        pred_with_router: bool = False,
        router_model: Router = None,
        llm_name: str = "llama3",
        top_k: int = 5
    ):
        self.rag = rag
        self.pred_with_router = pred_with_router
        self.router = router_model
        self.llm_name = llm_name
        self.top_k = top_k

        if self.rag:
            self.retriever = RetrievalSystem(corpus_path=config["db_moe_dir"])

    def generate(self, messages):
        print("Generating answer with LLM...")
        client = Groq(api_key=config["groq_api_key"])
        response = client.chat.completions.create(
        model=config["groq_model"],
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        )
        print("LLM generation completed.")
        return response.choices[0].message.content

    def answer(self, question: str, options=None):
        if not self.rag:
            print("RAG is disabled, directly generating answer without retrieval.")
            prompt = f"Question: {question}\nAnswer:"
            messages = [{"role": "user", "content": prompt}]
            return self.generate(messages), [], None

        if self.pred_with_router and self.router is not None:
            print("Running router to get retrieval probabilities...")
            router_probs = self.router.run(question)
            print("Router probabilities:", router_probs)
            retrieved_docs = self.retriever.retrieve_mog(question, router_probs, top_k=self.top_k)
            print(f"Retrieved {len(retrieved_docs)} documents using router probabilities.")

        else:
            print("Running standard retrieval...")
            router_probs = None
            retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)

        context = "\n".join([doc.get("contents", "") for doc in retrieved_docs])

        prompt = f"""You are a medical expert. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Options:
{options}

Instructions:
1. First, write a detailed medical explanation using the context provided.
2. Then, compare your explanation to the options above.
3. Finally, on the LAST LINE, write ONLY: "Answer: X" where X is the best matching option letter (A, B, C, or D).

Format your response exactly like this:
[Your detailed explanation here]

Answer: X"""

        messages = [{"role": "user", "content": prompt}]
        return self.generate(messages), retrieved_docs, router_probs
