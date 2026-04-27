from src.prompts.template import get_prompt
import os
import json
from typing import List, Dict
from rank_bm25 import BM25Okapi


class RetrievalSystem:
    """
    Simple BM25-based retrieval system.

    Loads corpus from JSONL files and retrieves top-k documents.
    """

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        self._load_corpus()
        self._build_index()

    def _load_corpus(self):
        """
        Load all JSONL files from corpus directory
        """
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus path not found: {self.corpus_path}")

        for root, _, files in os.walk(self.corpus_path):
            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)

                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                                self.documents.append(obj)
                            except:
                                continue

        if len(self.documents) == 0:
            raise ValueError("No documents loaded from corpus")

    def _build_index(self):
        """
        Build BM25 index
        """
        self.tokenized_corpus = [
            doc["contents"].lower().split() for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents for a query
        """

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)

        return results