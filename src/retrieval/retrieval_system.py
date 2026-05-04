import os
import json
from typing import List, Dict
from rank_bm25 import BM25Okapi

LEVELS = ["half", "1", "2", "4", "8"]

class RetrievalSystem:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.level_documents: Dict[str, List[Dict]] = {}
        self.level_bm25: Dict[str, BM25Okapi] = {}
        self.documents = []
        self.bm25 = None
        self._load_and_index_all()

    def _load_jsonl_dir(self, path: str) -> List[Dict]:
        docs = []
        if not os.path.exists(path):
            return docs
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".jsonl"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                docs.append(json.loads(line))
                            except:
                                continue
        return docs

    def _load_and_index_all(self):
        for level in LEVELS:
            level_docs = []
            for source in ["wikipedia", "pubmed", "statpearl"]:
                folder = os.path.join(self.corpus_path, f"{source}_{level}")
                level_docs.extend(self._load_jsonl_dir(folder))
            if level_docs:
                self.level_documents[level] = level_docs
                self.level_bm25[level] = BM25Okapi([d["contents"].lower().split() for d in level_docs])

        if not self.level_documents:
            raise ValueError("No documents loaded from corpus")

        default_level = "1" if "1" in self.level_documents else list(self.level_documents.keys())[0]
        self.documents = self.level_documents[default_level]
        self.bm25 = self.level_bm25[default_level]

    def retrieve(self, query: str, top_k: int = 5, level: str = None) -> List[Dict]:
        bm25 = self.level_bm25.get(level, self.bm25) if level else self.bm25
        documents = self.level_documents.get(level, self.documents) if level else self.documents
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            doc = documents[idx].copy()
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)
        return results

    def retrieve_mog(self, query: str, router_probs: Dict[str, float], top_k: int = 5) -> List[Dict]:
        all_docs: Dict[str, Dict] = {}
        for level, prob in router_probs.items():
            if level not in self.level_bm25:
                continue
            for doc in self.retrieve(query, top_k=top_k, level=level):
                doc_id = doc.get("id", doc.get("contents", "")[:50])
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc.copy()
                    all_docs[doc_id]["weighted_score"] = 0.0
                all_docs[doc_id]["weighted_score"] += prob * doc["bm25_score"]
        return sorted(all_docs.values(), key=lambda d: d["weighted_score"], reverse=True)[:top_k]
