import os
import json
import subprocess

from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import config

# =========================
# Fixed settings
# =========================
DB_MOE_DIR = config["db_moe_dir"]
CACHE_DIR  = config["cache_dir"]
TOP_K      = 3

CORPUS_NAMES = [
    "statpearl_half",
    "statpearl_1",
    "statpearl_2",
    "statpearl_4",
    "statpearl_8",
]

QUESTION    = "Which of the following is not true for myelinated nerve fibers:"
ANSWER_TEXT = "Impulse through myelinated fibers is slower than non-myelinated fibers"


# =========================
# BM25 index helper
# =========================
def ensure_bm25_index(corpus_name: str) -> str:
    chunk_dir = os.path.join(DB_MOE_DIR, corpus_name, "chunk")
    index_dir = os.path.join(DB_MOE_DIR, corpus_name, "index", "bm25")

    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    if not os.path.isdir(index_dir):
        print(f"[INFO] Building BM25 index for {corpus_name} ...")
        os.makedirs(os.path.dirname(index_dir), exist_ok=True)
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", chunk_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "8",
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ]
        subprocess.run(cmd, check=True)

    return index_dir


# =========================
# Similarity scorer
# =========================
class SimilarityScorer:
    def __init__(self):
        print("[INFO] Loading RoBERTa model ...")
        self.encoder = SentenceTransformer("stsb-roberta-large", cache_folder=CACHE_DIR)

    def compute(self, text_a: str, text_b: str) -> float:
        emb_a = self.encoder.encode(text_a).reshape(1, -1)
        emb_b = self.encoder.encode(text_b).reshape(1, -1)
        return float(cosine_similarity(emb_a, emb_b)[0][0])


# =========================
# Main
# =========================
def main():
    scorer = SimilarityScorer()

    print(f"\nQuestion   : {QUESTION}")
    print(f"Answer text: {ANSWER_TEXT}")
    print("=" * 80)

    similarity_scores = []

    for corpus_name in CORPUS_NAMES:
        print(f"\n[Corpus] {corpus_name}")
        index_dir = ensure_bm25_index(corpus_name)
        searcher  = LuceneSearcher(index_dir)
        hits      = searcher.search(QUESTION, k=TOP_K)

        if len(hits) == 0:
            print("  No hits returned by BM25.")
            similarity_scores.append(0.0)
            continue

        print(f"  BM25 returned {len(hits)} chunks:")

        chunk_similarities = []
        for rank, h in enumerate(hits):
            raw_doc  = searcher.doc(h.docid).raw()
            doc      = json.loads(raw_doc)
            contents = doc.get("contents", doc.get("content", ""))
            sim      = scorer.compute(contents, ANSWER_TEXT)
            chunk_similarities.append(sim)

            print(f"  Chunk {rank+1} | BM25 score: {h.score:.4f} | Cosine sim: {sim:.4f}")
            print(f"           | Text preview: {contents[:120].strip()} ...")

        # Current code behaviour: only top-1 chunk similarity is used
        top1_sim = chunk_similarities[0]
        avg_sim  = sum(chunk_similarities) / len(chunk_similarities)
        max_sim  = max(chunk_similarities)

        print(f"\n  >> Top-1 similarity (current code): {top1_sim:.4f}")
        print(f"  >> Avg similarity  (alternative)  : {avg_sim:.4f}")
        print(f"  >> Max similarity  (alternative)  : {max_sim:.4f}")

        similarity_scores.append(top1_sim)

    # Soft label logic (same as original)
    print("\n" + "=" * 80)
    print("Similarity scores (top-1 per level):", [f"{s:.4f}" for s in similarity_scores])

    best_idx   = similarity_scores.index(max(similarity_scores))
    remaining  = [(i, s) for i, s in enumerate(similarity_scores) if i != best_idx]
    second_idx = max(remaining, key=lambda x: x[1])[0]

    soft_label = [0.0] * len(similarity_scores)
    soft_label[best_idx]   = 0.8
    soft_label[second_idx] = 0.2

    print(f"Best corpus   : {CORPUS_NAMES[best_idx]}  (index {best_idx})")
    print(f"Second corpus : {CORPUS_NAMES[second_idx]}  (index {second_idx})")
    print(f"Soft label    : {soft_label}")

    # =========================
    # Save in same format as original soft label script
    # =========================
    OUTPUT_DIR = os.path.join(config["medrag_path"], "soft_labels")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Re-run retrieval to collect full retrieved_snippets and scores in original nesting format
    all_level_snippets = []
    all_level_scores   = []
    for corpus_name in CORPUS_NAMES:
        index_dir = ensure_bm25_index(corpus_name)
        searcher  = LuceneSearcher(index_dir)
        hits      = searcher.search(QUESTION, k=TOP_K)

        level_snippets = []
        level_scores   = []
        if len(hits) == 0:
            level_snippets.append("NO_TEXT_RETRIEVED")
            level_scores.append(0.0)
        else:
            for h in hits:
                raw_doc  = searcher.doc(h.docid).raw()
                doc      = json.loads(raw_doc)
                level_snippets.append({
                    "id":       h.docid,
                    "title":    doc.get("title", ""),
                    "contents": doc.get("contents", doc.get("content", "")),
                })
                level_scores.append(h.score)

        all_level_snippets.append(level_snippets)
        all_level_scores.append(level_scores)

    # Recompute per-chunk similarity scores for all levels (for saving)
    all_level_chunk_sims = []
    for level_snippets in all_level_snippets:
        chunk_sims = []
        for snippet in level_snippets:
            if snippet == "NO_TEXT_RETRIEVED":
                chunk_sims.append(0.0)
            else:
                sim = scorer.compute(snippet["contents"], ANSWER_TEXT)
                chunk_sims.append(sim)
        all_level_chunk_sims.append(chunk_sims)

    # top-1, avg, max per level — mirrors what was printed earlier
    similarity_summary = []
    for i, chunk_sims in enumerate(all_level_chunk_sims):
        similarity_summary.append({
            "corpus":    CORPUS_NAMES[i],
            "top1":      chunk_sims[0],
            "avg":       sum(chunk_sims) / len(chunk_sims),
            "max":       max(chunk_sims),
            "per_chunk": chunk_sims,
        })

    output_record = [{
        "dataset_name":       "medmcqa",
        "question":           QUESTION,
        "answer_text":        ANSWER_TEXT,
        "retrieved_snippets": [[all_level_snippets]],    # same [[...]] nesting as original
        "scores":             [[all_level_scores]],
        "similarity_scores":  [[all_level_chunk_sims]],  # per-chunk cosine sim, same nesting       # human-readable top1/avg/max per level
        "soft_label":         soft_label,
    }]
    print( "similarity_summary: ",similarity_summary, )
    output_path = os.path.join(OUTPUT_DIR, "cache_soft_labels_roberta_statpearl_debug_1q.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_record, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved to: {output_path}")


if __name__ == "__main__":
    main()