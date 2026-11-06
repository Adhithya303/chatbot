from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_PATH = PROCESSED_DIR / "corpus.json"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
EMBEDDINGS_META_PATH = PROCESSED_DIR / "embeddings_meta.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "there", "these",
    "they", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "would",
}

TOKEN_PATTERN = re.compile(r"\d+\.\d+|[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize queries and documents consistently, preserving table IDs and acronyms."""
    normalized = (
        text.lower()
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2019", "'")
    )
    raw_tokens = TOKEN_PATTERN.findall(normalized)
    expanded: list[str] = []
    for tok in raw_tokens:
        if tok not in STOPWORDS:
            expanded.append(tok)
        if "-" in tok:
            for part in tok.split("-"):
                if part and part not in STOPWORDS and len(part) > 1:
                    expanded.append(part)
    return expanded if expanded else raw_tokens


def fuse_rankings(*rankings: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, record_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda record_id: (-scores[record_id], record_id))


def compute_rrf_scores(*rankings: list[str], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, record_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """In-memory BM25 + cached FastEmbed dense index with Reciprocal Rank Fusion."""

    def __init__(
        self,
        corpus_path: Path = CORPUS_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.corpus_path = corpus_path
        self.model_name = model_name

        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus file not found at {self.corpus_path}. Run `python -m backend.ingest` first."
            )

        corpus_data = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        self.schema_version: str = str(corpus_data.get("schema_version", "1.0"))
        self.records: list[dict[str, Any]] = corpus_data["records"]
        self.table_metadata: dict[str, dict[str, Any]] = corpus_data["table_metadata"]

        self.record_by_id: dict[str, dict[str, Any]] = {
            r["id"]: r for r in self.records
        }
        self.record_ids: list[str] = [r["id"] for r in self.records]
        self.texts: list[str] = [r["text"] for r in self.records]

        # Build BM25 index in memory
        self.tokenized_docs: list[list[str]] = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Initialize embedding model once and load or build dense embeddings
        self.embedding_model = TextEmbedding(model_name=self.model_name)
        self.embeddings = self._load_or_create_embeddings()

    def _compute_fingerprint(self) -> str:
        payload = json.dumps(
            [(r["id"], r["text"]) for r in self.records],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / norms).astype(np.float32)

    def _load_or_create_embeddings(self) -> np.ndarray:
        fingerprint = self._compute_fingerprint()
        if EMBEDDINGS_PATH.exists() and EMBEDDINGS_META_PATH.exists():
            try:
                meta = json.loads(EMBEDDINGS_META_PATH.read_text(encoding="utf-8"))
                if (
                    meta.get("corpus_fingerprint") == fingerprint
                    and meta.get("model_name") == self.model_name
                    and meta.get("record_ids") == self.record_ids
                ):
                    loaded = np.load(EMBEDDINGS_PATH, allow_pickle=False)
                    if loaded.shape[0] == len(self.records):
                        return self._l2_normalize(loaded)
            except Exception:
                pass

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        raw_vectors = list(self.embedding_model.embed(self.texts))
        matrix = np.asarray(raw_vectors, dtype=np.float32)
        normalized = self._l2_normalize(matrix)

        np.save(EMBEDDINGS_PATH, normalized, allow_pickle=False)
        meta = {
            "corpus_fingerprint": fingerprint,
            "model_name": self.model_name,
            "record_ids": self.record_ids,
        }
        EMBEDDINGS_META_PATH.write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return normalized

    def bm25_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        positive_indices = [
            idx for idx, s in enumerate(scores) if float(s) > 0.0
        ]
        positive_indices.sort(
            key=lambda idx: (-float(scores[idx]), self.record_ids[idx])
        )
        return [
            (self.record_ids[idx], float(scores[idx]))
            for idx in positive_indices[:limit]
        ]

    def vector_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        q_vec = np.asarray(
            list(self.embedding_model.embed([query]))[0], dtype=np.float32
        )
        norm = float(np.linalg.norm(q_vec))
        if norm > 0.0:
            q_vec = q_vec / norm
        # Cosine similarity via dot product of unit-normalized vectors
        sims = self.embeddings @ q_vec
        indices = list(range(len(self.records)))
        indices.sort(key=lambda idx: (-float(sims[idx]), self.record_ids[idx]))
        return [
            (self.record_ids[idx], float(sims[idx]))
            for idx in indices[:limit]
        ]

    def search(
        self,
        query: str,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[dict[str, Any]]:
        bm25_hits = self.bm25_search(query, limit=candidate_k)
        vector_hits = self.vector_search(query, limit=candidate_k)

        bm25_score_map = dict(bm25_hits)
        vector_score_map = dict(vector_hits)

        bm25_ids = [rid for rid, _ in bm25_hits]
        vector_ids = [rid for rid, _ in vector_hits]

        if mode == "vector":
            selected_ids = vector_ids[:top_k]
            rrf_map: dict[str, float] = {}
        elif mode == "bm25":
            selected_ids = bm25_ids[:top_k]
            rrf_map = {}
        else:
            rrf_map = compute_rrf_scores(bm25_ids, vector_ids, k=60)
            fused_ids = fuse_rankings(bm25_ids, vector_ids, k=60)
            selected_ids = fused_ids[:top_k]

        results: list[dict[str, Any]] = []
        for rank, rid in enumerate(selected_ids, start=1):
            rec = dict(self.record_by_id[rid])
            rec["retrieval"] = {
                "mode": mode,
                "rank": rank,
                "rrf_score": round(rrf_map.get(rid, 0.0), 6) if mode == "hybrid" else None,
                "bm25_score": round(bm25_score_map[rid], 4) if rid in bm25_score_map else None,
                "cosine_similarity": round(vector_score_map[rid], 4) if rid in vector_score_map else None,
            }
            results.append(rec)
        return results


_retriever_instance: HybridRetriever | None = None


def get_retriever(force_reload: bool = False) -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None or force_reload:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
