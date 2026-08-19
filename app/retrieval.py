"""Persistent hybrid retrieval: BM25 + Chroma vectors + CrossEncoder reranking."""

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RetrievalChunk:
    id: str
    document: str
    content: str
    task_id: str
    document_id: str


class HybridRetriever:
    """Keeps a disk-backed BM25 index and a Chroma collection in sync with chunks."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.bm25_file = data_dir / "bm25_index.json"
        self.vector_dir = data_dir / "vector_store"
        self._embedder = None
        self._reranker = None
        self._chroma = None
        self._collection = None
        self._vector_error = None
        self._reranker_error = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        try:
            import jieba

            chinese_words = [word.strip().lower() for word in jieba.lcut(text) if word.strip()]
        except ImportError:
            # Kept only as a safe runtime fallback; install jieba for production quality.
            chinese_words = [text[index : index + 2] for index in range(len(text) - 1) if "\u4e00" <= text[index] <= "\u9fff"]
        ascii_words = __import__("re").findall(r"[a-zA-Z0-9_]+", text.lower())
        return chinese_words + ascii_words

    def _load_index(self) -> dict:
        if not self.bm25_file.exists():
            return {"version": 1, "tasks": {}}
        return json.loads(self.bm25_file.read_text(encoding="utf-8-sig"))

    def _save_index(self, index: dict) -> None:
        self.data_dir.mkdir(exist_ok=True)
        self.bm25_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def _task_index(self, index: dict, task_id: str) -> dict:
        return index.setdefault("tasks", {}).setdefault(task_id, {"chunks": {}, "df": {}, "avgdl": 0.0})

    def _refresh_stats(self, task: dict) -> None:
        chunks = task["chunks"]
        df = Counter()
        total_length = 0
        for item in chunks.values():
            total_length += item["length"]
            df.update(item["tf"].keys())
        task["df"] = dict(df)
        task["avgdl"] = total_length / len(chunks) if chunks else 0.0

    def _init_vector_backend(self) -> bool:
        if self._vector_error:
            return False
        if self._collection is not None:
            return True
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
            self._chroma = chromadb.PersistentClient(path=str(self.vector_dir))
            self._collection = self._chroma.get_or_create_collection(
                name="knowledgepilot_chunks",
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception as exc:  # model download / optional packages must not take down chat
            self._vector_error = str(exc)
            return False

    def backend_status(self) -> str | None:
        return self._vector_error or self._reranker_error

    def has_task(self, task_id: str) -> bool:
        return task_id in self._load_index().get("tasks", {})

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def upsert(self, chunks: list[RetrievalChunk]) -> None:
        if not chunks:
            return
        index = self._load_index()
        touched = set()
        for chunk in chunks:
            task = self._task_index(index, chunk.task_id)
            tokens = self.tokenize(chunk.content)
            task["chunks"][chunk.id] = {
                "document": chunk.document,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "tf": dict(Counter(tokens)),
                "length": max(1, len(tokens)),
            }
            touched.add(chunk.task_id)
        for task_id in touched:
            self._refresh_stats(self._task_index(index, task_id))
        self._save_index(index)

        if self._init_vector_backend():
            self._collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                metadatas=[{"task_id": chunk.task_id, "document": chunk.document, "document_id": chunk.document_id} for chunk in chunks],
                embeddings=self._embed([chunk.content for chunk in chunks]),
            )

    def rebuild_task(self, task_id: str, chunks: list[RetrievalChunk]) -> None:
        index = self._load_index()
        index.setdefault("tasks", {}).pop(task_id, None)
        self._save_index(index)
        if self._init_vector_backend():
            existing = self._collection.get(where={"task_id": task_id})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        self.upsert(chunks)

    def ensure_task(self, task_id: str, chunks: list[RetrievalChunk]) -> None:
        """Perform a one-time backfill for legacy data, never a scan on each query."""
        index = self._load_index()
        indexed = index.get("tasks", {}).get(task_id, {}).get("chunks", {})
        if set(indexed) != {chunk.id for chunk in chunks}:
            self.rebuild_task(task_id, chunks)
        elif self._init_vector_backend() and self._collection.count() == 0 and chunks:
            self._collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                metadatas=[{"task_id": chunk.task_id, "document": chunk.document, "document_id": chunk.document_id} for chunk in chunks],
                embeddings=self._embed([chunk.content for chunk in chunks]),
            )

    @staticmethod
    def _matches_filter(chunk: dict, metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        document_ids = set(metadata_filter.get("document_ids") or [])
        documents = set(metadata_filter.get("documents") or [])
        return (not document_ids or chunk["document_id"] in document_ids) and (not documents or chunk["document"] in documents)

    def _bm25(self, query: str, task_id: str, limit: int, metadata_filter: dict[str, Any] | None = None) -> list[tuple[str, float]]:
        task = self._load_index().get("tasks", {}).get(task_id)
        if not task or not task["chunks"]:
            return []
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        count = len(task["chunks"])
        avgdl = task["avgdl"] or 1.0
        k1, b = 1.5, 0.75
        scores = []
        for chunk_id, chunk in task["chunks"].items():
            if not self._matches_filter(chunk, metadata_filter):
                continue
            score = 0.0
            for token in query_tokens:
                frequency = chunk["tf"].get(token, 0)
                if not frequency:
                    continue
                df = task["df"].get(token, 0)
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * chunk["length"] / avgdl))
            if score:
                scores.append((chunk_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]

    def _vector(self, query: str, task_id: str, limit: int, metadata_filter: dict[str, Any] | None = None) -> list[tuple[str, float]]:
        if not self._init_vector_backend():
            return []
        conditions: list[dict] = [{"task_id": task_id}]
        if metadata_filter and metadata_filter.get("document_ids"):
            conditions.append({"document_id": {"$in": metadata_filter["document_ids"]}})
        if metadata_filter and metadata_filter.get("documents"):
            conditions.append({"document": {"$in": metadata_filter["documents"]}})
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        result = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=limit,
            where=where,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return list(zip(ids, [1 - distance for distance in distances]))

    def _ensure_vectors_for_task(self, task_id: str, task: dict) -> None:
        """Backfill Chroma from the persistent BM25 records, without rereading index.json."""
        if not task["chunks"] or not self._init_vector_backend():
            return
        existing = set(self._collection.get(where={"task_id": task_id}, include=[])["ids"])
        missing = [(chunk_id, item) for chunk_id, item in task["chunks"].items() if chunk_id not in existing]
        if not missing:
            return
        self._collection.upsert(
            ids=[chunk_id for chunk_id, _ in missing],
            documents=[item["content"] for _, item in missing],
            metadatas=[{"task_id": task_id, "document": item["document"], "document_id": item["document_id"]} for _, item in missing],
            embeddings=self._embed([item["content"] for _, item in missing]),
        )

    def _rerank(self, query: str, candidates: list[tuple[str, float]], chunks: dict) -> list[tuple[str, float]]:
        if not candidates:
            return []
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))
            scores = self._reranker.predict([(query, chunks[chunk_id]["content"]) for chunk_id, _ in candidates])
            return sorted(zip((chunk_id for chunk_id, _ in candidates), (float(score) for score in scores)), key=lambda item: item[1], reverse=True)
        except Exception as exc:
            # Hybrid reciprocal-rank fusion remains a meaningful fallback if reranker is unavailable.
            self._reranker_error = f"reranker unavailable: {exc}"
            return candidates

    @staticmethod
    def _dynamic_final_k(ranked: list[tuple[str, float]], maximum: int) -> int:
        """Keep more evidence for an ambiguous score curve and less for a clear winner."""
        minimum = max(1, min(int(os.getenv("RAG_DYNAMIC_MIN_K", "2")), maximum))
        if len(ranked) <= minimum:
            return len(ranked)
        drop_threshold = float(os.getenv("RAG_DYNAMIC_DROP_THRESHOLD", "0.25"))
        for index in range(minimum - 1, min(len(ranked) - 1, maximum - 1)):
            current, following = ranked[index][1], ranked[index + 1][1]
            relative_drop = (current - following) / max(abs(current), 1e-8)
            if relative_drop >= drop_threshold:
                return index + 1
        return min(len(ranked), maximum)

    def search(self, query: str, task_id: str, top_k: int | None = None, metadata_filter: dict[str, Any] | None = None) -> tuple[list[dict], list[str]]:
        task = self._load_index().get("tasks", {}).get(task_id, {"chunks": {}})
        corpus_size = len(task["chunks"])
        candidate_k = min(max(8, int(math.sqrt(corpus_size) * 3)), 24) if corpus_size else 0
        maximum_k = min(max(3, int(math.sqrt(corpus_size))), int(os.getenv("RAG_DYNAMIC_MAX_K", "8")))
        bm25 = self._bm25(query, task_id, candidate_k, metadata_filter)
        self._ensure_vectors_for_task(task_id, task)
        vector = self._vector(query, task_id, candidate_k, metadata_filter)
        fused = Counter()
        for rank, (chunk_id, _) in enumerate(bm25, start=1):
            fused[chunk_id] += 1 / (60 + rank)
        for rank, (chunk_id, _) in enumerate(vector, start=1):
            fused[chunk_id] += 1 / (60 + rank)
        candidates = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:candidate_k]
        ranked = self._rerank(query, candidates, task["chunks"])
        final_k = top_k if top_k is not None else self._dynamic_final_k(ranked, maximum_k)
        trace = [f"BM25 召回 {len(bm25)} 个候选", f"向量召回 {len(vector)} 个候选", f"融合后交给 reranker 精排 {len(candidates)} 个候选"]
        if self.backend_status():
            trace.append(f"向量或 reranker 降级：{self.backend_status()}")
        if metadata_filter:
            trace.append(f"元数据过滤：{metadata_filter}")
        trace.append(f"动态 top-k 返回 {final_k} 个片段（最大 {maximum_k}）" if top_k is None else f"调用方指定 top-k={top_k}")
        return [
            {"chunk_id": chunk_id, "document": task["chunks"][chunk_id]["document"], "content": task["chunks"][chunk_id]["content"], "score": round(score, 4)}
            for chunk_id, score in ranked[:final_k]
        ], trace
