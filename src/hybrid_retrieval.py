"""Hybrid lexical and dense retrieval helpers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

DEFAULT_BM25_INDEX_PATH = Path("data") / "bm25_index.json"
DEFAULT_DENSE_K = 20
DEFAULT_BM25_K = 20
DEFAULT_HYBRID_K = 5
DEFAULT_RRF_K = 60

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")


@dataclass(frozen=True)
class BM25SearchResult:
    document: Document
    score: float
    rank: int


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    source = metadata.get("source_file") or metadata.get("source", "unknown")
    page = metadata.get("page_number") or metadata.get("page", "")
    chunk_number = metadata.get("chunk_number", "")
    return f"{source}|{page}|{chunk_number}|{document.page_content}"


def _to_bm25_document_payload(document: Document) -> dict[str, Any]:
    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata or {}),
    }


def _from_bm25_document_payload(payload: dict[str, Any]) -> Document:
    return Document(
        page_content=str(payload["page_content"]),
        metadata=dict(payload.get("metadata") or {}),
    )


class BM25Index:
    def __init__(
        self,
        *,
        documents: Sequence[Document],
        tokenized_documents: Sequence[Sequence[str]],
        idf: dict[str, float],
        average_document_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        self.tokenized_documents = [list(tokens) for tokens in tokenized_documents]
        self.idf = dict(idf)
        self.average_document_length = average_document_length
        self.k1 = k1
        self.b = b
        self._term_frequencies = [
            Counter(tokens) for tokens in self.tokenized_documents
        ]

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        tokenized_documents = [
            tokenize(document.page_content) for document in documents
        ]
        document_count = len(tokenized_documents)
        document_frequencies: Counter[str] = Counter()
        for tokens in tokenized_documents:
            document_frequencies.update(set(tokens))

        idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }
        average_document_length = (
            sum(len(tokens) for tokens in tokenized_documents) / document_count
            if document_count
            else 0.0
        )
        return cls(
            documents=documents,
            tokenized_documents=tokenized_documents,
            idf=idf,
            average_document_length=average_document_length,
            k1=k1,
            b=b,
        )

    def search(
        self, query: str, *, top_k: int = DEFAULT_BM25_K
    ) -> list[BM25SearchResult]:
        query_terms = tokenize(query)
        scores: list[tuple[int, float]] = []

        for document_index, term_frequencies in enumerate(self._term_frequencies):
            score = self._score_document(
                query_terms,
                term_frequencies,
                document_length=len(self.tokenized_documents[document_index]),
            )
            if score > 0:
                scores.append((document_index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            BM25SearchResult(
                document=self.documents[document_index],
                score=score,
                rank=rank,
            )
            for rank, (document_index, score) in enumerate(scores[:top_k], start=1)
        ]

    def _score_document(
        self,
        query_terms: Sequence[str],
        term_frequencies: Counter[str],
        *,
        document_length: int,
    ) -> float:
        if not query_terms or not document_length or not self.average_document_length:
            return 0.0

        score = 0.0
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / self.average_document_length
            )
            score += idf * (frequency * (self.k1 + 1)) / denominator
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "documents": [
                _to_bm25_document_payload(document) for document in self.documents
            ],
            "tokenized_documents": self.tokenized_documents,
            "idf": self.idf,
            "average_document_length": self.average_document_length,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BM25Index:
        documents = [
            _from_bm25_document_payload(document_payload)
            for document_payload in payload["documents"]
        ]
        return cls(
            documents=documents,
            tokenized_documents=payload["tokenized_documents"],
            idf={str(key): float(value) for key, value in payload["idf"].items()},
            average_document_length=float(payload["average_document_length"]),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )


def save_bm25_index(
    bm25_index: BM25Index, path: str | Path = DEFAULT_BM25_INDEX_PATH
) -> None:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(bm25_index.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_bm25_index(path: str | Path = DEFAULT_BM25_INDEX_PATH) -> BM25Index:
    return BM25Index.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def rrf_merge(
    dense_documents: Sequence[Document],
    bm25_results: Sequence[BM25SearchResult],
    *,
    final_k: int = DEFAULT_HYBRID_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for rank, document in enumerate(dense_documents, start=1):
        key = _document_key(document)
        scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
        documents.setdefault(key, document)

    for result in bm25_results:
        key = _document_key(result.document)
        scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + result.rank)
        documents.setdefault(key, result.document)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [documents[key] for key in ranked_keys[:final_k]]


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_retriever: Any
    bm25_index: BM25Index
    dense_k: int = DEFAULT_DENSE_K
    bm25_k: int = DEFAULT_BM25_K
    final_k: int = DEFAULT_HYBRID_K
    rrf_k: int = DEFAULT_RRF_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any | None = None
    ) -> list[Document]:
        dense_documents = self.dense_retriever.invoke(query)
        bm25_results = self.bm25_index.search(query, top_k=self.bm25_k)
        return rrf_merge(
            dense_documents,
            bm25_results,
            final_k=self.final_k,
            rrf_k=self.rrf_k,
        )
