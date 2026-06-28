"""Local retrieval evaluation baseline for the medical RAG corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from src.helper import load_pdf_file, text_split
from src.hybrid_retrieval import (
    DEFAULT_BM25_K,
    DEFAULT_DENSE_K,
    BM25Index,
    rrf_merge,
)
from src.indexing import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, apply_chunk_metadata

DEFAULT_BENCHMARK_PATH = Path("data") / "retrieval_eval.json"
DEFAULT_DATA_DIRECTORY = Path("data")
DEFAULT_RESULTS_DIRECTORY = Path("data") / "eval_runs"
DEFAULT_MODEL_NAME = "NeuML/biomedbert-small-embeddings"
DEFAULT_TOP_K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class RetrievalExample:
    question: str
    relevant_pages: tuple[int, ...]
    reference_answer: str | None = None


@dataclass(frozen=True)
class RetrievalBenchmark:
    examples: list[RetrievalExample]
    candidate_pages: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RetrievalExampleResult:
    question: str
    relevant_pages: tuple[int, ...]
    relevant_chunk_count: int
    top_pages: tuple[int, ...]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]


@dataclass(frozen=True)
class PersistedEvaluationRun:
    model_name: str
    benchmark_path: str
    benchmark_hash: str
    chunk_count: int
    example_count: int
    metrics: dict[str, float]
    results: list[RetrievalExampleResult]


def load_retrieval_benchmark(
    path: str | Path = DEFAULT_BENCHMARK_PATH,
) -> list[RetrievalExample]:
    benchmark = load_benchmark(path)
    return benchmark.examples


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_PATH) -> RetrievalBenchmark:
    benchmark_path = Path(path)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))

    raw_examples = payload["examples"] if isinstance(payload, dict) else payload
    examples: list[RetrievalExample] = []

    for raw_example in raw_examples:
        relevant_pages = tuple(int(page) for page in raw_example["relevant_pages"])
        examples.append(
            RetrievalExample(
                question=str(raw_example["question"]),
                relevant_pages=relevant_pages,
                reference_answer=raw_example.get("reference_answer"),
            )
        )

    candidate_pages: tuple[int, ...] | None = None
    if isinstance(payload, dict) and payload.get("candidate_pages") is not None:
        candidate_pages = tuple(int(page) for page in payload["candidate_pages"])

    return RetrievalBenchmark(examples=examples, candidate_pages=candidate_pages)


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def _benchmark_slug(benchmark_path: str | Path) -> str:
    return Path(benchmark_path).stem


def _evaluation_result_path(
    *,
    results_directory: str | Path,
    model_name: str,
    benchmark_path: str | Path,
    retrieval_mode: str = "dense",
) -> Path:
    result_file_name = (
        f"{_model_slug(model_name)}__{retrieval_mode}"
        f"__{_benchmark_slug(benchmark_path)}.json"
    )
    return Path(results_directory) / result_file_name


def _serialize_result(result: RetrievalExampleResult) -> dict[str, Any]:
    return {
        "question": result.question,
        "relevant_pages": list(result.relevant_pages),
        "relevant_chunk_count": result.relevant_chunk_count,
        "top_pages": list(result.top_pages),
        "recall_at_k": {str(key): value for key, value in result.recall_at_k.items()},
        "precision_at_k": {
            str(key): value for key, value in result.precision_at_k.items()
        },
        "mrr": result.mrr,
        "ndcg_at_k": {str(key): value for key, value in result.ndcg_at_k.items()},
    }


def _deserialize_result(payload: dict[str, Any]) -> RetrievalExampleResult:
    return RetrievalExampleResult(
        question=str(payload["question"]),
        relevant_pages=tuple(int(page) for page in payload["relevant_pages"]),
        relevant_chunk_count=int(payload["relevant_chunk_count"]),
        top_pages=tuple(int(page) for page in payload["top_pages"]),
        recall_at_k={
            int(key): float(value) for key, value in payload["recall_at_k"].items()
        },
        precision_at_k={
            int(key): float(value) for key, value in payload["precision_at_k"].items()
        },
        mrr=float(payload["mrr"]),
        ndcg_at_k={
            int(key): float(value) for key, value in payload["ndcg_at_k"].items()
        },
    )


def save_evaluation_run(
    *,
    results_directory: str | Path,
    model_name: str,
    benchmark_path: str | Path,
    benchmark_hash: str,
    chunk_count: int,
    results: Sequence[RetrievalExampleResult],
    metrics: dict[str, float],
    retrieval_mode: str = "dense",
) -> Path:
    results_path = _evaluation_result_path(
        results_directory=results_directory,
        model_name=model_name,
        benchmark_path=benchmark_path,
        retrieval_mode=retrieval_mode,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = PersistedEvaluationRun(
        model_name=model_name,
        benchmark_path=str(benchmark_path),
        benchmark_hash=benchmark_hash,
        chunk_count=chunk_count,
        example_count=len(results),
        metrics=metrics,
        results=list(results),
    )
    serialized_payload = {
        "model_name": payload.model_name,
        "benchmark_path": payload.benchmark_path,
        "benchmark_hash": payload.benchmark_hash,
        "chunk_count": payload.chunk_count,
        "example_count": payload.example_count,
        "metrics": payload.metrics,
        "results": [_serialize_result(result) for result in payload.results],
    }
    results_path.write_text(
        json.dumps(serialized_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return results_path


def load_saved_evaluation_run(path: str | Path) -> PersistedEvaluationRun:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PersistedEvaluationRun(
        model_name=str(payload["model_name"]),
        benchmark_path=str(payload["benchmark_path"]),
        benchmark_hash=str(payload["benchmark_hash"]),
        chunk_count=int(payload["chunk_count"]),
        example_count=int(payload["example_count"]),
        metrics={str(key): float(value) for key, value in payload["metrics"].items()},
        results=[_deserialize_result(result) for result in payload["results"]],
    )


def load_corpus_documents(
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
) -> list[Document]:
    return load_pdf_file(str(data_directory))


def filter_corpus_documents(
    source_documents: Sequence[Document],
    candidate_pages: Sequence[int] | None,
) -> list[Document]:
    if not candidate_pages:
        return list(source_documents)

    candidate_page_set = {int(page) for page in candidate_pages}
    filtered_documents: list[Document] = []
    for document in source_documents:
        page_number = document.metadata.get("page")
        if page_number is None:
            page_number = document.metadata.get("page_number")
        if page_number is None:
            continue
        if int(page_number) in candidate_page_set:
            filtered_documents.append(document)
    return filtered_documents


def build_corpus_chunks(
    source_documents: Sequence[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    chunks = text_split(
        list(source_documents), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    apply_chunk_metadata(chunks)
    return chunks


def _model_family(model_name: str) -> str:
    normalized = model_name.lower()
    if "e5" in normalized:
        return "e5"
    if "bge" in normalized and "m3" not in normalized:
        return "bge"
    return "default"


def _prepare_query_text(model_name: str, question: str) -> str:
    family = _model_family(model_name)
    if family == "e5":
        return f"query: {question}"
    if family == "bge":
        return f"Represent this sentence for searching relevant passages: {question}"
    return question


def _prepare_corpus_text(model_name: str, chunk: Document) -> str:
    family = _model_family(model_name)
    if family == "e5":
        return f"passage: {chunk.page_content}"
    return chunk.page_content


def _build_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _to_numpy(matrix: Any) -> np.ndarray:
    if isinstance(matrix, np.ndarray):
        return matrix
    return np.asarray(matrix)


def _sorted_top_indices(scores: np.ndarray) -> list[int]:
    return list(np.argsort(-scores))


def _dcg(relevance: Sequence[int]) -> float:
    return sum(
        value / math.log2(position + 2) for position, value in enumerate(relevance)
    )


def score_example(
    ranked_chunk_indices: Sequence[int],
    relevant_chunk_indices: Sequence[int],
    *,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
) -> tuple[dict[int, float], dict[int, float], float, dict[int, float]]:
    relevant_set = set(relevant_chunk_indices)
    total_relevant = len(relevant_set)

    recall_at_k: dict[int, float] = {}
    precision_at_k: dict[int, float] = {}
    ndcg_at_k: dict[int, float] = {}

    first_relevant_rank = None

    for rank, chunk_index in enumerate(ranked_chunk_indices, start=1):
        if chunk_index in relevant_set:
            first_relevant_rank = rank
            break

    for top_k in top_k_values:
        top_indices = ranked_chunk_indices[:top_k]
        hits = [1 if chunk_index in relevant_set else 0 for chunk_index in top_indices]
        hit_count = sum(hits)

        recall_at_k[top_k] = hit_count / total_relevant if total_relevant else 0.0
        precision_at_k[top_k] = hit_count / top_k if top_k else 0.0

        ideal_hits = [1] * min(top_k, total_relevant)
        actual_dcg = _dcg(hits)
        ideal_dcg = _dcg(ideal_hits)
        ndcg_at_k[top_k] = actual_dcg / ideal_dcg if ideal_dcg else 0.0

    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    return recall_at_k, precision_at_k, mrr, ndcg_at_k


def evaluate_retrieval_model(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    batch_size: int = 32,
    results_directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
    retrieval_mode: str = "dense",
    dense_top_k: int = DEFAULT_DENSE_K,
    bm25_top_k: int = DEFAULT_BM25_K,
) -> tuple[list[RetrievalExampleResult], dict[str, float], Path | None]:
    retrieval_mode = retrieval_mode.lower()
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError("retrieval_mode must be 'dense' or 'hybrid'")

    benchmark_path = Path(benchmark_path)
    benchmark_hash = _hash_file(benchmark_path)
    results_path = _evaluation_result_path(
        results_directory=results_directory,
        model_name=model_name,
        benchmark_path=benchmark_path,
        retrieval_mode=retrieval_mode,
    )

    if results_path.exists():
        cached_run = load_saved_evaluation_run(results_path)
        if cached_run.benchmark_hash == benchmark_hash:
            return list(cached_run.results), dict(cached_run.metrics), results_path

    benchmark = load_benchmark(benchmark_path)
    examples = benchmark.examples
    source_documents = load_corpus_documents(data_directory)
    source_documents = filter_corpus_documents(
        source_documents, benchmark.candidate_pages
    )
    chunk_documents = build_corpus_chunks(
        source_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    model = _build_embedding_model(model_name)

    corpus_texts = [
        _prepare_corpus_text(model_name, chunk) for chunk in chunk_documents
    ]
    query_texts = [
        _prepare_query_text(model_name, example.question) for example in examples
    ]

    corpus_embeddings = _to_numpy(
        model.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    )
    query_embeddings = _to_numpy(
        model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    )

    ranked_scores = query_embeddings @ corpus_embeddings.T
    bm25_index = (
        BM25Index.from_documents(chunk_documents)
        if retrieval_mode == "hybrid"
        else None
    )
    chunk_index_by_id = {
        id(chunk): index for index, chunk in enumerate(chunk_documents)
    }
    results: list[RetrievalExampleResult] = []

    for example_index, example in enumerate(examples):
        dense_ranked_indices = _sorted_top_indices(ranked_scores[example_index])
        if bm25_index is None:
            ranked_indices = dense_ranked_indices
        else:
            dense_documents = [
                chunk_documents[chunk_index]
                for chunk_index in dense_ranked_indices[:dense_top_k]
            ]
            bm25_results = bm25_index.search(example.question, top_k=bm25_top_k)
            merged_documents = rrf_merge(
                dense_documents,
                bm25_results,
                final_k=max(top_k_values),
            )
            ranked_indices = [
                chunk_index_by_id[id(document)] for document in merged_documents
            ]
        relevant_indices = [
            chunk_index
            for chunk_index, chunk in enumerate(chunk_documents)
            if int(chunk.metadata.get("page_number", -1)) in example.relevant_pages
        ]
        recall_at_k, precision_at_k, mrr, ndcg_at_k = score_example(
            ranked_indices, relevant_indices, top_k_values=top_k_values
        )
        top_pages = tuple(
            int(chunk_documents[chunk_index].metadata.get("page_number", -1))
            for chunk_index in ranked_indices[: max(top_k_values)]
        )
        results.append(
            RetrievalExampleResult(
                question=example.question,
                relevant_pages=example.relevant_pages,
                relevant_chunk_count=len(relevant_indices),
                top_pages=top_pages,
                recall_at_k=recall_at_k,
                precision_at_k=precision_at_k,
                mrr=mrr,
                ndcg_at_k=ndcg_at_k,
            )
        )

    metrics: dict[str, float] = {}
    for top_k in top_k_values:
        recall_values = [result.recall_at_k[top_k] for result in results]
        precision_values = [result.precision_at_k[top_k] for result in results]
        ndcg_values = [result.ndcg_at_k[top_k] for result in results]
        metrics[f"recall@{top_k}"] = float(np.mean(recall_values))
        metrics[f"precision@{top_k}"] = float(np.mean(precision_values))
        metrics[f"ndcg@{top_k}"] = float(np.mean(ndcg_values))
    metrics["mrr"] = float(np.mean([result.mrr for result in results]))
    metrics["example_count"] = float(len(results))
    metrics["chunk_count"] = float(len(chunk_documents))

    saved_path = save_evaluation_run(
        results_directory=results_directory,
        model_name=model_name,
        benchmark_path=benchmark_path,
        retrieval_mode=retrieval_mode,
        benchmark_hash=benchmark_hash,
        chunk_count=len(chunk_documents),
        results=results,
        metrics=metrics,
    )

    return results, metrics, saved_path


def _format_summary(metrics: dict[str, float], *, model_name: str) -> str:
    lines = [f"Model: {model_name}"]
    if "retrieval_mode" in metrics:
        lines.append(f"Retrieval mode: {metrics['retrieval_mode']}")
    lines.append(f"Examples: {int(metrics['example_count'])}")
    lines.append(f"Chunks: {int(metrics['chunk_count'])}")
    lines.append(f"MRR: {metrics['mrr']:.4f}")
    for key in sorted(metrics):
        if (
            key.startswith("recall@")
            or key.startswith("precision@")
            or key.startswith("ndcg@")
        ):
            lines.append(f"{key}: {metrics[key]:.4f}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a retrieval evaluation baseline.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformers model name to evaluate.",
    )
    parser.add_argument(
        "--benchmark-path",
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to the retrieval benchmark JSON file.",
    )
    parser.add_argument(
        "--data-directory",
        default=str(DEFAULT_DATA_DIRECTORY),
        help="Directory containing the source PDF corpus.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for evaluation splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap for evaluation splitting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        dest="top_k_values",
        help="Track metrics at the given cutoff. May be supplied multiple times.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the metrics as JSON instead of a summary.",
    )
    parser.add_argument(
        "--results-directory",
        default=str(DEFAULT_RESULTS_DIRECTORY),
        help="Directory used to persist evaluation runs.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "hybrid"],
        default="dense",
        help="Evaluate dense-only retrieval or local BM25+dense hybrid retrieval.",
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=DEFAULT_DENSE_K,
        help="Dense candidate count used before RRF in hybrid mode.",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=DEFAULT_BM25_K,
        help="BM25 candidate count used before RRF in hybrid mode.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    top_k_values = tuple(sorted(set(args.top_k_values or list(DEFAULT_TOP_K_VALUES))))
    _, metrics, results_path = evaluate_retrieval_model(
        model_name=args.model_name,
        benchmark_path=args.benchmark_path,
        data_directory=args.data_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k_values=top_k_values,
        batch_size=args.batch_size,
        results_directory=args.results_directory,
        retrieval_mode=args.retrieval_mode,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "model_name": args.model_name,
                    "retrieval_mode": args.retrieval_mode,
                    "metrics": metrics,
                    "results_path": str(results_path) if results_path else None,
                },
                indent=2,
            )
        )
    else:
        print(_format_summary(metrics, model_name=args.model_name))
        if results_path:
            print(f"Saved: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
