"""Small LLM answer-quality evaluation for the medical RAG benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.evaluate import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_DATA_DIRECTORY,
    RetrievalExample,
    build_corpus_chunks,
    filter_corpus_documents,
    load_benchmark,
    load_corpus_documents,
)
from src.llm_config import DEFAULT_LLM_MODEL_NAME

DEFAULT_LLM_RESULTS_DIRECTORY = Path("data") / "eval_results"
DEFAULT_LLM_MODELS = (
    DEFAULT_LLM_MODEL_NAME,
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
)
DEFAULT_MAX_EXAMPLES = 3
DEFAULT_MAX_CONTEXT_CHARS = 2200
DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_TEMPERATURE = 0.1
DEFAULT_HF_PROVIDERS = (
    None,
    "hf-inference",
    "novita",
    "together",
    "featherless-ai",
    "fireworks-ai",
    "nebius",
    "hyperbolic",
)

_WORD_PATTERN = re.compile(r"[a-z][a-z0-9-]{2,}")
_STOPWORDS = {
    "and",
    "are",
    "can",
    "for",
    "from",
    "how",
    "include",
    "into",
    "that",
    "the",
    "this",
    "used",
    "what",
    "when",
    "with",
}


@dataclass(frozen=True)
class LLMExamplePayload:
    question: str
    reference_answer: str
    context: str
    relevant_pages: tuple[int, ...]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", value).strip("_")


def _keywords(text: str) -> set[str]:
    return {
        word
        for word in _WORD_PATTERN.findall(text.lower())
        if word not in _STOPWORDS and len(word) >= 4
    }


def _keyword_coverage(answer: str, reference_answer: str) -> float:
    reference_keywords = _keywords(reference_answer)
    if not reference_keywords:
        return 0.0
    answer_keywords = _keywords(answer)
    return len(reference_keywords & answer_keywords) / len(reference_keywords)


def _citation_score(answer: str) -> float:
    if re.search(r"\[(?:source|page|[^\]]*page)[^\]]*\]", answer, re.IGNORECASE):
        return 1.0
    if re.search(r"\bpage\s+\d+\b", answer, re.IGNORECASE):
        return 0.7
    return 0.0


def _medical_safety_score(answer: str) -> float:
    normalized = answer.lower()
    signals = [
        "doctor",
        "clinician",
        "medical care",
        "urgent",
        "emergency",
        "red flag",
        "healthcare professional",
        "not a diagnosis",
        "evidence is insufficient",
    ]
    return min(1.0, sum(1 for signal in signals if signal in normalized) / 2)


def _brevity_score(answer: str) -> float:
    word_count = len(answer.split())
    if 45 <= word_count <= 180:
        return 1.0
    if word_count < 20 or word_count > 260:
        return 0.0
    return 0.5


def score_answer(answer: str, reference_answer: str) -> dict[str, float]:
    scores = {
        "keyword_coverage": _keyword_coverage(answer, reference_answer),
        "citation_score": _citation_score(answer),
        "medical_safety_score": _medical_safety_score(answer),
        "brevity_score": _brevity_score(answer),
    }
    scores["overall"] = (
        0.55 * scores["keyword_coverage"]
        + 0.2 * scores["citation_score"]
        + 0.15 * scores["medical_safety_score"]
        + 0.1 * scores["brevity_score"]
    )
    return scores


def build_llm_examples(
    *,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> list[LLMExamplePayload]:
    benchmark = load_benchmark(benchmark_path)
    selected_examples = benchmark.examples[:max_examples]
    source_documents = load_corpus_documents(data_directory)
    examples: list[LLMExamplePayload] = []

    for example in selected_examples:
        context = _build_context_for_example(
            example,
            source_documents=source_documents,
            max_context_chars=max_context_chars,
        )
        examples.append(
            LLMExamplePayload(
                question=example.question,
                reference_answer=example.reference_answer or "",
                context=context,
                relevant_pages=example.relevant_pages,
            )
        )

    return examples


def _build_context_for_example(
    example: RetrievalExample,
    *,
    source_documents: Sequence[Any],
    max_context_chars: int,
) -> str:
    filtered_pages = filter_corpus_documents(source_documents, example.relevant_pages)
    chunks = build_corpus_chunks(filtered_pages)
    context_parts: list[str] = []
    used_chars = 0

    for chunk in chunks:
        metadata = chunk.metadata or {}
        source = metadata.get("source_file") or metadata.get("source", "source")
        page = metadata.get("page_number") or metadata.get("page", "")
        chunk_text = str(chunk.page_content).strip()
        if not chunk_text:
            continue
        formatted = f"[source={source} page={page}] {chunk_text}"
        if used_chars + len(formatted) > max_context_chars:
            break
        context_parts.append(formatted)
        used_chars += len(formatted)

    return "\n\n".join(context_parts)


def build_prompt(example: LLMExamplePayload) -> str:
    return (
        "You are a cautious medical information assistant. Answer only from the "
        "provided context. Do not give a definitive diagnosis. If evidence is "
        "insufficient, say so. Keep the answer concise and include source/page "
        "citations in square brackets.\n\n"
        f"Context:\n{example.context}\n\n"
        f"Question: {example.question}\n\n"
        "Answer:"
    )


def call_huggingface_endpoint(
    *,
    model_name: str,
    prompt: str,
    token: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    from huggingface_hub import InferenceClient

    attempts: list[str] = []
    for provider in DEFAULT_HF_PROVIDERS:
        provider_label = provider or "auto"
        client = InferenceClient(
            model=model_name,
            provider=provider,
            token=token,
            timeout=60,
        )
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            content = str(response.choices[0].message.content or "").strip()
            if content:
                return content
            attempts.append(f"{provider_label}/chat: empty response")
        except Exception as exc:
            attempts.append(f"{provider_label}/chat: {type(exc).__name__}: {exc}")

        try:
            content = str(
                client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_full_text=False,
                )
            ).strip()
            if content:
                return content
            attempts.append(f"{provider_label}/text-generation: empty response")
        except Exception as exc:
            attempts.append(
                f"{provider_label}/text-generation: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError("; ".join(attempts))


def evaluate_llm_models(
    *,
    model_names: Sequence[str] = DEFAULT_LLM_MODELS,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    results_directory: str | Path = DEFAULT_LLM_RESULTS_DIRECTORY,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Path:
    load_dotenv()
    token = os.environ.get("HUGGINGFACEHUB_API_KEY")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_KEY is required for LLM evaluation")

    examples = build_llm_examples(
        benchmark_path=benchmark_path,
        data_directory=data_directory,
        max_examples=max_examples,
        max_context_chars=max_context_chars,
    )
    started_at = datetime.now(UTC).isoformat()
    model_results: list[dict[str, Any]] = []

    for model_name in model_names:
        example_results: list[dict[str, Any]] = []
        for example in examples:
            started = time.perf_counter()
            error = None
            answer = ""
            try:
                answer = call_huggingface_endpoint(
                    model_name=model_name,
                    prompt=build_prompt(example),
                    token=token,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                error = str(exc)
            latency_seconds = time.perf_counter() - started
            scores = score_answer(answer, example.reference_answer) if answer else {}
            example_results.append(
                {
                    "question": example.question,
                    "relevant_pages": list(example.relevant_pages),
                    "reference_answer": example.reference_answer,
                    "answer": answer,
                    "scores": scores,
                    "latency_seconds": latency_seconds,
                    "error": error,
                }
            )

        completed = [
            result
            for result in example_results
            if result["answer"] and not result["error"]
        ]
        summary_scores: dict[str, float] = {}
        if completed:
            metric_names = completed[0]["scores"].keys()
            for metric_name in metric_names:
                summary_scores[metric_name] = sum(
                    result["scores"][metric_name] for result in completed
                ) / len(completed)

        model_results.append(
            {
                "model_name": model_name,
                "completed_count": len(completed),
                "error_count": len(example_results) - len(completed),
                "summary_scores": summary_scores,
                "mean_latency_seconds": (
                    sum(result["latency_seconds"] for result in completed)
                    / len(completed)
                    if completed
                    else None
                ),
                "examples": example_results,
            }
        )

    payload = {
        "created_at": started_at,
        "benchmark_path": str(benchmark_path),
        "data_directory": str(data_directory),
        "max_examples": max_examples,
        "max_context_chars": max_context_chars,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "models": model_results,
    }
    results_path = (
        Path(results_directory)
        / f"llm_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small LLM RAG evaluation.")
    parser.add_argument(
        "--model",
        action="append",
        dest="model_names",
        help="Model repo ID to evaluate. May be supplied multiple times.",
    )
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--data-directory", default=str(DEFAULT_DATA_DIRECTORY))
    parser.add_argument(
        "--results-directory", default=str(DEFAULT_LLM_RESULTS_DIRECTORY)
    )
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    results_path = evaluate_llm_models(
        model_names=tuple(args.model_names or DEFAULT_LLM_MODELS),
        benchmark_path=args.benchmark_path,
        data_directory=args.data_directory,
        results_directory=args.results_directory,
        max_examples=args.max_examples,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Saved: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
