from pathlib import Path

import numpy as np

from src.evaluate import (
    RetrievalExampleResult,
    build_argument_parser,
    load_benchmark,
    load_retrieval_benchmark,
    load_saved_evaluation_run,
    save_evaluation_run,
    score_example,
)


def test_load_retrieval_benchmark_returns_examples():
    examples = load_retrieval_benchmark(Path("data") / "retrieval_eval.json")

    assert len(examples) >= 5
    assert examples[0].question
    assert examples[0].relevant_pages


def test_load_benchmark_exposes_candidate_pages():
    benchmark = load_benchmark(Path("data") / "retrieval_eval.json")

    assert benchmark.candidate_pages is not None
    assert 27 in benchmark.candidate_pages


def test_save_and_load_evaluation_run_round_trips(tmp_path):
    result = RetrievalExampleResult(
        question="What is acne?",
        relevant_pages=(38, 39),
        relevant_chunk_count=3,
        top_pages=(39, 40, 41),
        recall_at_k={1: 0.0, 3: 0.66},
        precision_at_k={1: 0.0, 3: 0.33},
        mrr=0.5,
        ndcg_at_k={1: 0.0, 3: 0.42},
    )

    saved_path = save_evaluation_run(
        results_directory=tmp_path,
        model_name="NeuML/biomedbert-small-embeddings",
        benchmark_path=Path("data") / "retrieval_eval.json",
        benchmark_hash="abc123",
        chunk_count=12,
        results=[result],
        metrics={"mrr": 0.5, "recall@1": 0.0},
    )

    loaded = load_saved_evaluation_run(saved_path)

    assert loaded.model_name == "NeuML/biomedbert-small-embeddings"
    assert loaded.benchmark_hash == "abc123"
    assert loaded.chunk_count == 12
    assert loaded.results[0].question == result.question


def test_score_example_computes_rank_metrics():
    ranked_chunk_indices = [2, 4, 1, 0, 3]
    relevant_chunk_indices = [1, 3]

    recall_at_k, precision_at_k, mrr, ndcg_at_k = score_example(
        ranked_chunk_indices,
        relevant_chunk_indices,
        top_k_values=(1, 3, 5),
    )

    assert np.isclose(recall_at_k[1], 0.0)
    assert np.isclose(precision_at_k[1], 0.0)
    assert np.isclose(mrr, 1 / 3)
    assert ndcg_at_k[5] > 0.0


def test_argument_parser_supports_hybrid_retrieval_options():
    parser = build_argument_parser()

    args = parser.parse_args(
        ["--retrieval-mode", "hybrid", "--dense-top-k", "30", "--bm25-top-k", "25"]
    )

    assert args.retrieval_mode == "hybrid"
    assert args.dense_top_k == 30
    assert args.bm25_top_k == 25
