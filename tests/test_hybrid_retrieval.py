from langchain_core.documents import Document

from src.hybrid_retrieval import (
    BM25Index,
    HybridRetriever,
    load_bm25_index,
    rrf_merge,
    save_bm25_index,
    tokenize,
)


class DummyDenseRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.invocations = []

    def invoke(self, query):
        self.invocations.append(query)
        return self.documents


def test_tokenize_normalizes_medical_terms():
    assert tokenize("BP, UTI-like symptoms; COVID-19") == [
        "bp",
        "uti-like",
        "symptoms",
        "covid-19",
    ]


def test_bm25_prioritizes_exact_medical_terms():
    documents = [
        Document(
            page_content="Drink water and rest for mild headache.",
            metadata={"chunk_id": "general"},
        ),
        Document(
            page_content="UTI symptoms include dysuria and urinary frequency.",
            metadata={"chunk_id": "uti"},
        ),
        Document(
            page_content="Blood pressure monitoring can help track hypertension.",
            metadata={"chunk_id": "bp"},
        ),
    ]
    index = BM25Index.from_documents(documents)

    results = index.search("UTI dysuria", top_k=2)

    assert [result.document.metadata["chunk_id"] for result in results] == ["uti"]
    assert results[0].score > 0
    assert results[0].rank == 1


def test_bm25_index_round_trips_to_json(tmp_path):
    documents = [
        Document(
            page_content="MI can mean myocardial infarction.",
            metadata={"chunk_id": "mi"},
        )
    ]
    index_path = tmp_path / "bm25.json"

    save_bm25_index(BM25Index.from_documents(documents), index_path)
    loaded = load_bm25_index(index_path)

    assert (
        loaded.search("myocardial infarction")[0].document.metadata["chunk_id"] == "mi"
    )


def test_rrf_merge_deduplicates_by_chunk_id_and_blends_rankings():
    dense_documents = [
        Document(page_content="Dense first", metadata={"chunk_id": "dense"}),
        Document(page_content="Shared", metadata={"chunk_id": "shared"}),
    ]
    lexical_documents = [
        Document(page_content="Shared lexical", metadata={"chunk_id": "shared"}),
        Document(page_content="Lexical only", metadata={"chunk_id": "lexical"}),
    ]
    bm25_results = [
        type("Result", (), {"document": lexical_documents[0], "rank": 1})(),
        type("Result", (), {"document": lexical_documents[1], "rank": 2})(),
    ]

    merged = rrf_merge(dense_documents, bm25_results, final_k=3)

    assert [document.metadata["chunk_id"] for document in merged] == [
        "shared",
        "dense",
        "lexical",
    ]


def test_hybrid_retriever_invokes_dense_and_bm25_then_returns_fused_documents():
    dense_documents = [
        Document(
            page_content="General fever guidance.", metadata={"chunk_id": "dense"}
        ),
    ]
    lexical_documents = [
        Document(
            page_content="UTI symptoms include dysuria.", metadata={"chunk_id": "uti"}
        ),
    ]
    dense_retriever = DummyDenseRetriever(dense_documents)
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_index=BM25Index.from_documents(lexical_documents),
        bm25_k=5,
        final_k=2,
    )

    documents = retriever.invoke("UTI dysuria")

    assert dense_retriever.invocations == ["UTI dysuria"]
    assert {document.metadata["chunk_id"] for document in documents} == {"dense", "uti"}
