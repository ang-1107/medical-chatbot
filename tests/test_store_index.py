from types import SimpleNamespace

import pytest

from src.indexing import (
    _index_names,
    apply_chunk_metadata,
    ensure_pinecone_index,
    validate_index_compatibility,
)


class FakePinecone:
    def __init__(self, indexes, description=None):
        self.indexes = indexes
        self.description = description or {"dimension": 384, "metric": "cosine"}
        self.created_indexes = []

    def list_indexes(self):
        return self.indexes

    def create_index(self, **kwargs):
        self.created_indexes.append(kwargs)

    def describe_index(self, _index_name):
        return self.description


class IndexListWithNames:
    def __init__(self, names):
        self._names = names

    def names(self):
        return self._names


class FakeChunk:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def test_index_names_supports_names_method():
    indexes = IndexListWithNames(["medbot", "other"])

    assert _index_names(indexes) == {"medbot", "other"}


def test_index_names_supports_dict_and_object_entries():
    indexes = [{"name": "medbot"}, SimpleNamespace(name="other")]

    assert _index_names(indexes) == {"medbot", "other"}


def test_ensure_pinecone_index_skips_existing_index():
    pc = FakePinecone(IndexListWithNames(["medbot"]))

    created = ensure_pinecone_index(
        pc,
        index_name="medbot",
        dimension=384,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
    )

    assert created is False
    assert pc.created_indexes == []


def test_ensure_pinecone_index_creates_missing_index():
    pc = FakePinecone(IndexListWithNames([]))

    created = ensure_pinecone_index(
        pc,
        index_name="medbot",
        dimension=384,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
        spec_factory=lambda **kwargs: kwargs,
    )

    assert created is True
    assert len(pc.created_indexes) == 1
    assert pc.created_indexes[0]["name"] == "medbot"
    assert pc.created_indexes[0]["dimension"] == 384
    assert pc.created_indexes[0]["metric"] == "cosine"
    assert pc.created_indexes[0]["spec"] == {"cloud": "aws", "region": "us-east-1"}


def test_validate_index_compatibility_raises_on_dimension_mismatch():
    pc = FakePinecone(
        IndexListWithNames(["medbot"]),
        description={"dimension": 768, "metric": "cosine"},
    )

    with pytest.raises(RuntimeError, match="dimension"):
        validate_index_compatibility(
            pc,
            index_name="medbot",
            expected_dimension=384,
            expected_metric="cosine",
        )


def test_validate_index_compatibility_raises_on_metric_mismatch():
    pc = FakePinecone(
        IndexListWithNames(["medbot"]),
        description={"dimension": 384, "metric": "euclidean"},
    )

    with pytest.raises(RuntimeError, match="metric"):
        validate_index_compatibility(
            pc,
            index_name="medbot",
            expected_dimension=384,
            expected_metric="cosine",
        )


def test_apply_chunk_metadata_adds_stable_ids_and_metadata():
    chunks = [
        FakeChunk(
            page_content="headache and nausea",
            metadata={"source": "data/a.pdf", "page": 1},
        ),
        FakeChunk(
            page_content="headache and nausea",
            metadata={"source": "data/a.pdf", "page": 1},
        ),
    ]

    chunk_ids = apply_chunk_metadata(chunks)

    assert len(chunk_ids) == 2
    assert chunk_ids[0] != chunk_ids[1]
    assert chunks[0].metadata["chunk_id"] == chunk_ids[0]
    assert chunks[1].metadata["chunk_id"] == chunk_ids[1]
    assert chunks[0].metadata["content_hash"] == chunks[1].metadata["content_hash"]
    assert chunks[0].metadata["source_file"] == "data/a.pdf"
    assert chunks[0].metadata["page_number"] == "1"
