import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv

DEFAULT_INDEX_NAME = "medbot"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_PINECONE_CLOUD = "aws"
DEFAULT_PINECONE_REGION = "us-east-1"
DEFAULT_INDEX_METRIC = "cosine"


def _index_names(indexes: Any) -> set[str]:
    """Normalize Pinecone list_indexes responses across client versions."""
    if hasattr(indexes, "names"):
        return set(indexes.names())

    names: set[str] = set()
    if isinstance(indexes, Iterable):
        for index in indexes:
            if isinstance(index, str):
                names.add(index)
            elif isinstance(index, dict) and "name" in index:
                names.add(str(index["name"]))
            elif not isinstance(index, dict) and hasattr(index, "name"):
                names.add(str(index.name))

    return names


def _get_index_dimension_and_metric(
    index_description: Any,
) -> tuple[int | None, str | None]:
    """Extract dimension and metric from varying Pinecone describe responses."""
    if isinstance(index_description, dict):
        dimension = index_description.get("dimension")
        metric = index_description.get("metric")
        return _as_int(dimension), _as_str(metric)

    dimension = getattr(index_description, "dimension", None)
    metric = getattr(index_description, "metric", None)

    if dimension is not None or metric is not None:
        return _as_int(dimension), _as_str(metric)

    if hasattr(index_description, "to_dict"):
        return _get_index_dimension_and_metric(index_description.to_dict())

    return None, None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def validate_index_compatibility(
    pc: Any,
    *,
    index_name: str,
    expected_dimension: int,
    expected_metric: str,
) -> None:
    description = pc.describe_index(index_name)
    actual_dimension, actual_metric = _get_index_dimension_and_metric(description)

    if actual_dimension is not None and actual_dimension != expected_dimension:
        raise RuntimeError(
            f"Pinecone index '{index_name}' has dimension {actual_dimension}, "
            f"but EMBEDDING_DIMENSION is {expected_dimension}. Use a compatible index "
            "or rebuild the index for the current embedding model."
        )

    if actual_metric and actual_metric.lower() != expected_metric.lower():
        raise RuntimeError(
            f"Pinecone index '{index_name}' uses metric '{actual_metric}', "
            f"but expected '{expected_metric}'."
        )


def ensure_pinecone_index(
    pc: Any,
    *,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
    spec_factory: Any | None = None,
) -> bool:
    """Create the Pinecone index if missing.

    Returns True when a new index is created and False when it already exists.
    """
    existing_indexes = _index_names(pc.list_indexes())

    if index_name in existing_indexes:
        validate_index_compatibility(
            pc,
            index_name=index_name,
            expected_dimension=dimension,
            expected_metric=metric,
        )
        return False

    if spec_factory is None:
        from pinecone import ServerlessSpec

        spec_factory = ServerlessSpec

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=spec_factory(cloud=cloud, region=region),
    )
    return True


def _hash_content(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def apply_chunk_metadata(text_chunks: list[Any]) -> list[str]:
    """Attach deterministic metadata and stable IDs for repeatable upserts."""
    per_chunk_occurrences: defaultdict[str, int] = defaultdict(int)
    chunk_ids: list[str] = []

    for chunk in text_chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = str(chunk.metadata.get("page", "unknown"))
        content_hash = _hash_content(chunk.page_content)

        base_key = f"{source}|{page}|{content_hash}"
        occurrence = per_chunk_occurrences[base_key]
        per_chunk_occurrences[base_key] += 1

        chunk_id = f"{base_key}|{occurrence}"

        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["page_number"] = page
        chunk.metadata["source_file"] = source
        chunk_ids.append(chunk_id)

    return chunk_ids


def _embedding_dimension(embeddings: Any) -> int | None:
    if hasattr(embeddings, "client") and hasattr(
        embeddings.client, "get_sentence_embedding_dimension"
    ):
        value = embeddings.client.get_sentence_embedding_dimension()
        if value is not None:
            return int(value)
    return None


def main() -> None:
    from inspect import signature

    from langchain_community.vectorstores import Pinecone as LangchainPinecone
    from pinecone.grpc import PineconeGRPC as Pinecone

    from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split

    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")

    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    index_name = os.environ.get("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    cloud = os.environ.get("PINECONE_CLOUD", DEFAULT_PINECONE_CLOUD)
    region = os.environ.get("PINECONE_REGION", DEFAULT_PINECONE_REGION)
    embedding_model_name = os.environ.get(
        "EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME
    )
    dimension = int(
        os.environ.get("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
    )

    extracted_data = load_pdf_file(data="data/")
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings(model_name=embedding_model_name)

    inferred_dimension = _embedding_dimension(embeddings)
    if inferred_dimension is not None and inferred_dimension != dimension:
        raise RuntimeError(
            f"Configured EMBEDDING_DIMENSION={dimension}, but model "
            f"'{embedding_model_name}' reports dimension {inferred_dimension}."
        )

    chunk_ids = apply_chunk_metadata(text_chunks)

    pc = Pinecone(api_key=pinecone_api_key)
    ensure_pinecone_index(
        pc,
        index_name=index_name,
        dimension=dimension,
        metric=DEFAULT_INDEX_METRIC,
        cloud=cloud,
        region=region,
    )

    from_documents_kwargs: dict[str, Any] = {
        "documents": text_chunks,
        "index_name": index_name,
        "embedding": embeddings,
    }
    if "ids" in signature(LangchainPinecone.from_documents).parameters:
        from_documents_kwargs["ids"] = chunk_ids

    LangchainPinecone.from_documents(**from_documents_kwargs)


if __name__ == "__main__":
    main()
