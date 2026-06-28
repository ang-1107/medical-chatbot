import os
import re

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL_NAME = "NeuML/biomedbert-small-embeddings"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80

_HEADING_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Z0-9 ,:/()\-]{4,}$")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


# To extract the data from our pdf file
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)  # type: ignore[arg-type]
    documents = loader.load()
    return documents


def _split_into_paragraphs(page_content: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", page_content)]
    return [paragraph for paragraph in paragraphs if paragraph]


def _normalize_heading(paragraph: str) -> str:
    return re.sub(r"\s+", " ", paragraph.replace("\n", " ")).strip().rstrip(":")


def _classify_paragraph(paragraph: str) -> str:
    first_line = paragraph.splitlines()[0].strip()
    if _HEADING_PATTERN.match(first_line):
        return "heading"
    if _BULLET_PATTERN.match(first_line):
        return "list"
    return "paragraph"


def _split_large_component(
    text: str, *, chunk_size: int, chunk_overlap: int
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )
    return splitter.split_text(text)


def text_split(
    extracted_data,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    text_chunks: list[Document] = []

    for page_document in extracted_data:
        page_metadata = dict(getattr(page_document, "metadata", {}) or {})
        source_file = str(page_metadata.get("source", "unknown"))
        page_number = str(
            page_metadata.get("page", page_metadata.get("page_number", ""))
        )
        current_heading = ""
        chunk_number = 0

        for paragraph_index, paragraph in enumerate(
            _split_into_paragraphs(page_document.page_content)
        ):
            component_type = _classify_paragraph(paragraph)
            section_heading = current_heading

            if component_type == "heading":
                current_heading = _normalize_heading(paragraph)
                section_heading = current_heading
                component_chunks = [paragraph]
            else:
                component_chunks = [paragraph]
                if len(paragraph) > chunk_size:
                    component_chunks = _split_large_component(
                        paragraph, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )

            for component_index, component_chunk in enumerate(component_chunks):
                emitted_component_type = component_type
                if component_type != "heading" and len(component_chunks) > 1:
                    emitted_component_type = f"{component_type}-part"

                metadata = {
                    **page_metadata,
                    "source_file": source_file,
                    "page_number": page_number,
                    "paragraph_index": paragraph_index,
                    "component_index": component_index,
                    "component_type": emitted_component_type,
                    "section_heading": section_heading,
                    "chunk_number": chunk_number,
                }
                text_chunks.append(
                    Document(page_content=component_chunk, metadata=metadata)
                )
                chunk_number += 1

    return text_chunks


def download_hugging_face_embeddings(model_name=DEFAULT_EMBEDDING_MODEL_NAME):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
