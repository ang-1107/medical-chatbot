from pathlib import Path

from langchain_core.documents import Document

from src.prompt import system_prompt
from src.webapp import app


class DummyChain:
    def __init__(self, answer: str):
        self.answer = answer
        self.invocations = []

    def invoke(self, payload):
        self.invocations.append(payload)
        return {"answer": self.answer}


class DummyRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.invocations = []

    def invoke(self, payload):
        self.invocations.append(payload)
        return self.documents


class DummyService:
    def __init__(self, retriever, chain):
        self.retriever = retriever
        self.chain = chain


def test_get_returns_json_answer_for_message():
    documents = [
        Document(
            page_content="Headache guidance comes from the context.",
            metadata={
                "source_file": "data/a.pdf",
                "page_number": "3",
                "chunk_id": "chunk-1",
            },
        )
    ]
    retriever = DummyRetriever(documents)
    chain = DummyChain("Use the supplied context only.")
    app.config["TESTING"] = True
    app.config["RAG_SERVICE"] = DummyService(retriever, chain)

    with app.test_client() as client:
        response = client.post("/get", json={"msg": "I have a headache"})

    assert response.status_code == 200
    assert response.get_json() == {
        "answer": "Use the supplied context only.",
        "citations": [
            {
                "source": "data/a.pdf",
                "page": "3",
                "chunk_id": "chunk-1",
                "snippet": "Headache guidance comes from the context.",
            }
        ],
    }
    assert retriever.invocations == ["I have a headache"]
    assert chain.invocations == [{"input": "I have a headache"}]


def test_get_accepts_form_encoded_message():
    retriever = DummyRetriever([])
    chain = DummyChain("Form input works.")
    app.config["TESTING"] = True
    app.config["RAG_SERVICE"] = DummyService(retriever, chain)

    with app.test_client() as client:
        response = client.post("/get", data={"msg": "  What is BP?  "})

    assert response.status_code == 200
    assert response.get_json()["answer"] == "Form input works."
    assert retriever.invocations == ["What is BP?"]
    assert chain.invocations == [{"input": "What is BP?"}]


def test_get_rejects_missing_message():
    app.config["TESTING"] = True
    app.config["RAG_SERVICE"] = DummyService(DummyRetriever([]), DummyChain("unused"))

    with app.test_client() as client:
        response = client.post("/get", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "Message is required."}


def test_get_rejects_too_long_message():
    app.config["TESTING"] = True
    app.config["RAG_SERVICE"] = DummyService(DummyRetriever([]), DummyChain("unused"))

    with app.test_client() as client:
        response = client.post("/get", json={"msg": "x" * 4001})

    assert response.status_code == 413
    assert response.get_json() == {"error": "Message is too long."}


def test_citations_are_truncated_and_deduplicated():
    long_snippet = "x" * 300
    documents = [
        Document(
            page_content=long_snippet,
            metadata={
                "source_file": "data/a.pdf",
                "page_number": "3",
                "chunk_id": "chunk-1",
            },
        ),
        Document(
            page_content=long_snippet,
            metadata={
                "source_file": "data/a.pdf",
                "page_number": "3",
                "chunk_id": "chunk-1",
            },
        ),
    ]
    app.config["TESTING"] = True
    app.config["RAG_SERVICE"] = DummyService(
        DummyRetriever(documents),
        DummyChain("Answer."),
    )

    with app.test_client() as client:
        response = client.post("/get", json={"msg": "question"})

    citations = response.get_json()["citations"]
    assert len(citations) == 1
    assert len(citations[0]["snippet"]) == 240
    assert citations[0]["snippet"].endswith("...")


def test_template_uses_text_nodes_instead_of_raw_html_concatenation():
    template_text = Path("templates/chat.html").read_text(encoding="utf-8")

    assert "textContent" in template_text
    assert "createElement" in template_text
    assert "rawText +" not in template_text
    assert "data +" not in template_text


def test_prompt_contains_medical_safety_contract():
    assert "evidence is insufficient" in system_prompt
    assert "definitive diagnosis" in system_prompt
    assert "urgent red flags" in system_prompt
    assert "Summary:" in system_prompt
    assert "{context}" in system_prompt
    assert "source references" in system_prompt
