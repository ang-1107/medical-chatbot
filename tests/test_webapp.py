from pathlib import Path

from src.prompt import system_prompt
from src.webapp import app


class DummyChain:
    def __init__(self, answer: str):
        self.answer = answer
        self.invocations = []

    def invoke(self, payload):
        self.invocations.append(payload)
        return {"answer": self.answer}


def test_get_returns_json_answer_for_message():
    chain = DummyChain("Use the supplied context only.")
    app.config["TESTING"] = True
    app.config["RAG_CHAIN"] = chain

    with app.test_client() as client:
        response = client.post("/get", json={"msg": "I have a headache"})

    assert response.status_code == 200
    assert response.get_json() == {"answer": "Use the supplied context only."}
    assert chain.invocations == [{"input": "I have a headache"}]


def test_get_rejects_missing_message():
    app.config["TESTING"] = True
    app.config["RAG_CHAIN"] = DummyChain("unused")

    with app.test_client() as client:
        response = client.post("/get", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "Message is required."}


def test_get_rejects_too_long_message():
    app.config["TESTING"] = True
    app.config["RAG_CHAIN"] = DummyChain("unused")

    with app.test_client() as client:
        response = client.post("/get", json={"msg": "x" * 4001})

    assert response.status_code == 413
    assert response.get_json() == {"error": "Message is too long."}


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
