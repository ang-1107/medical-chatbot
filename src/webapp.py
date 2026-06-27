import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)
MAX_MESSAGE_LENGTH = 4000
_rag_chain = None


def _build_rag_chain():
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.vectorstores import Pinecone as LangchainPinecone
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_huggingface import HuggingFaceEndpoint

    from src.helper import download_hugging_face_embeddings

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    huggingfacehub_api_key = os.environ.get("HUGGINGFACEHUB_API_KEY")

    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")

    if not huggingfacehub_api_key:
        raise RuntimeError("HUGGINGFACEHUB_API_KEY is required")

    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["HUGGINGFACEHUB_API_KEY"] = huggingfacehub_api_key

    embeddings = download_hugging_face_embeddings()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "medbot")

    docsearch = LangchainPinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = HuggingFaceEndpoint(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token=huggingfacehub_api_key,
        temperature=0.4,
        max_new_tokens=500,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def get_rag_chain():
    configured_chain = app.config.get("RAG_CHAIN")
    if configured_chain is not None:
        return configured_chain

    global _rag_chain
    if _rag_chain is None:
        _rag_chain = _build_rag_chain()

    return _rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or (request.get_json(silent=True) or {}).get("msg")

    if msg is None or not str(msg).strip():
        return jsonify({"error": "Message is required."}), 400

    msg = str(msg).strip()
    if len(msg) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": "Message is too long."}), 413

    try:
        response = get_rag_chain().invoke({"input": msg})
        return jsonify({"answer": str(response["answer"])})
    except Exception:
        app.logger.exception("RAG request failed")
        return (
            jsonify({"error": "Sorry, I could not generate an answer right now."}),
            500,
        )
