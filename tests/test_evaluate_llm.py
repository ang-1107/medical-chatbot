from src.evaluate_llm import (
    DEFAULT_LLM_MODELS,
    LLMExamplePayload,
    build_argument_parser,
    build_prompt,
    score_answer,
)
from src.llm_config import DEFAULT_LLM_MODEL_NAME


def test_default_llm_model_is_llama_3_1_8b_instruct():
    assert DEFAULT_LLM_MODEL_NAME == "meta-llama/Llama-3.1-8B-Instruct"
    assert DEFAULT_LLM_MODELS[0] == DEFAULT_LLM_MODEL_NAME


def test_score_answer_rewards_keywords_citations_safety_and_brevity():
    answer = (
        "An abscess is a pocket of pus caused by infection. Treatment often "
        "involves drainage and sometimes antibiotics. Seek urgent medical care "
        "for worsening fever or severe symptoms [source=data/book.pdf page=27]."
    )
    scores = score_answer(
        answer,
        "An abscess is a pocket of pus caused by infection. Treatment usually "
        "involves drainage and sometimes antibiotics.",
    )

    assert scores["keyword_coverage"] > 0.7
    assert scores["citation_score"] == 1.0
    assert scores["medical_safety_score"] > 0
    assert scores["overall"] > 0.6


def test_build_prompt_includes_context_question_and_citation_instruction():
    example = LLMExamplePayload(
        question="What is acne?",
        reference_answer="Acne is an inflammatory skin condition.",
        context="[source=data/book.pdf page=38] Acne is a skin condition.",
        relevant_pages=(38,),
    )

    prompt = build_prompt(example)

    assert "Answer only from the provided context" in prompt
    assert "source/page citations" in prompt
    assert "What is acne?" in prompt
    assert "Acne is a skin condition" in prompt


def test_argument_parser_defaults_to_five_candidate_models():
    parser = build_argument_parser()
    args = parser.parse_args([])

    assert args.model_names is None
    assert len(DEFAULT_LLM_MODELS) == 5


def test_argument_parser_accepts_custom_models():
    parser = build_argument_parser()
    args = parser.parse_args(["--model", "a/b", "--model", "c/d"])

    assert args.model_names == ["a/b", "c/d"]
