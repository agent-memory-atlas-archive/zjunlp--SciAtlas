import argparse
import json

from sciatlas import cli


_LLM_ENV_NAMES = (
    "LLM_PROVIDER",
    "SCIATLAS_LLM_PROVIDER",
    "LLM_API_KEY",
    "SCIATLAS_LLM_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "SCIATLAS_LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "LLM_CHAT_COMPLETIONS_URL",
    "SCIATLAS_LLM_CHAT_COMPLETIONS_URL",
    "OPENAI_CHAT_COMPLETIONS_URL",
    "LLM_MODEL",
    "SCIATLAS_LLM_MODEL",
    "OPENAI_MODEL",
)


def _clear_llm_environment(monkeypatch):
    for name in _LLM_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_config_reports_status_without_llm_configuration(monkeypatch, capsys):
    _clear_llm_environment(monkeypatch)
    args = argparse.Namespace(
        base_url="http://example.invalid",
        api_key="",
        runs_dir="runs",
    )

    assert cli.cmd_config(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["optional_provider_config"]["llm_keyword_extraction_enabled"] is False


def test_text_plan_uses_local_keywords_without_llm_configuration(monkeypatch):
    _clear_llm_environment(monkeypatch)

    plan = cli.build_plan_from_text(text="retrieval augmented generation")

    assert plan["query_text"] == "retrieval augmented generation"
    assert plan["keywords"]


def test_optional_llm_config_preserves_explicit_endpoint(monkeypatch):
    _clear_llm_environment(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_URL", "https://gateway.invalid/custom")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    config = cli._optional_llm_config()

    assert config is not None
    assert config["chat_completions_url"] == "https://gateway.invalid/custom"
