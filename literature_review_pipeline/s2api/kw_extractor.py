#!/usr/bin/env python3
"""Standalone academic search seed extraction service backed by an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_URL = "https://www.dmxapi.cn/v1/chat/completions"
DEFAULT_MODEL = "DeepSeek-V3.2"
DEFAULT_ENV_PATH = Path("/home/weiyunxiang/yunx/.env")
DEFAULT_TIMEOUT = 60

KEYWORD_PROMPT = """You are an expert assistant that extracts academic search seed phrases for literature retrieval.

Goal:
From an academic text, extract a very small set of canonical seed phrases that can be expanded into search queries for related-paper retrieval.

Important:
The goal is NOT to build knowledge graph entities.
The goal is to capture the paper's distinctive task/problem phrasing in a form that could plausibly appear in related paper titles.

Requirements:
- Return only 1-3 seed phrases.
- Prefer concise noun phrases of 2-5 words.
- Copy the canonical task/problem wording from the input when possible.
- Preserve distinctive task nouns such as research idea evaluation instead of broadening to only evaluation.
- Favor phrases that can seed paper-search queries for neighboring literature.

Avoid:
- High-level umbrella fields used alone, such as large language models, evaluation, benchmark, reasoning, knowledge grounding.
- Broad management or economics terms such as innovation assessment.
- Framework names, organizational details, consensus mechanisms, or highly customized phrases.
- Long descriptive clauses copied from the input.

Good seed examples:
research idea evaluation
scientific idea generation
novelty assessment
retrieval augmented generation
protein structure prediction

Bad seed examples:
large language models
innovation evaluation framework
knowledge grounded multi-perspective reasoning
review board consensus
multi-criteria decision making

Input text:
{input_text}

Output format:
Return ONLY a JSON object: {{"queries": ["seed1", "seed2"]}} with no extra text."""


class KeywordExtractionError(RuntimeError):
    """Raised when keyword extraction fails."""


@dataclass(frozen=True)
class KeywordExtractorConfig:
    api_url: str = API_URL
    model: str = DEFAULT_MODEL
    env_path: Path = DEFAULT_ENV_PATH
    timeout: int = DEFAULT_TIMEOUT
    use_env_proxy: bool = False


API_KEY_ALIASES = ("DMX-API-KEY", "DMX_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
_API_URL_ENV_NAMES = (
    "SEARCH_LLM_API_URL",
    "LLM_CHAT_COMPLETIONS_URL",
    "LLM_API_URL",
    "OPENAI_CHAT_COMPLETIONS_URL",
)
_BASE_URL_ENV_NAMES = ("LLM_BASE_URL", "OPENAI_BASE_URL")
_MODEL_ENV_NAMES = ("SEARCH_LLM_MODEL", "LLM_MODEL", "OPENAI_MODEL")


def _load_env_file_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _lookup_env(env_values: dict[str, str], *names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip() or env_values.get(name, "").strip()
        if value:
            return value
    return ""


def load_api_key(env_path: Path) -> str:
    env_values = _load_env_file_values(env_path)
    api_key = _lookup_env(env_values, *API_KEY_ALIASES)
    if api_key:
        return api_key
    raise ValueError(
        f"No LLM API key found in the environment or {env_path}. "
        "Set LLM_API_KEY (preferred), OPENAI_API_KEY or DMX-API-KEY (legacy)."
    )


def resolve_api_url(env_path: Path, configured: str) -> str:
    if configured and configured != API_URL:
        return configured
    env_values = _load_env_file_values(env_path)
    explicit = _lookup_env(env_values, *_API_URL_ENV_NAMES)
    if explicit:
        return explicit
    base_url = _lookup_env(env_values, *_BASE_URL_ENV_NAMES).rstrip("/")
    if base_url:
        return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    return configured


def resolve_model(env_path: Path, configured: str) -> str:
    if configured and configured != DEFAULT_MODEL:
        return configured
    env_values = _load_env_file_values(env_path)
    return _lookup_env(env_values, *_MODEL_ENV_NAMES) or configured


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def parse_keywords_response(content: str) -> list[str]:
    cleaned = strip_code_fence(content)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise KeywordExtractionError(f"Invalid response content: {content!r}") from None
        payload = json.loads(cleaned[start : end + 1])

    keywords = payload.get("keywords")
    if keywords is None:
        keywords = payload.get("queries")
    if not isinstance(keywords, list):
        raise KeywordExtractionError(f"Missing keywords in response: {payload!r}")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in keywords:
        keyword = " ".join(str(item).split()).strip()
        if not keyword:
            continue
        dedupe_key = keyword.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(keyword)

    if not normalized:
        raise KeywordExtractionError(f"Empty keyword list in response: {payload!r}")
    return normalized


class KeywordExtractor:
    def __init__(self, config: KeywordExtractorConfig | None = None) -> None:
        self.config = config or KeywordExtractorConfig()
        self.api_key = load_api_key(self.config.env_path)
        self.api_url = resolve_api_url(self.config.env_path, self.config.api_url)
        self.model = resolve_model(self.config.env_path, self.config.model)
        self.opener = self._build_opener()

    def _build_opener(self) -> urllib.request.OpenerDirector:
        if self.config.use_env_proxy:
            return urllib.request.build_opener()
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def extract_keywords(self, input_text: str) -> list[str]:
        text = input_text.strip()
        if not text:
            raise ValueError("input_text must not be empty")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You extract academic search seed phrases and return strict JSON only.",
                },
                {"role": "user", "content": KEYWORD_PROMPT.format(input_text=text)},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with self.opener.open(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise KeywordExtractionError(f"HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise KeywordExtractionError(f"Request failed: {exc}") from exc

        try:
            response_payload: dict[str, Any] = json.loads(body)
            content = response_payload["choices"][0]["message"]["content"]
        except Exception as exc:
            raise KeywordExtractionError(f"Unexpected response: {body}") from exc

        return parse_keywords_response(content)


def extract_keywords(
    input_text: str,
    *,
    env_path: Path = DEFAULT_ENV_PATH,
    model: str = DEFAULT_MODEL,
    api_url: str = API_URL,
    timeout: int = DEFAULT_TIMEOUT,
    use_env_proxy: bool = False,
) -> list[str]:
    extractor = KeywordExtractor(
        KeywordExtractorConfig(
            api_url=api_url,
            model=model,
            env_path=env_path,
            timeout=timeout,
            use_env_proxy=use_env_proxy,
        )
    )
    return extractor.extract_keywords(input_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract high-level keywords from input text.")
    parser.add_argument("input_text", nargs="*", help="Input text to extract keywords from.")
    parser.add_argument("--stdin", action="store_true", help="Read input text from stdin.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DMX model name.")
    parser.add_argument("--api-url", default=API_URL, help="Chat completions endpoint.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds.")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from the environment instead of direct connection.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.stdin:
        input_text = sys.stdin.read().strip()
    else:
        input_text = " ".join(args.input_text).strip()

    if not input_text:
        print("input_text must not be empty", file=sys.stderr)
        return 1

    try:
        keywords = extract_keywords(
            input_text,
            env_path=Path(args.env),
            model=args.model,
            api_url=args.api_url,
            timeout=args.timeout,
            use_env_proxy=args.use_env_proxy,
        )
    except Exception as exc:
        print(f"keyword extraction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(keywords, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
