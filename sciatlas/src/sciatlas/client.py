from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import SciAtlasConfig, load_config


class SciAtlasClient:
    """Small Python client for the hosted SciAtlas API service."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.config: SciAtlasConfig = load_config(base_url=base_url, api_key=api_key, timeout=timeout)

    def _url(self, endpoint: str) -> str:
        return self.config.base_url.rstrip("/") + "/" + endpoint.lstrip("/")

    @staticmethod
    def _raise_for_envelope_status(body: Any, endpoint: str) -> None:
        """KG2API Search API v1.0.0 envelope: ``status != "ok"`` is a server error."""
        if not isinstance(body, dict):
            return
        status = body.get("status")
        if status is None or str(status).lower() == "ok":
            return
        message = body.get("error") or body.get("message") or body.get("detail")
        if isinstance(message, (dict, list)):
            message = json.dumps(message, ensure_ascii=False)
        prefix = f"SciAtlas API returned status={status!r} for {endpoint}"
        request_id = body.get("request_id")
        if request_id:
            prefix += f" (request_id={request_id})"
        raise RuntimeError(f"{prefix}: {message or 'no error message provided'}")

    def request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json"}
        data = None

        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        if endpoint.rstrip("/") != "/healthz":
            if not self.config.api_key:
                raise RuntimeError("SCIATLAS_API_KEY is required for this endpoint.")
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            headers["X-API-Key"] = self.config.api_key

        req = urllib.request.Request(
            self._url(endpoint),
            data=data,
            headers=headers,
            method=method.upper(),
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                body = json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw)
            except json.JSONDecodeError:
                detail = raw
            raise RuntimeError(f"SciAtlas API error {exc.code}: {detail}") from exc

        if endpoint.rstrip("/") != "/healthz":
            self._raise_for_envelope_status(body, endpoint)
        return body

    def health(self) -> Any:
        return self.request("GET", "/healthz")

    def token_status(self) -> Any:
        return self.request("GET", "/v1/auth/token/status")

    def usage(self, days: int = 7) -> Any:
        return self.request("GET", f"/v1/auth/usage?days={days}")

    def search(self, plan: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
        return self.request("POST", "/v1/search", {"plan": plan, "options": options or {}})

    def search_papers(
        self,
        *,
        query: str,
        keywords: list[dict[str, Any]] | None = None,
        titles: list[dict[str, Any]] | None = None,
        reference_titles: list[str] | None = None,
        top_k: int = 10,
        retrieval_mode: str = "hybrid",
        **options: Any,
    ) -> Any:
        plan = {
            "query_text": query,
            "source_type": "idea_text",
            "source_title": None,
            "keywords": keywords or [{"text": query, "score": 8}],
            "titles": titles or [],
            "reference_titles": reference_titles or [],
        }
        merged_options = {"top_k": top_k, "retrieval_mode": retrieval_mode, **options}
        return self.search(plan, merged_options)

    def related_authors(self, *, query: str, keywords: list[dict[str, Any]] | None = None, top_k: int = 5) -> Any:
        plan = {
            "query_text": query,
            "source_type": "idea_text",
            "source_title": None,
            "keywords": keywords or [{"text": query, "score": 8}],
            "titles": [],
            "reference_titles": [],
        }
        return self.request("POST", "/v1/authors/related", {"plan": plan, "options": {"top_k": top_k}})

    def author_papers(self, author: str, *, limit: int = 10, include_abstract: bool = False) -> Any:
        payload = {
            "identifier": author,
            "search_by": "name",
            "options": {
                "limit": limit,
                "include_abstract": include_abstract,
                "include_embeddings": False,
                "merge_same_name_authors": True,
                "dedupe_papers": True,
            },
        }
        return self.request("POST", "/v1/authors/papers", payload)
