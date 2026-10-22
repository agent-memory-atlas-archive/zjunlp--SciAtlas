"""Hosted SciAtlas API backend for KG search.

Drop-in replacement for ``kg_search.service.run_search_with_authors`` that calls
the hosted SciAtlas API (http://sciatlas.openkg.cn) instead of a local Neo4j
instance, so external machines without the knowledge graph can still run the
search stage. Requires ``SCIATLAS_API_KEY`` (register at sciatlas.openkg.cn/register).

Speaks the KG2API Search API v1.0.0 envelope
(``{"request_id", "status", "result", "meta"}``): papers come from
``result.papers[]`` (typed fields: ``paper_id, title, abstract, authors, year,
venue, citation_count, score``) and authors from ``result.authors[]``
(``author_id, name, affiliation, paper_count, citation_count, h_index,
relevance_score``). Legacy ``ranking.papers`` / ``sources.kg.*`` shapes are kept
as fallbacks only.

The return shape mirrors the local backend:
    {"status", "backend", "papers": [...], "authors": [...], "paper_count", "author_count"}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCIATLAS_SRC = REPO_ROOT / "sciatlas" / "src"
REVIEW_PIPELINE = REPO_ROOT / "review_pipeline"

DEFAULT_API_BASE_URL = "http://sciatlas.openkg.cn"
DEFAULT_TIMEOUT_SECONDS = 300


def _load_env_values(env_path: str | None) -> dict[str, str]:
    if not env_path:
        return {}
    path = Path(env_path).expanduser()
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_lookup(env_values: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
        value = str(env_values.get(name) or "").strip()
        if value:
            return value
    return ""


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _unwrap_result(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    result = response.get("result")
    return result if isinstance(result, dict) else response


def _extract_papers(response: Any) -> list[dict[str, Any]]:
    result = _unwrap_result(response)
    # KG2API Search API v1.0.0 envelope: result.papers is the canonical list.
    papers = result.get("papers")
    if isinstance(papers, list):
        extracted = [item for item in papers if isinstance(item, dict)]
        if extracted:
            return extracted
    ranking = result.get("ranking")
    if isinstance(ranking, dict) and isinstance(ranking.get("papers"), list):
        papers = [item for item in ranking["papers"] if isinstance(item, dict)]
        if papers:
            return papers
    sources = result.get("sources")
    if isinstance(sources, dict):
        kg_payload = sources.get("kg")
        if isinstance(kg_payload, dict) and isinstance(kg_payload.get("papers"), list):
            return [item for item in kg_payload["papers"] if isinstance(item, dict)]
    if isinstance(result.get("results"), list):
        return [item for item in result["results"] if isinstance(item, dict)]
    return []


def _extract_authors(response: Any) -> list[dict[str, Any]]:
    result = _unwrap_result(response)
    # KG2API Search API v1.0.0 envelope: result.authors (search and authors/related).
    authors = result.get("authors")
    if isinstance(authors, list):
        extracted = [item for item in authors if isinstance(item, dict)]
        if extracted:
            return extracted
    sources = result.get("sources")
    if isinstance(sources, dict):
        kg_payload = sources.get("kg")
        if isinstance(kg_payload, dict) and isinstance(kg_payload.get("authors"), list):
            return [item for item in kg_payload["authors"] if isinstance(item, dict)]
    if isinstance(result.get("related_authors"), list):
        return [item for item in result["related_authors"] if isinstance(item, dict)]
    return []


def _flatten_paper(raw: dict[str, Any]) -> dict[str, Any]:
    inner = raw.get("paper") if isinstance(raw.get("paper"), dict) else {}
    return {**inner, **{key: value for key, value in raw.items() if key != "paper" and value is not None}}


def _normalize_paper(raw: dict[str, Any]) -> dict[str, Any] | None:
    title = str(_first(raw.get("title"), raw.get("display_name"), raw.get("paper_title")) or "").strip()
    if not title:
        return None
    # KG2API Search API v1.0.0 typed fields come first; legacy aliases follow.
    paper_id = str(_first(raw.get("paper_id"), raw.get("id"), raw.get("paper_url")) or "").strip()
    year = _first(raw.get("year"), raw.get("publication_year"))
    cited_by = _first(raw.get("citation_count"), raw.get("cited_by_count"), raw.get("citations"))
    doi = str(_first(raw.get("doi")) or "").strip()
    pdf_url = str(_first(raw.get("pdf_url"), raw.get("open_access_pdf_url")) or "").strip()
    if not pdf_url:
        open_access = raw.get("openAccessPdf") or raw.get("open_access_pdf")
        if isinstance(open_access, dict):
            pdf_url = str(open_access.get("url") or "").strip()

    raw_authors = _first(raw.get("authors"), raw.get("author_names")) or []
    author_names: list[str] = []
    if isinstance(raw_authors, list):
        for item in raw_authors:
            if isinstance(item, str) and item.strip():
                author_names.append(item.strip())
            elif isinstance(item, dict):
                name = str(_first(item.get("name"), item.get("display_name")) or "").strip()
                if name:
                    author_names.append(name)

    paper: dict[str, Any] = {
        "title": title,
        "abstract": str(_first(raw.get("abstract"), raw.get("abstract_text")) or ""),
        "authors": author_names,
        "source": "kg_hosted",
    }
    if paper_id:
        paper["id"] = paper_id
    paper_url = str(_first(raw.get("paper_url"), raw.get("url")) or "").strip()
    if paper_url:
        paper["url"] = paper_url
    if year is not None:
        paper["publication_year"] = year
    if cited_by is not None:
        paper["cited_by_count"] = cited_by
    if doi:
        paper["doi"] = doi
    if pdf_url:
        paper["pdf_url"] = pdf_url
    score = _first(raw.get("score"), raw.get("final_score"))
    if score is not None:
        paper["score"] = score
    venue = str(_first(raw.get("venue"), raw.get("source_name")) or "").strip()
    if venue:
        paper["venue"] = venue
    return paper


def _normalize_author(raw: dict[str, Any]) -> dict[str, Any] | None:
    name = str(_first(raw.get("name"), raw.get("display_name")) or "").strip()
    if not name:
        return None
    author: dict[str, Any] = {
        "author_id": str(_first(raw.get("author_id"), raw.get("id")) or "").strip(),
        "name": name,
        # KG2API Search API v1.0.0 related-authors field is relevance_score.
        "score": float(_first(raw.get("relevance_score"), raw.get("score"), 0.0) or 0.0),
    }
    affiliation = str(_first(raw.get("affiliation")) or "").strip()
    if affiliation:
        author["affiliation"] = affiliation
    works_count = _first(raw.get("paper_count"), raw.get("works_count"))
    if works_count is not None:
        author["works_count"] = works_count
    citation_count = _first(raw.get("citation_count"), raw.get("citations"))
    if citation_count is not None:
        author["citation_count"] = citation_count
    h_index = _first(raw.get("h_index"))
    if h_index is not None:
        author["h_index"] = h_index
    return author


def _build_client(env_path: str | Path | None):
    if str(SCIATLAS_SRC) not in sys.path:
        sys.path.insert(0, str(SCIATLAS_SRC))
    from sciatlas.client import SciAtlasClient

    env_values = _load_env_values(env_path)
    api_key = _env_lookup(env_values, "SCIATLAS_API_KEY")
    if not api_key:
        raise RuntimeError("SCIATLAS_API_KEY is required for the hosted KG backend (register at sciatlas.openkg.cn)")
    base_url = _env_lookup(env_values, "SCIATLAS_API_BASE_URL") or DEFAULT_API_BASE_URL
    timeout_raw = _env_lookup(env_values, "SCIATLAS_TIMEOUT")
    timeout = int(timeout_raw) if timeout_raw.isdigit() else DEFAULT_TIMEOUT_SECONDS
    return SciAtlasClient(api_key=api_key, base_url=base_url, timeout=timeout)


def hosted_kg_search(*, query_text: str, top_k: int, env_path: str | Path | None) -> dict[str, Any]:
    """Query the hosted SciAtlas KG ``/v1/search`` endpoint.

    Shared by the kg_search service backend and the rubric sources stage. Returns
    ``{"papers": [...normalized...], "authors": [...normalized...], "response": raw}``.
    """
    client = _build_client(env_path)
    query = str(query_text or "").strip()
    if not query:
        raise ValueError("hosted KG backend requires a non-empty query text")

    response = client.search_papers(query=query, top_k=top_k)
    papers = [
        normalized
        for normalized in (_normalize_paper(_flatten_paper(raw)) for raw in _extract_papers(response))
        if normalized is not None
    ]
    authors = [
        normalized
        for normalized in (_normalize_author(raw) for raw in _extract_authors(response))
        if normalized is not None
    ]
    result = _unwrap_result(response)
    total_count = result.get("total_count")
    return {"papers": papers, "authors": authors, "total_count": total_count, "response": response}


def run_search_with_authors(args: Any) -> dict[str, Any]:
    query = str(getattr(args, "idea_text", "") or "").strip()
    if not query:
        raise ValueError("hosted KG backend requires idea_text")
    top_k = int(getattr(args, "top_k", 20) or 20)

    search = hosted_kg_search(query_text=query, top_k=top_k, env_path=getattr(args, "env", None))
    papers = search["papers"]
    authors = search["authors"]
    if not authors:
        try:
            client = _build_client(getattr(args, "env", None))
            authors_response = client.related_authors(query=query, top_k=max(min(top_k, 10), 5))
            authors = [
                normalized
                for normalized in (_normalize_author(raw) for raw in _extract_authors(authors_response))
                if normalized is not None
            ]
        except Exception as exc:
            print(
                f"[service_hosted] related-authors lookup failed; continuing without authors: "
                f"{exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )
            authors = []

    if not papers:
        raise RuntimeError("hosted SciAtlas search returned no papers")

    return {
        "status": "ok",
        "backend": "hosted",
        "papers": papers,
        "authors": authors,
        "paper_count": len(papers),
        "author_count": len(authors),
    }
