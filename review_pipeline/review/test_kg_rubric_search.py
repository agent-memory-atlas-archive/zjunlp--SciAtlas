from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REVIEW_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(REVIEW_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REVIEW_PIPELINE_ROOT))
SCIAATLAS_SRC = Path(__file__).resolve().parents[2] / "sciatlas" / "src"
if str(SCIAATLAS_SRC) not in sys.path:
    sys.path.insert(0, str(SCIAATLAS_SRC))

from kg_search import service_hosted
from review import idea_rubric
from review.idea_rubric import (
    IdeaRubricConfig,
    IdeaRubricRunner,
    KgRubricSearch,
)
from sciatlas.client import SciAtlasClient


def fake_search_envelope() -> dict:
    """KG2API Search API v1.0.0 envelope for POST /v1/search."""
    return {
        "request_id": "req-test-search",
        "status": "ok",
        "result": {
            "papers": [
                {
                    "paper_id": "W1",
                    "title": "Graph Neural Networks for Catalyst Stability Prediction",
                    "abstract": "We predict catalyst stability with GNNs.",
                    "authors": ["A. Author"],
                    "year": 2023,
                    "venue": "Nature Catalysis",
                    "citation_count": 12,
                    "score": 0.9,
                },
                {
                    "paper_id": "W2",
                    "title": "Unrelated Vision Transformer Study",
                    "abstract": "Image classification with ViTs.",
                    "authors": ["B. Author"],
                    "year": 2022,
                    "venue": "CVPR",
                    "citation_count": 3,
                    "score": 0.4,
                },
                {
                    "paper_id": "W3",
                    "title": "Graph Neural Networks for Catalyst Stability Prediction",
                    "abstract": "Duplicate title entry.",
                    "authors": [],
                    "year": None,
                    "venue": "",
                    "citation_count": 0,
                    "score": 0.5,
                },
            ],
            "total_count": 3,
        },
        "meta": None,
    }


def fake_related_authors_envelope() -> dict:
    """KG2API Search API v1.0.0 envelope for POST /v1/authors/related."""
    return {
        "request_id": "req-test-authors",
        "status": "ok",
        "result": {"authors": [], "total_count": 0},
        "meta": None,
    }


class FakeHostedClient:
    """Stand-in for SciAtlasClient returning canned envelopes without network I/O."""

    def __init__(self, search_response: dict, related_authors_response: dict | None = None) -> None:
        self._search_response = search_response
        self._related_authors_response = related_authors_response or fake_related_authors_envelope()

    def search_papers(self, *, query: str, top_k: int, **kwargs) -> dict:
        return self._search_response

    def related_authors(self, *, query: str, top_k: int = 5, **kwargs) -> dict:
        return self._related_authors_response


def patch_hosted_client(search_response: dict | None = None):
    client = FakeHostedClient(search_response or fake_search_envelope())
    return patch.object(service_hosted, "_build_client", return_value=client)


class ServiceHostedExtractionTests(unittest.TestCase):
    def test_extract_papers_prefers_result_papers_over_legacy_shapes(self) -> None:
        response = {
            "status": "ok",
            "result": {
                "papers": [
                    {
                        "paper_id": "W1",
                        "title": "Envelope Paper",
                        "abstract": "abstract text",
                        "authors": ["A. Author", "B. Author"],
                        "year": 2024,
                        "venue": "NeurIPS",
                        "citation_count": 5,
                        "score": 0.7,
                    }
                ],
                "total_count": 1,
                "ranking": {"papers": [{"title": "Legacy Ranking Paper", "id": "L1", "publication_year": 2020}]},
            },
            "meta": None,
        }
        papers = service_hosted._extract_papers(response)
        self.assertEqual([paper.get("paper_id") for paper in papers], ["W1"])

        normalized = service_hosted._normalize_paper(papers[0])
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["id"], "W1")
        self.assertEqual(normalized["publication_year"], 2024)
        self.assertEqual(normalized["cited_by_count"], 5)
        self.assertEqual(normalized["score"], 0.7)
        self.assertEqual(normalized["authors"], ["A. Author", "B. Author"])
        self.assertEqual(normalized["venue"], "NeurIPS")
        self.assertEqual(normalized["source"], "kg_hosted")

    def test_extract_papers_falls_back_to_legacy_ranking_shape(self) -> None:
        response = {"result": {"ranking": {"papers": [{"title": "Legacy Paper", "id": "L1"}]}}}
        papers = service_hosted._extract_papers(response)
        self.assertEqual([paper.get("id") for paper in papers], ["L1"])

    def test_extract_authors_prefers_result_authors_with_relevance_score(self) -> None:
        response = {
            "status": "ok",
            "result": {
                "authors": [
                    {
                        "author_id": "A1",
                        "name": "Jane Doe",
                        "affiliation": "Example University",
                        "paper_count": 10,
                        "citation_count": 100,
                        "h_index": 8,
                        "relevance_score": 0.95,
                    }
                ],
                "total_count": 1,
            },
            "meta": None,
        }
        authors = service_hosted._extract_authors(response)
        self.assertEqual(len(authors), 1)

        normalized = service_hosted._normalize_author(authors[0])
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["author_id"], "A1")
        self.assertEqual(normalized["score"], 0.95)
        self.assertEqual(normalized["works_count"], 10)
        self.assertEqual(normalized["citation_count"], 100)
        self.assertEqual(normalized["h_index"], 8)
        self.assertEqual(normalized["affiliation"], "Example University")

    def test_hosted_kg_search_returns_normalized_papers_and_total_count(self) -> None:
        with patch_hosted_client():
            result = service_hosted.hosted_kg_search(query_text="graph neural networks", top_k=10, env_path=None)
        self.assertEqual(result["total_count"], 3)
        self.assertEqual([paper["id"] for paper in result["papers"]], ["W1", "W2", "W3"])
        self.assertEqual(result["papers"][0]["publication_year"], 2023)
        self.assertEqual(result["papers"][0]["authors"], ["A. Author"])


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False


class ClientEnvelopeTests(unittest.TestCase):
    def _client(self) -> SciAtlasClient:
        return SciAtlasClient(api_key="test-key", base_url="http://sciatlas.openkg.cn", timeout=5)

    def test_search_papers_builds_documented_request(self) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = {key.lower(): value for key, value in req.header_items()}
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeHttpResponse(
                {"request_id": "r1", "status": "ok", "result": {"papers": [], "total_count": 0}, "meta": None}
            )

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            response = self._client().search_papers(query="graph neural networks", top_k=3)

        self.assertEqual(response["status"], "ok")
        self.assertEqual(captured["url"], "http://sciatlas.openkg.cn/v1/search")
        self.assertEqual(captured["headers"].get("authorization"), "Bearer test-key")
        self.assertEqual(captured["headers"].get("x-api-key"), "test-key")
        plan = captured["payload"]["plan"]
        self.assertEqual(plan["query_text"], "graph neural networks")
        self.assertEqual(plan["source_type"], "idea_text")
        self.assertEqual(plan["keywords"], [{"text": "graph neural networks", "score": 8}])
        self.assertEqual(captured["payload"]["options"]["top_k"], 3)

    def test_related_authors_posts_to_documented_endpoint(self) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeHttpResponse(
                {"request_id": "r2", "status": "ok", "result": {"authors": [], "total_count": 0}, "meta": None}
            )

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            response = self._client().related_authors(query="graph neural networks", top_k=5)

        self.assertEqual(response["status"], "ok")
        self.assertEqual(captured["url"], "http://sciatlas.openkg.cn/v1/authors/related")
        self.assertEqual(captured["payload"]["plan"]["query_text"], "graph neural networks")
        self.assertEqual(captured["payload"]["options"]["top_k"], 5)

    def test_error_status_envelope_raises_with_server_message(self) -> None:
        def fake_urlopen(req, timeout=None):
            return _FakeHttpResponse(
                {
                    "request_id": "r3",
                    "status": "error",
                    "result": None,
                    "meta": None,
                    "message": "invalid plan: keywords required",
                }
            )

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as ctx:
                self._client().search_papers(query="graph neural networks", top_k=3)
        self.assertIn("invalid plan: keywords required", str(ctx.exception))
        self.assertIn("r3", str(ctx.exception))


class KgRubricSearchTests(unittest.TestCase):
    def _config(self, root: Path) -> IdeaRubricConfig:
        return IdeaRubricConfig(
            target_idea="Using GNNs to predict catalyst stability under reaction conditions",
            artifact_root=root,
            output_path=root / "unused.idea_rubric.json",
            sources_root=root,
            llm_api_key="test-key",
            search_top_k=10,
            search_final_k=2,
            rerank_batch_size=2,
        )

    def test_search_maps_reranks_and_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            engine = KgRubricSearch(config, call_llm=lambda **kwargs: {"papers": [
                {"paper_index": 1, "score": 2.0, "reason": "same problem"},
                {"paper_index": 2, "score": 9.5, "reason": "directly relevant"},
            ]})
            with patch_hosted_client():
                docs = engine.search(config.target_idea, top_k=10, final_k=2)

            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["Title"], "Unrelated Vision Transformer Study")
            self.assertEqual(docs[0]["llm_score"], 9.5)
            self.assertEqual(docs[1]["Title"], "Graph Neural Networks for Catalyst Stability Prediction")
            self.assertEqual(docs[1]["paper_id"], "W1")
            self.assertEqual(docs[1]["kg_score"], 0.9)
            self.assertEqual(docs[1]["authors"], ["A. Author"])
            self.assertEqual(docs[1]["source"], "kg_hosted")

    def test_search_falls_back_to_kg_order_on_rerank_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            engine = KgRubricSearch(config, call_llm=lambda **kwargs: None)
            with patch_hosted_client():
                docs = engine.search(config.target_idea, top_k=10, final_k=2)
            self.assertEqual([doc["Title"] for doc in docs][0], "Graph Neural Networks for Catalyst Stability Prediction")
            self.assertNotIn("llm_score", docs[0])

    def test_search_raises_when_server_returns_no_papers(self) -> None:
        empty_envelope = {
            "request_id": "req-empty",
            "status": "ok",
            "result": {"papers": [], "total_count": 0},
            "meta": None,
        }
        with TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            engine = KgRubricSearch(config, call_llm=lambda **kwargs: None)
            with patch_hosted_client(empty_envelope):
                with self.assertRaises(RuntimeError):
                    engine.search(config.target_idea, top_k=10, final_k=2)


def build_config(root: Path, sources_root: Path | None = None) -> IdeaRubricConfig:
    return IdeaRubricConfig(
        target_idea="Using GNNs to predict catalyst stability under reaction conditions",
        artifact_root=root,
        output_path=root / "idea_rubric.json",
        sources_root=sources_root or root,
        llm_api_key="test-key",
        search_top_k=10,
        search_final_k=2,
        max_workers=2,
    )


def fake_llm_dispatch(**kwargs):
    label = kwargs.get("label") or ""
    schema = kwargs.get("response_model_schema")
    if label == "rubric_rerank":
        return {"papers": [
            {"paper_index": 1, "score": 2.0, "reason": "same problem"},
            {"paper_index": 2, "score": 9.5, "reason": "directly relevant"},
        ]}
    if label.startswith("norms:"):
        return {"norms": ["Demand ablation of the GNN encoder.", "Report error bars across seeds."]}
    if label.startswith("dimensions:"):
        return {
            "introduction_standards": [
                {"dimension_name": "catalyst problem framing", "core_reason": "motivation overlap", "required_evidence": ["cited prior work"]}
            ],
            "methods_standards": [
                {"dimension_name": "GNN architecture choice", "core_reason": "method soundness", "required_evidence": ["ablation"]}
            ],
            "results_standards": [
                {"dimension_name": "stability metric definition", "core_reason": "evaluability", "required_evidence": ["metric formula"]}
            ],
            "discussion_standards": [
                {"dimension_name": "reaction condition boundary", "core_reason": "generalizability", "required_evidence": ["failure cases"]}
            ],
        }
    if label == "general_rubric":
        return {
            "Idea_Summary": "Predict catalyst stability with GNNs.",
            "Motivation_Standards": [
                {"dimension_name": "catalyst need", "core_philosophy": "domain need", "required_evidence": "prior gap"}
            ],
            "Method_Standards": [
                {"dimension_name": "message passing design", "core_philosophy": "soundness", "required_evidence": "ablation"}
            ],
            "Result_Standards": [
                {"dimension_name": "stability metric", "core_philosophy": "measurability", "required_evidence": "metric"}
            ],
            "Discussion_Standards": [
                {"dimension_name": "boundary conditions", "core_philosophy": "scope", "required_evidence": "limits"}
            ],
        }
    if label == "detailed_rubric":
        return {
            "Evaluation_Standards": [
                {"dimension_name": "seed variance", "core_reason": "reliability", "required_evidence": "error bars"}
            ]
        }
    if label == "sectioned_rubric_synthesis":
        return {
            "Idea_Summary": "Predict catalyst stability with GNNs.",
            "Idea_Breakdown": {
                "motivation_and_problem": "catalyst stability",
                "proposed_method": "GNN",
                "experiment_and_data": "catalyst dataset",
            },
            "Motivation_Standards": [
                {"dimension_name": "catalyst problem framing", "core_reason": "domain need", "required_evidence": "prior gap", "source_tag": "Detailed", "target_section": "Motivation"}
            ],
            "Method_Standards": [
                {"dimension_name": "GNN architecture choice", "core_reason": "soundness", "required_evidence": "ablation", "source_tag": "Detailed", "target_section": "Method"}
            ],
            "Result_Standards": [
                {"dimension_name": "stability metric definition", "core_reason": "measurability", "required_evidence": "metric", "source_tag": "Detailed", "target_section": "Result"}
            ],
            "Discussion_Standards": [
                {"dimension_name": "reaction condition boundary", "core_reason": "scope", "required_evidence": "limits", "source_tag": "General", "target_section": "Discussion"}
            ],
        }
    raise AssertionError(f"unexpected LLM call label: {label}")


class RubricPipelineTests(unittest.TestCase):
    def test_build_sources_and_run_llm_end_to_end_offline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources_root = root / "rubric_sources"
            llm_root = root / "rubric_llm"
            sources_root.mkdir()
            llm_root.mkdir()

            with patch_hosted_client(), patch.object(
                idea_rubric, "call_llm_json_with_retry", side_effect=fake_llm_dispatch
            ):
                sources_runner = IdeaRubricRunner(build_config(sources_root, sources_root))
                sources_result = sources_runner.build_sources()
                self.assertEqual(sources_result["status"], "ok")
                self.assertEqual(sources_result["retrieved_paper_count"], 2)

                contexts = json.loads((sources_root / "paper_contexts.json").read_text(encoding="utf-8"))
                self.assertEqual(len(contexts["papers"]), 2)
                for item in contexts["papers"]:
                    for key in (
                        "title",
                        "doi",
                        "folder_name",
                        "artifact_dir",
                        "sections_path",
                        "review_text_path",
                        "source_meta_path",
                    ):
                        self.assertIn(key, item)

                for paper_dir in (sources_root / "papers").iterdir():
                    sections = json.loads((paper_dir / "extracted_sections.json").read_text(encoding="utf-8"))
                    self.assertIn("abstract", sections)
                    self.assertIs(sections["full_text_available"], False)
                    self.assertTrue((paper_dir / "review_text.txt").exists())
                    self.assertTrue((paper_dir / "source_meta.json").exists())

                llm_runner = IdeaRubricRunner(build_config(llm_root, sources_root))
                llm_result = llm_runner.run_llm()
                self.assertEqual(llm_result["status"], "ok")

            rubric = json.loads((llm_root / "idea_rubric.json").read_text(encoding="utf-8"))
            self.assertEqual(rubric["schema_version"], idea_rubric.RUBRIC_SCHEMA_VERSION)
            for key in (
                "Motivation_Standards",
                "Method_Standards",
                "Result_Standards",
                "Discussion_Standards",
            ):
                self.assertTrue(rubric[key], f"{key} must not be empty")

            historical_context = (llm_root / "historical_context.txt").read_text(encoding="utf-8")
            self.assertIn("GNN architecture choice", historical_context)

    def test_norms_llm_failure_falls_back_to_personas(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources_root = root / "rubric_sources"
            sources_root.mkdir()

            def failing_norms(**kwargs):
                label = kwargs.get("label") or ""
                if label.startswith("norms:"):
                    return None
                return fake_llm_dispatch(**kwargs)

            with patch_hosted_client(), patch.object(
                idea_rubric, "call_llm_json_with_retry", side_effect=failing_norms
            ):
                runner = IdeaRubricRunner(build_config(sources_root, sources_root))
                result = runner.build_sources()
                self.assertEqual(result["status"], "ok")

            review_texts = list((sources_root / "papers").glob("*/review_text.txt"))
            self.assertTrue(review_texts)
            for path in review_texts:
                content = path.read_text(encoding="utf-8")
                self.assertTrue(content.strip())

    def test_resolve_config_has_no_model_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = IdeaRubricRunner(build_config(root))
            field_names = set(runner.config.__dataclass_fields__)
            self.assertEqual(
                field_names,
                {
                    "target_idea",
                    "artifact_root",
                    "output_path",
                    "sources_root",
                    "env_path",
                    "llm_api_key",
                    "llm_base_url",
                    "llm_model_name",
                    "llm_temperature",
                    "llm_timeout_seconds",
                    "search_top_k",
                    "search_final_k",
                    "max_workers",
                    "rerank_batch_size",
                },
            )
            self.assertEqual(runner.config.search_final_k, 2)
            self.assertEqual(runner.config.llm_api_key, "test-key")


if __name__ == "__main__":
    unittest.main()
