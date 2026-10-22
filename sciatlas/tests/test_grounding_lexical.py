from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sciatlas.evidence import grounding


def _make_paragraph(text: str, index: int = 0, paper_dir: str = "paper_1") -> grounding.ParagraphRecord:
    return grounding.ParagraphRecord(
        paragraph_id=f"p{index:04d}",
        paper_rank=1,
        paper_title="Test Paper",
        paper_dir=paper_dir,
        section_path=["Introduction"],
        paragraph_index=index,
        text=text,
    )


class TestLexicalSearch(unittest.TestCase):
    def test_ranks_relevant_paragraph_first_and_keeps_dense_score_key(self):
        paragraphs = [
            _make_paragraph("We propose a graph neural network architecture for molecular property prediction.", 0),
            _make_paragraph("Cooking recipes require careful temperature control over several hours of preparation.", 1),
            _make_paragraph("The transformer attention mechanism scales linearly with sequence length.", 2),
        ]
        results = grounding.lexical_search("graph neural network molecular property prediction", paragraphs, top_k=3)

        self.assertTrue(results)
        self.assertEqual(results[0]["paragraph_index"], 0)
        self.assertEqual(set(results[0].keys()), {"paragraph_index", "dense_score"})
        self.assertGreater(results[0]["dense_score"], 0)

    def test_respects_top_k_and_skips_zero_scores(self):
        paragraphs = [
            _make_paragraph("graph neural network molecular property prediction research", 0),
            _make_paragraph("unrelated text about cooking temperatures and kitchen tools for chefs", 1),
        ]
        results = grounding.lexical_search("graph neural network", paragraphs, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["paragraph_index"], 0)

        empty = grounding.lexical_search("quantum entanglement", paragraphs, top_k=2)
        self.assertEqual(empty, [])

    def test_empty_query_or_corpus_returns_empty(self):
        self.assertEqual(grounding.lexical_search("", [_make_paragraph("some text")], top_k=2), [])
        self.assertEqual(grounding.lexical_search("some query", [], top_k=2), [])

    def test_repeated_terms_score_higher_than_single_occurrence(self):
        paragraphs = [
            _make_paragraph("diffusion models generate images. diffusion models sample noise. diffusion models are popular.", 0),
            _make_paragraph("diffusion appears once in this paragraph about physics.", 1),
        ]
        results = grounding.lexical_search("diffusion models", paragraphs, top_k=2)
        self.assertEqual(results[0]["paragraph_index"], 0)
        self.assertGreater(results[0]["dense_score"], results[1]["dense_score"])


class TestParseRerankerScores(unittest.TestCase):
    def test_parses_json_object_from_wrapped_content(self):
        content = "Here are the scores.\n{\"scores\":[{\"paragraph_index\":1,\"score\":8.5},{\"paragraph_index\":2,\"score\":3.0}]}"
        scores = grounding._parse_reranker_llm_scores(content, expected_count=2)
        self.assertEqual(scores, [8.5, 3.0])

    def test_missing_scores_array_raises(self):
        with self.assertRaises(grounding.GroundingError):
            grounding._parse_reranker_llm_scores('{"other": []}', expected_count=2)

    def test_out_of_range_and_invalid_entries_are_ignored(self):
        content = json.dumps(
            {
                "scores": [
                    {"paragraph_index": 0, "score": 9.9},
                    {"paragraph_index": 1, "score": "not-a-number"},
                    {"paragraph_index": 5, "score": 7.7},
                    {"paragraph_index": 2, "score": 4.2},
                ]
            }
        )
        scores = grounding._parse_reranker_llm_scores(content, expected_count=2)
        self.assertEqual(scores, [0.0, 4.2])


class _FakeGroundingLlmClient:
    model_name = "fake-grounding-model"
    provider = "openai_compatible"

    def __init__(self, config: grounding.QueryGeneratorConfig | None = None) -> None:
        self.config = config
        self.request_count = 0

    def _request_content(self, *, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        self.request_count += 1
        if system_prompt == grounding.RERANKER_LLM_SYSTEM_PROMPT:
            lines = [line for line in user_prompt.splitlines() if line.startswith("[") and "]" in line]
            scores = []
            for number, line in enumerate(lines, start=1):
                marker = "HIGHRELEVANCE" in line
                scores.append({"paragraph_index": number, "score": 9.5 if marker else 1.5})
            return json.dumps({"scores": scores})
        raise AssertionError("unexpected LLM call in fake client")


class TestLlmParagraphReranker(unittest.TestCase):
    def test_scores_batches_and_marks_relevant_paragraphs(self):
        client = _FakeGroundingLlmClient()
        reranker = grounding.LlmParagraphReranker(client, batch_size=2)
        texts = [
            "HIGHRELEVANCE paragraph about graph neural networks.",
            "Unrelated cooking paragraph.",
            "Another unrelated paragraph about sports.",
            "HIGHRELEVANCE paragraph about molecular property prediction.",
        ]
        scores = reranker.score("graph neural networks", texts)

        self.assertEqual(scores, [9.5, 1.5, 1.5, 9.5])
        self.assertEqual(client.request_count, 2)

    def test_raises_grounding_error_when_llm_fails(self):
        class _FailingClient(_FakeGroundingLlmClient):
            def _request_content(self, **kwargs):
                raise grounding.GroundingError("boom")

        reranker = grounding.LlmParagraphReranker(_FailingClient(), batch_size=4)
        with self.assertRaises(grounding.GroundingError):
            reranker.score("query", ["paragraph one text", "paragraph two text"])


def _write_fake_corpus(root: Path) -> Path:
    papers_root = root / "corpus"
    papers_root.mkdir(parents=True)

    paper_1_dir = papers_root / "paper_1"
    paper_1_dir.mkdir()
    (paper_1_dir / "parsed.json").write_text(
        json.dumps(
            {
                "title": "Graph Neural Networks for Molecules",
                "abstract": "We study GNNs for molecular property prediction.",
                "body": [
                    {
                        "heading": "Introduction",
                        "paragraphs": [
                            "Graph neural networks encode molecular structures directly and enable property prediction with strong inductive bias. HIGHRELEVANCE marker included.",
                            "Cooking recipes need careful temperature control and many hours of preparation time for chefs.",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    paper_2_dir = papers_root / "paper_2"
    paper_2_dir.mkdir()
    (paper_2_dir / "parsed.json").write_text(
        json.dumps(
            {
                "title": "Unrelated Cooking Science",
                "abstract": "A study of kitchen workflows.",
                "body": [
                    {
                        "heading": "Introduction",
                        "paragraphs": [
                            "Kitchen workflows optimize timing across many cooking stations in professional restaurant environments.",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = papers_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "status": "ok",
                        "rank": 1,
                        "title": "Graph Neural Networks for Molecules",
                        "paper_dir": "paper_1",
                        "tei": {"parsed_json_path": "paper_1/parsed.json"},
                    },
                    {
                        "status": "ok",
                        "rank": 2,
                        "title": "Unrelated Cooking Science",
                        "paper_dir": "paper_2",
                        "tei": {"parsed_json_path": "paper_2/parsed.json"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


class TestRunGroundingOffline(unittest.TestCase):
    def _build_args(self, root: Path, manifest_path: Path, *, disable_reranker: bool = False):
        cli_args = [
            "--idea-text",
            "We propose using graph neural networks to predict molecular properties with improved inductive bias.",
            "--manifest",
            str(manifest_path),
            "--target-dir",
            str(root / "target"),
            "--output",
            str(root / "result.json"),
            "--disable-experiment-grounding",
            "--dense-candidate-k",
            "5",
            "--reranker-top-n",
            "5",
            "--final-top-k",
            "2",
            "--env",
            str(root / "nonexistent.env"),
        ]
        if disable_reranker:
            cli_args.append("--disable-reranker")
        return grounding.build_parser().parse_args(cli_args)

    def _patch_pipeline(self, reranker_client_cls=_FakeGroundingLlmClient):
        queries = [
            grounding.AtomicQuery(
                query_id="q1",
                section="method",
                sentence="graph neural networks for molecules",
                query_text="graph neural networks molecular property prediction",
            )
        ]
        extraction = grounding.StructuredExtraction(
            basic_idea=["GNN for molecular property prediction"],
            motivation=[],
            method=["graph neural network encoder"],
            experimental_focus=[],
        )
        meta = {
            "status": "ok",
            "error": None,
            "extraction": extraction.to_dict(),
            "strategy": "llm",
            "fallback_used": False,
        }
        return (
            mock.patch.object(
                grounding,
                "generate_queries",
                lambda idea_text, *, args: (queries, extraction, meta),
            ),
            mock.patch.object(grounding, "GroundingLlmClient", reranker_client_cls),
        )

    def test_lexical_prefilter_then_llm_rerank_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fake_corpus(root)
            args = self._build_args(root, manifest_path)
            patch_queries, patch_client = self._patch_pipeline()
            with patch_queries, patch_client:
                payload = grounding.run_grounding(args)

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["corpus"]["retrieval_method"], "lexical_tfidf")
            self.assertEqual(payload["corpus"]["paragraph_count"], 3)
            self.assertFalse(payload["corpus"]["cache_hit"])
            self.assertNotIn("embedding_model", payload["corpus"])

            retrieval = payload["retrieval"]
            self.assertEqual(retrieval["reranker"], "llm")
            self.assertEqual(retrieval["reranker_llm_model"], "fake-grounding-model")
            self.assertEqual(retrieval["reranker_status"], "ok")

            results = retrieval["results"]
            self.assertEqual(len(results), 1)
            matches = results[0]["matches"]
            self.assertTrue(matches)
            best = matches[0]
            self.assertIn("graph neural networks", best["text"].lower())
            self.assertEqual(best["rerank_score"], 9.5)
            self.assertGreater(best["dense_score"], 0)

    def test_reranker_failure_falls_back_to_lexical_order(self):
        class _FailingClient(_FakeGroundingLlmClient):
            def _request_content(self, **kwargs):
                raise grounding.GroundingError("rerank LLM unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fake_corpus(root)
            args = self._build_args(root, manifest_path)
            patch_queries, patch_client = self._patch_pipeline(_FailingClient)
            with patch_queries, patch_client:
                payload = grounding.run_grounding(args)

            retrieval = payload["retrieval"]
            self.assertEqual(retrieval["reranker_status"], "partial_fallback")
            self.assertTrue(retrieval["reranker_error"])
            matches = retrieval["results"][0]["matches"]
            self.assertTrue(matches)
            self.assertTrue(all(match.get("rerank_score") is None for match in matches))

    def test_disable_reranker_uses_lexical_only(self):
        class _NoCallClient(_FakeGroundingLlmClient):
            def _request_content(self, **kwargs):
                raise AssertionError("no LLM call expected when reranker is disabled")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fake_corpus(root)
            args = self._build_args(root, manifest_path, disable_reranker=True)
            patch_queries, patch_client = self._patch_pipeline(_NoCallClient)
            with patch_queries, patch_client:
                payload = grounding.run_grounding(args)

            retrieval = payload["retrieval"]
            self.assertIsNone(retrieval["reranker"])
            self.assertEqual(retrieval["reranker_status"], "disabled")
            matches = retrieval["results"][0]["matches"]
            self.assertTrue(matches)
            self.assertTrue(all(match.get("rerank_score") is None for match in matches))

    def test_corpus_cache_round_trip_without_embeddings(self):
        class _CountingClient(_FakeGroundingLlmClient):
            instances = 0

            def __init__(self, config=None):
                super().__init__(config)
                _CountingClient.instances += 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fake_corpus(root)
            args = self._build_args(root, manifest_path)
            patch_queries, patch_client = self._patch_pipeline(_CountingClient)
            _CountingClient.instances = 0

            with patch_queries, patch_client:
                first = grounding.run_grounding(args)
                second = grounding.run_grounding(args)

            self.assertFalse(first["corpus"]["cache_hit"])
            self.assertTrue(second["corpus"]["cache_hit"])
            self.assertEqual(second["corpus"]["paragraph_count"], first["corpus"]["paragraph_count"])

            cache_files = list((root / "target").glob("corpus_*"))
            self.assertTrue(cache_files)
            self.assertTrue(all(path.suffix == ".json" for path in cache_files))

    def test_parser_no_longer_exposes_model_flags(self):
        parser = grounding.build_parser()
        help_text = parser.format_help()
        for removed_flag in ("--embedding-model", "--reranker-model", "--query-prefix", "--device"):
            self.assertNotIn(removed_flag, help_text)
        for kept_flag in ("--dense-candidate-k", "--reranker-top-n", "--disable-reranker"):
            self.assertIn(kept_flag, help_text)


if __name__ == "__main__":
    unittest.main()
