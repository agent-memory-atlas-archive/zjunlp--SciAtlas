#!/usr/bin/env python3
"""Standalone rubric-generation entry point.

Runs only the rubric branch (rubric_sources -> rubric_llm) of the review
pipeline, without the KG / Semantic Scholar / GROBID / reviewer infrastructure.

Prerequisites:
  1. An LLM API key in .env (LLM_API_KEY / DMX-API-KEY / OPENAI_API_KEY).
  2. Embedding/rerank models under <repo>/models/ (auto-detected; see README
     quickstart for download commands).
  3. The precomputed asset pack (python scripts/download_rubric_assets.py),
     or local access to the NC dataset.

Run `python scripts/check_rubric_setup.py` to verify the full setup.

Example:
  python run_rubric.py --idea "Using graph neural networks to predict catalyst stability"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVIEW_PIPELINE = ROOT / "review_pipeline"
DEFAULT_ASSETS_ROOT = ROOT / "assets" / "rubric"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an idea-specific review rubric (rubric branch only).",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea", "--idea-text", dest="idea", default=None, help="Research idea text to evaluate.")
    input_group.add_argument("--idea-file", default=None, help="Path to a UTF-8 file containing the idea text.")
    parser.add_argument("--env", default=str(ROOT / ".env"), help="Path to the .env file with API credentials.")
    parser.add_argument("--output-dir", default=None, help="Run output directory. Default: runs/rubric_<timestamp>.")
    parser.add_argument(
        "--precomputed-root",
        default=None,
        help="Precomputed asset pack root. Default: <repo>/assets/rubric when present.",
    )
    parser.add_argument("--search-top-k", type=int, default=50, help="FAISS recall depth.")
    parser.add_argument("--search-final-k", type=int, default=15, help="Papers kept after reranking.")
    parser.add_argument("--max-workers", type=int, default=8, help="Per-paper processing concurrency.")
    parser.add_argument("--llm-timeout-seconds", type=int, default=300, help="Per-call LLM timeout.")
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Comma-separated CUDA device ids for local mode, e.g. 0,7 (sets INNOEVAL_CUDA_DEVICES).",
    )
    return parser


def _resolve_idea(args: argparse.Namespace) -> str:
    if args.idea:
        return args.idea.strip()
    return Path(args.idea_file).expanduser().read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    idea_text = _resolve_idea(args)
    if not idea_text:
        raise SystemExit("Empty idea text.")

    if args.cuda_devices:
        os.environ["INNOEVAL_CUDA_DEVICES"] = args.cuda_devices
    precomputed_root = args.precomputed_root
    if not precomputed_root and (DEFAULT_ASSETS_ROOT / "papers").is_dir():
        precomputed_root = str(DEFAULT_ASSETS_ROOT)
    if precomputed_root:
        os.environ["RUBRIC_PRECOMPUTED_ROOT"] = str(Path(precomputed_root).expanduser().resolve())

    env_path = Path(args.env).expanduser().resolve()
    if not env_path.exists():
        print(f"[run_rubric] WARNING: env file not found: {env_path}", file=sys.stderr)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT / "runs" / f"rubric_{datetime.now():%Y%m%d_%H%M%S}"
    )
    sources_dir = output_dir / "rubric_sources"
    llm_dir = output_dir / "rubric_llm"
    sources_dir.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REVIEW_PIPELINE))
    from review.idea_rubric import IdeaRubricConfig, run_rubric_llm, run_rubric_sources

    common_kwargs = dict(
        target_idea=idea_text,
        env_path=env_path if env_path.exists() else None,
        search_top_k=args.search_top_k,
        search_final_k=args.search_final_k,
        max_workers=args.max_workers,
        llm_timeout_seconds=args.llm_timeout_seconds,
    )
    result_payload: dict = {"idea_chars": len(idea_text), "output_dir": str(output_dir)}
    try:
        sources_config = IdeaRubricConfig(
            artifact_root=sources_dir,
            sources_root=sources_dir,
            output_path=sources_dir / "unused.idea_rubric.json",
            **common_kwargs,
        )
        sources_result = run_rubric_sources(sources_config)
        result_payload["rubric_sources"] = sources_result

        llm_config = IdeaRubricConfig(
            artifact_root=llm_dir,
            sources_root=sources_dir,
            output_path=llm_dir / "idea_rubric.json",
            **common_kwargs,
        )
        llm_result = run_rubric_llm(llm_config)
        result_payload["rubric_llm"] = llm_result
        result_payload["status"] = llm_result.get("status", "error")
        result_payload["rubric_path"] = llm_result.get("result_path")
    except Exception as exc:
        result_payload["status"] = "error"
        result_payload["error_type"] = exc.__class__.__name__
        result_payload["error"] = str(exc)
        result_payload["traceback"] = traceback.format_exc()

    (output_dir / "result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "input.json").write_text(
        json.dumps({"idea_text": idea_text}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if result_payload.get("status") == "ok":
        print(f"Rubric generated: {result_payload.get('rubric_path')}")
        print(f"Run artifacts: {output_dir}")
        return 0
    print(json.dumps({key: result_payload.get(key) for key in ("status", "error_type", "error")}, ensure_ascii=False, indent=2))
    print(f"See {output_dir / 'result.json'} for details.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
