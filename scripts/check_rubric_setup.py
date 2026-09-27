#!/usr/bin/env python3
"""Pre-flight check for the rubric branch (run_rubric.py / idea-evaluate).

Verifies dependencies, local embedding/rerank models, the precomputed asset
pack, and LLM credentials — and prints the exact commands to fix whatever is
missing. Exit code 0 means ready to run.

Usage:
  python scripts/check_rubric_setup.py [--env .env]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets" / "rubric"
EMBED_DIRNAME = "bge-large-en-v1.5"
RERANK_DIRNAME = "bge-reranker-large"
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
SERVER_EMBED_DEFAULT = "/data1/bge-model/AI-ModelScope/bge-large-en-v1.5"
SERVER_RERANK_DEFAULT = "/data1/bge-reranker-large"

REQUIRED_MODULES = (
    "faiss",
    "numpy",
    "pydantic",
    "requests",
    "openai",
    "json_repair",
    "torch",
    "sentence_transformers",
)

DOWNLOAD_CMDS = f"""  export HF_ENDPOINT=https://hf-mirror.com   # mainland-China mirror; omit when huggingface.co is reachable
  pip install -U "huggingface_hub[cli]"
  huggingface-cli download BAAI/{EMBED_DIRNAME} --local-dir <repo>/models/{EMBED_DIRNAME}
  huggingface-cli download BAAI/{RERANK_DIRNAME} --local-dir <repo>/models/{RERANK_DIRNAME}
# ModelScope alternative (mainland China):
#   pip install modelscope
#   modelscope download --model AI-ModelScope/{EMBED_DIRNAME} --local_dir <repo>/models/{EMBED_DIRNAME}
#   modelscope download --model BAAI/{RERANK_DIRNAME} --local_dir <repo>/models/{RERANK_DIRNAME}"""


def load_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_or(*names: str) -> str:
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.fixes: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK]      {message}")

    def missing(self, message: str, fix: str) -> None:
        print(f"[MISSING] {message}")
        self.problems.append(message)
        if fix not in self.fixes:
            self.fixes.append(fix)


def resolve_model_dir(env_names: tuple[str, ...], dirname: str, server_default: str) -> Path | None:
    pinned = env_or(*env_names)
    candidates = []
    if pinned:
        candidates.append(Path(pinned).expanduser())
    candidates.append(MODELS_DIR / dirname)
    candidates.append(Path(server_default))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def check_model(report: Report, kind: str, env_names: tuple[str, ...], dirname: str, server_default: str) -> None:
    model_dir = resolve_model_dir(env_names, dirname, server_default)
    if model_dir is None:
        report.missing(
            f"{kind} model not found (looked in {'/'.join(env_names)}, <repo>/models/{dirname})",
            f"Download the models into <repo>/models/ :\n{DOWNLOAD_CMDS}",
        )
        return
    config_file = model_dir / "config.json"
    weights = [name for name in WEIGHT_NAMES if (model_dir / name).is_file()]
    if not config_file.is_file() or not weights:
        report.missing(
            f"{kind} model at {model_dir} is incomplete (config.json: {config_file.is_file()}, weights: {weights})",
            f"Re-download the {kind} model:\n{DOWNLOAD_CMDS}",
        )
        return
    report.ok(f"{kind} model: {model_dir}")


def check_dependencies(report: Report) -> None:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        report.missing(
            f"Python packages missing: {', '.join(missing)}",
            "  pip install -r requirements-rubric.txt",
        )
    else:
        report.ok(f"python dependencies ({len(REQUIRED_MODULES)} modules)")


def check_assets(report: Report) -> None:
    pinned = env_or("RUBRIC_PRECOMPUTED_ROOT")
    assets = Path(pinned).expanduser() if pinned else ASSETS_DIR
    index_file = assets / "faiss_nc.index"
    meta_file = assets / "nc_meta.json"
    papers_dir = assets / "papers"
    if index_file.is_file() and meta_file.is_file() and papers_dir.is_dir():
        paper_count = sum(1 for item in papers_dir.iterdir() if item.is_dir())
        report.ok(f"asset pack: {assets} (papers: {paper_count})")
        return
    report.missing(
        f"precomputed asset pack not found at {assets}",
        "  python scripts/download_rubric_assets.py",
    )


def check_llm_key(report: Report, env_path: Path) -> None:
    env_values = load_env_values(env_path)
    key_names = ("LLM_API_KEY", "DMX-API-KEY", "DMX_API_KEY", "OPENAI_API_KEY")
    found = env_or(*key_names) or next((env_values[name] for name in key_names if env_values.get(name)), "")
    if found:
        report.ok(f"LLM API key configured (via {'env file' if not env_or(*key_names) else 'environment'})")
        return
    report.missing(
        f"no LLM API key found (checked environment and {env_path})",
        f"  cp .env.example {env_path.name}   # then fill in LLM_API_KEY (or DMX-API-KEY / OPENAI_API_KEY)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=str(ROOT / ".env"), help="Path to the .env file. Default: <repo>/.env")
    args = parser.parse_args(argv)

    print("SciAtlas rubric setup check")
    print(f"  repo: {ROOT}")
    report = Report()
    check_dependencies(report)
    check_model(report, "embedding", ("RUBRIC_EMBED_MODEL_PATH", "INNOEVAL_EMBEDDING_MODEL_PATH", "SCIATLAS_EMBEDDING_MODEL_PATH"), EMBED_DIRNAME, SERVER_EMBED_DEFAULT)
    check_model(report, "reranker", ("RUBRIC_RERANK_MODEL_PATH", "INNOEVAL_RERANKER_MODEL_PATH", "SCIATLAS_RERANKER_MODEL_PATH"), RERANK_DIRNAME, SERVER_RERANK_DEFAULT)
    check_assets(report)
    check_llm_key(report, Path(args.env).expanduser().resolve())

    if not report.problems:
        print("\nAll checks passed. Run:\n  python run_rubric.py --idea \"your research idea\"")
        return 0
    print(f"\n{len(report.problems)} problem(s) found. Fix with:\n")
    for fix in report.fixes:
        print(fix)
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
