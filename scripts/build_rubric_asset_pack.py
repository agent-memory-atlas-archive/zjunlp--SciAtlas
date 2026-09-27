#!/usr/bin/env python3
"""Build the portable rubric asset pack from a local NC dataset (server-side, one-off).

The pack replaces the ~290GB paper dataset for external users: it contains the
FAISS index, nc_meta.json, and per-paper precomputed artifacts
(summary.json / dimensions.json / review_text.txt), so the rubric branch runs
without Europe PMC fetches, review PDFs, or per-paper LLM calls.

Layout produced under --output:
  sciatlas_rubric_assets/
    faiss_nc.index
    nc_meta.json
    MANIFEST.json
    papers/<folder_name>/{summary.json, dimensions.json, review_text.txt}
  sciatlas_rubric_assets.tar.gz
  SHA256SUMS

Resumable: existing review_text.txt / copied json files are skipped on rerun.

Example:
  python scripts/build_rubric_asset_pack.py \
      --dataset-root /data1/nc_dataset/merged_nc_dataset \
      --output /data2/<user>/sciatlas_assets --workers 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PIPELINE = ROOT / "review_pipeline"
if str(REVIEW_PIPELINE) not in sys.path:
    sys.path.insert(0, str(REVIEW_PIPELINE))

PACK_DIRNAME = "sciatlas_rubric_assets"
COPY_FILES = ("summary.json", "dimensions.json", "extracted_sections.json")
DEFAULT_DATASET_ROOT = "/data1/nc_dataset/merged_nc_dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="merged_nc_dataset directory.")
    parser.add_argument("--output", required=True, help="Output directory for the staging pack and tarball.")
    parser.add_argument("--pack-name", default=f"{PACK_DIRNAME}.tar.gz", help="Tarball file name.")
    parser.add_argument("--workers", type=int, default=32, help="Parallel review-text extraction workers.")
    parser.add_argument(
        "--include-sections",
        action="store_true",
        help="Also bundle extracted_sections.json (+~440MB; only needed to regenerate summaries via LLM).",
    )
    parser.add_argument("--skip-tar", action="store_true", help="Only build the staging directory.")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_review_text(source_folder: Path, target_path: Path) -> str:
    from review.idea_rubric import _extract_pdf_text, _filter_review_report

    review_pdf = next(source_folder.glob("review*.pdf"), None)
    if review_pdf is None:
        return "missing_review_pdf"
    raw_text = _extract_pdf_text(review_pdf)
    if not raw_text:
        return "pdf_text_extraction_failed"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(_filter_review_report(raw_text), encoding="utf-8")
    return "ok"


def process_paper(entry: dict, dataset_root: Path, papers_dir: Path, include_sections: bool) -> dict:
    folder_name = Path(str(entry.get("folder_path") or "").rstrip("/")).name
    if not folder_name:
        return {"folder": "", "status": "skipped", "reason": "no folder_path"}
    source_folder = dataset_root / folder_name
    target_dir = papers_dir / folder_name
    result: dict = {"folder": folder_name, "status": "ok"}
    if not source_folder.is_dir():
        return {"folder": folder_name, "status": "skipped", "reason": "source folder missing"}

    copy_names = list(COPY_FILES) if include_sections else [n for n in COPY_FILES if n != "extracted_sections.json"]
    copied = []
    for name in copy_names:
        source_file = source_folder / name
        target_file = target_dir / name
        if source_file.is_file() and not target_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target_file)
        if target_file.exists():
            copied.append(name)
    result["copied"] = copied

    review_text_target = target_dir / "review_text.txt"
    if not review_text_target.exists():
        outcome = extract_review_text(source_folder, review_text_target)
        if outcome != "ok":
            result["status"] = "partial"
            result["review_text"] = outcome
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    staging = output_root / PACK_DIRNAME
    papers_dir = staging / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    meta_source = dataset_root / "nc_meta.json"
    index_source = dataset_root / "faiss_nc.index"
    for required in (meta_source, index_source):
        if not required.is_file():
            raise SystemExit(f"Required dataset file missing: {required}")
    for name in ("nc_meta.json", "faiss_nc.index"):
        target = staging / name
        if not target.exists():
            print(f"Copying {name} ...", flush=True)
            shutil.copyfile(dataset_root / name, target)

    metadata = json.loads(meta_source.read_text(encoding="utf-8"))
    print(f"Papers in nc_meta.json: {len(metadata)}", flush=True)

    started_at = time.perf_counter()
    stats = {"ok": 0, "partial": 0, "skipped": 0}
    problems: list[dict] = []
    # Multiple resumable passes with process isolation: a corrupt PDF can crash
    # the native lib inside a worker; that only breaks the pool, and the next
    # pass continues from the files already written.
    max_passes = 4
    for pass_index in range(1, max_passes + 1):
        pass_stats = {"ok": 0, "partial": 0, "skipped": 0}
        pass_problems: list[dict] = []
        done_count = 0
        broken = False
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_paper, entry, dataset_root, papers_dir, args.include_sections): entry
                    for entry in metadata
                }
                for future in as_completed(futures):
                    result = future.result()
                    done_count += 1
                    pass_stats[result["status"]] = pass_stats.get(result["status"], 0) + 1
                    if result["status"] != "ok":
                        pass_problems.append(result)
                    if done_count % 500 == 0:
                        elapsed = time.perf_counter() - started_at
                        print(
                            f"  [pass {pass_index}] {done_count}/{len(metadata)} papers ({elapsed:.0f}s) stats={pass_stats}",
                            flush=True,
                        )
        except BrokenProcessPool as exc:
            broken = True
            print(
                f"  [pass {pass_index}] worker pool broken after {done_count} papers ({exc}); "
                "resuming in a fresh pass",
                flush=True,
            )
        # The last completed pass reflects the final on-disk state; earlier
        # passes may have been cut short by a broken pool.
        if not broken or pass_index == max_passes:
            stats = pass_stats
            problems = pass_problems
        print(f"  [pass {pass_index}] finished stats={pass_stats} broken={broken}", flush=True)
        if not broken:
            break

    manifest = {
        "pack": PACK_DIRNAME,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "paper_count": len(metadata),
        "stats": stats,
        "problems": problems[:200],
        "include_sections": bool(args.include_sections),
        "contents": {
            "faiss_nc.index": {"sha256": sha256_file(staging / "faiss_nc.index")},
            "nc_meta.json": {"sha256": sha256_file(staging / "nc_meta.json")},
        },
        "schema": {
            "summary.json": "PaperSummary (paper_title, abstract_content, introduction_content, methods_content, results_content, discussion_content)",
            "dimensions.json": "ModuleStandards per section (introduction_standards, methods_standards, results_standards, discussion_standards)",
            "review_text.txt": "Filtered peer-review report text extracted from review*.pdf",
        },
    }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Staging complete: {staging} stats={stats}", flush=True)

    if args.skip_tar:
        return 0

    tarball = output_root / args.pack_name
    print(f"Packing {tarball} ...", flush=True)
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(staging, arcname=PACK_DIRNAME)
    checksum = sha256_file(tarball)
    (output_root / "SHA256SUMS").write_text(f"{checksum}  {tarball.name}\n", encoding="utf-8")
    size_mb = tarball.stat().st_size / 1e6
    print(f"Done: {tarball} ({size_mb:.0f} MB) sha256={checksum}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
