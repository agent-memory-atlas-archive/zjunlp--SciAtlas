#!/usr/bin/env python3
"""Download the precomputed rubric asset pack.

The pack contains the NC-paper FAISS index, nc_meta.json, and per-paper
precomputed artifacts (summary.json / dimensions.json / review_text.txt), so
the rubric branch runs without the ~290GB paper dataset, Europe PMC fetches,
or per-paper LLM calls.

Default source: HuggingFace dataset (works through HF_ENDPOINT mirrors such as
https://hf-mirror.com). Use --url to point at any mirror (e.g. a GitHub Release
asset).

Example:
  python scripts/download_rubric_assets.py
  HF_ENDPOINT=https://hf-mirror.com python scripts/download_rubric_assets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_REPO = "zjunlp/sciatlas-rubric-assets"
DEFAULT_FILENAME = "sciatlas_rubric_assets.tar.gz"
PACK_DIRNAME = "sciatlas_rubric_assets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=None, help=f"HF dataset repo id. Default: $SCIATLAS_ASSET_REPO or {DEFAULT_ASSET_REPO}.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Asset file name inside the repo.")
    parser.add_argument("--url", default=None, help="Full download URL override (skips HF resolution).")
    parser.add_argument("--dest", default=str(ROOT / "assets" / "rubric"), help="Destination directory for the unpacked assets.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded tarball next to --dest.")
    return parser


def resolve_url(args: argparse.Namespace) -> str:
    if args.url:
        return args.url
    repo = args.repo or os.environ.get("SCIATLAS_ASSET_REPO") or DEFAULT_ASSET_REPO
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{repo}/resolve/main/{args.filename}"


def fetch_sha256_expected(url: str) -> str:
    sums_url = url.rsplit("/", 1)[0] + "/SHA256SUMS"
    try:
        with urllib.request.urlopen(sums_url, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                return parts[0].strip().lower()
    except Exception:
        pass
    return ""


def download(url: str, target: Path) -> str:
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "sciatlas-asset-downloader/1.0"})
    digest = hashlib.sha256()
    tmp_path = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp_path.replace(target)
    return digest.hexdigest()


def extract(tarball: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmp_dir:
        with tarfile.open(tarball, "r:*") as tar:
            try:
                tar.extractall(tmp_dir, filter="data")
            except TypeError:
                tar.extractall(tmp_dir)
        extracted = Path(tmp_dir) / PACK_DIRNAME
        if not extracted.is_dir():
            entries = [entry for entry in Path(tmp_dir).iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                extracted = entries[0]
            else:
                raise RuntimeError(f"Unexpected archive layout under {tmp_dir}")
        if dest.exists():
            backup = dest.with_name(dest.name + f".bak.{datetime.now():%Y%m%d%H%M%S}")
            dest.rename(backup)
            print(f"Existing {dest} moved to {backup}")
        shutil.move(str(extracted), str(dest))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    url = resolve_url(args)
    dest = Path(args.dest).expanduser().resolve()
    archive_path = dest.parent / Path(url.split("?")[0].rsplit("/", 1)[-1] or DEFAULT_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)

    checksum = download(url, archive_path)
    expected = fetch_sha256_expected(url)
    if expected and expected != checksum:
        archive_path.unlink(missing_ok=True)
        raise SystemExit(f"SHA256 mismatch: expected {expected}, got {checksum}")
    if not expected:
        print("WARNING: SHA256SUMS not published next to the asset; skipped verification.", file=sys.stderr)

    extract(archive_path, dest)
    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)

    (dest / ".download_complete.json").write_text(
        json.dumps(
            {
                "url": url,
                "sha256": checksum,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    papers = dest / "papers"
    paper_count = len([item for item in papers.iterdir() if item.is_dir()]) if papers.is_dir() else 0
    print(f"Assets ready at {dest} (papers: {paper_count})")
    print("The rubric branch auto-detects <repo>/assets/rubric; otherwise set RUBRIC_PRECOMPUTED_ROOT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
