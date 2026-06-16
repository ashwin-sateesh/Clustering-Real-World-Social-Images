#!/usr/bin/env python3
"""Download and prepare the congressional Instagram dataset.

Usage:
    python scripts/prepare_data.py --output-dir ./data

Downloads the top20account.zip from Dropbox, extracts per-account
archives from both Democrat and Republican directories, and flattens
all images into a single directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://www.dropbox.com/scl/fi/7omlfrp7dbdhatow0f98r/"
    "top20account.zip?rlkey=oryn4ulr514vlir8ih6ld27ih&dl=1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare dataset")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if zip already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "top20account.zip"
    extract_dir = output_dir / "top20account_raw"
    all_dir = output_dir / "all"

    # 1. Download
    if not args.skip_download or not zip_path.exists():
        print("Downloading dataset...")
        subprocess.run(["wget", "-O", str(zip_path), DATASET_URL], check=True)
    else:
        print(f"Using existing zip: {zip_path}")

    # 2. Extract main zip
    print("Extracting main archive...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # 3. Find D/ and R/ directories (handles nested top20account/top20account/ structure)
    party_dirs = []
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if d in ("D", "R"):
                party_dirs.append(Path(root) / d)

    if not party_dirs:
        print("Error: Could not find D/ and R/ directories in the archive.")
        sys.exit(1)

    # 4. Extract per-account archives and flatten
    print("Extracting per-account archives and flattening...")
    all_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_temp_accounts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for party_dir in party_dirs:
        for archive in party_dir.iterdir():
            if archive.suffix in (".zip", ".tar", ".gz", ".rar"):
                try:
                    shutil.unpack_archive(str(archive), str(temp_dir))
                except Exception as e:
                    print(f"  Warning: Could not extract {archive.name}: {e}")

    # Move all images to the flat directory
    count = 0
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                src = Path(root) / f
                dst = all_dir / f
                if dst.exists():
                    dst = all_dir / f"{Path(root).name}_{f}"
                shutil.move(str(src), str(dst))
                count += 1

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"\nDone! {count} images saved to {all_dir}")
    print(f"You can now run: python scripts/run_pipeline.py --images-dir {all_dir}")


if __name__ == "__main__":
    main()
