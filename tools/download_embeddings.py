#!/usr/bin/env python3
"""Download UNI2-h embeddings from HuggingFace (MahmoodLab/UNI2-h-features)."""

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path("/workspace/dp-code")

COHORTS = {
    "tcga-brca": {
        "remote_path": "TCGA/TCGA-BRCA_OTHERS.tar.gz",
        "local_dir": ROOT / ".datasets/tcga-brca/embeddings",
    },
    "cptac-brca": {
        "remote_path": "CPTAC/cptac_brca.tar.gz",
        "local_dir": ROOT / ".datasets/cptac-brca/embeddings",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=sorted(COHORTS), default="tcga-brca")
    parser.add_argument("--download_only", action="store_true",
                        help="Fetch the archive but skip extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    cohort = COHORTS[args.cohort]
    local_dir = cohort["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {cohort['remote_path']} to {local_dir}...", flush=True)
    # Set HF_TOKEN env var if the dataset requires authentication
    archive_path = hf_hub_download(
        repo_id="MahmoodLab/UNI2-h-features",
        filename=cohort["remote_path"],
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"Archive at {archive_path}", flush=True)

    if args.download_only:
        return

    print(f"Extracting {archive_path}...", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(local_dir)

    print(f"Done. Embeddings extracted to {local_dir}", flush=True)


if __name__ == "__main__":
    main()
