#!/usr/bin/env python3
"""Download UNI2-h embeddings for TCGA-BRCA from HuggingFace."""

import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_NAME = "TCGA"
PROJECT_NAME = "TCGA-BRCA"
LOCAL_DIR = Path("/workspace/dp-code/.datasets/embeddings")


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    archive_name = f"TCGA-BRCA_OTHERS.tar.gz"
    remote_path = f"{DATASET_NAME}/{archive_name}"

    print(f"Downloading {remote_path} to {LOCAL_DIR}...")
    # Set HF_TOKEN env var if the dataset requires authentication
    archive_path = hf_hub_download(
        repo_id="MahmoodLab/UNI2-h-features",
        filename=remote_path,
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
    )

    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(LOCAL_DIR)

    print(f"Done. Embeddings extracted to {LOCAL_DIR}")


if __name__ == "__main__":
    main()
