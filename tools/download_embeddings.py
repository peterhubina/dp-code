#!/usr/bin/env python3
"""Download pre-extracted UNI2-h WSI features for one cohort.

The features for BOTH cohorts come from one **gated** HuggingFace dataset,
``MahmoodLab/UNI2-h-features``. Nothing in this repository ever runs the UNI2-h
encoder over a slide: the 66 GB TCGA store and the 34 GB CPTAC store arrive
pre-extracted from that repo, which is why the gate is the single hard
prerequisite of the whole imaging path.

Access is granted per user on the hub. Request it at

    https://huggingface.co/datasets/MahmoodLab/UNI2-h-features

then export ``HF_TOKEN`` (or run ``hf auth login``). A token alone is not
enough — the request has to be approved, and until it is, every download here
fails with :class:`GatedRepoError`.

Where the files land is decided by ``dpcode/conf/paths/default.yaml``
(``paths.tcga_embeddings`` / ``paths.cptac_embeddings``), so ``DP_DATA_ROOT``
moves 100 GB of features off the clone without editing anything. This script used
to hold one machine's clone path as a module-level constant and then
``mkdir(parents=True)`` under it: run as root in a container with a
differently-placed clone, that wrote tens of gigabytes into a directory nothing
else read, without an error. Hence also the post-extraction count assertion
below.

    python tools/download_embeddings.py --cohort tcga-brca
    python tools/download_embeddings.py --cohort cptac-brca --dry_run

or, equivalently, through the config tree:

    dp-data embeddings acquire.cohort=cptac-brca
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

#: The gated dataset repository. Both cohorts live in it.
HF_REPO_ID = "MahmoodLab/UNI2-h-features"
HF_REPO_TYPE = "dataset"
HF_REQUEST_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

#: Environment variables `huggingface_hub` itself honours, in its own order of
#: precedence. Credentials are read HERE, at the call site, and never enter the
#: config tree — a resolved config snapshot is meant to be publishable.
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")

COHORTS = {
    "tcga-brca": {
        "remote_path": "TCGA/TCGA-BRCA_OTHERS.tar.gz",
        "paths_key": "tcga_embeddings",
        # Verified against the store that produced every published number.
        "expected_h5": 1126,
    },
    "cptac-brca": {
        "remote_path": "CPTAC/cptac_brca.tar.gz",
        "paths_key": "cptac_embeddings",
        "expected_h5": 653,
    },
}


def default_local_dir(cohort: str) -> Path:
    """Where `cohort`'s features belong, per `dpcode/conf/paths/default.yaml`."""
    from dpcode.paths import resolve_paths

    return Path(resolve_paths()[COHORTS[cohort]["paths_key"]])


def hf_token() -> str | None:
    """The HuggingFace token from the environment, or `None`.

    `huggingface_hub` would pick these up on its own, but passing `token=`
    explicitly means the error below can state which variable was consulted
    rather than leaving "did it see my token?" unanswerable.

    `None` is not "no credentials": it is `huggingface_hub`'s own sentinel for
    "resolve normally", which still finds a stored `hf auth login`. That
    fallback is deliberate and is reported by :func:`token_report`.
    """
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def token_report() -> str:
    """One line describing which credential will be used — never its value."""
    for name in TOKEN_ENV_VARS:
        if os.environ.get(name):
            return f"${name} (environment)"
    try:
        from huggingface_hub import get_token

        if get_token():
            return "stored login (~/.cache/huggingface/token)"
    except Exception:  # pragma: no cover - hub internals; absence is the answer
        pass
    return "none found"


def gate_message(cohort: str, exc: Exception) -> str:
    return "\n".join(
        [
            f"Cannot read {HF_REPO_ID} ({HF_REPO_TYPE}); {cohort} features are unavailable.",
            f"  underlying error: {type(exc).__name__}: {str(exc).splitlines()[0]}",
            "",
            f"This repository is GATED. Request access at {HF_REQUEST_URL},",
            "wait for approval, then supply a token that carries it:",
            "",
            "    export HF_TOKEN=hf_...        # or: hf auth login",
            "",
            f"  credential in use: {token_report()}"
            f" (checked {', '.join(TOKEN_ENV_VARS)}, then the stored login)",
            "",
            "There is no ungated mirror. Nothing in this repository can rebuild these",
            "features locally either -- the UNI2-h encoder weights are behind a second",
            "gate (MahmoodLab/UNI2-h) and re-tiling both cohorts would take days.",
        ]
    )


def check_access(cohort: str, token: str | None) -> None:
    """Fail before the transfer starts if the gate is closed.

    `HfApi.auth_check` asks the Hub's `/auth-check` endpoint whether this token
    may read the repo. It transfers nothing, and it turns "16 GB in, 401" into a
    one-second refusal.
    """
    try:
        HfApi().auth_check(HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token)
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(gate_message(cohort, exc)) from exc


def safe_members(tar: tarfile.TarFile, destination: Path):
    """Yield members that stay inside `destination`.

    Python 3.10.11 has no `tarfile` extraction filter (PEP 706 landed in
    3.10.12), so the check is explicit. The published archives are flat and
    trigger none of this; it is here so that "download an archive and unpack it
    wherever the config points" is not a path-traversal primitive.
    """
    root = destination.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise SystemExit(
                f"Refusing to extract {member.name!r}: it would write outside {root}."
            )
        if member.issym() or member.islnk():
            link = (target.parent / member.linkname).resolve()
            if not link.is_relative_to(root):
                raise SystemExit(
                    f"Refusing to extract link {member.name!r} -> {member.linkname!r}: "
                    f"it points outside {root}."
                )
        yield member


def count_h5(local_dir: Path) -> int:
    """Feature files directly under `local_dir`.

    Top level only, and `is_file()` filtered: the TCGA store carries a legacy
    `h5_files/` mirror of 1126 dangling symlinks, which a recursive count would
    either double or trip over.
    """
    return sum(1 for path in local_dir.glob("*.h5") if path.is_file())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cohort", choices=sorted(COHORTS), default="tcga-brca")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override the destination. Default: paths.{tcga,cptac}_embeddings "
        "from dpcode/conf/paths/default.yaml.",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Fetch the archive but skip extraction.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Check gate access and report the archive size; transfer nothing.",
    )
    parser.add_argument(
        "--expected_h5",
        type=int,
        default=None,
        help="Assert this many .h5 files after extraction (0 disables). "
        "Default: the cohort's known count.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cohort = COHORTS[args.cohort]
    local_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_local_dir(args.cohort)
    )
    expected = cohort["expected_h5"] if args.expected_h5 is None else args.expected_h5
    token = hf_token()

    print(f"cohort      : {args.cohort}")
    print(f"repo        : {HF_REPO_ID} ({HF_REPO_TYPE}, GATED)")
    print(f"archive     : {cohort['remote_path']}")
    print(f"destination : {local_dir}")
    print(f"credential  : {token_report()}")
    sys.stdout.flush()

    check_access(args.cohort, token)

    if args.dry_run:
        info = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=cohort["remote_path"],
            repo_type=HF_REPO_TYPE,
            local_dir=str(local_dir),
            token=token,
            dry_run=True,
        )
        print(f"dry run: {info}")
        print(f"dry run: nothing written to {local_dir}")
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {cohort['remote_path']} ...", flush=True)
    try:
        archive_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=cohort["remote_path"],
            repo_type=HF_REPO_TYPE,
            local_dir=str(local_dir),
            token=token,
        )
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(gate_message(args.cohort, exc)) from exc
    print(f"archive at {archive_path}", flush=True)

    if args.download_only:
        return 0

    print(f"extracting {archive_path} ...", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(local_dir, members=safe_members(tar, local_dir))

    found = count_h5(local_dir)
    print(f"extracted {found} .h5 files to {local_dir}")
    if expected and found != expected:
        raise SystemExit(
            f"Expected {expected} .h5 files directly under {local_dir}, found {found}. "
            "The archive changed, the extraction was interrupted, or the destination "
            "already held a different cohort. Nothing downstream will match the "
            "published counts until this is resolved; re-run with --expected_h5 0 "
            "to skip this check deliberately."
        )
    print(
        f"done. The {Path(cohort['remote_path']).name} archive is still on disk under "
        f"{local_dir}; it is safe to delete once this count looks right."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
