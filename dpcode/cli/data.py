"""`dp-data` — acquire the project's inputs, and package the cheap ones.

    dp-data embeddings                       GATED UNI2-h WSI features (66/34 GB)
    dp-data cnv                              arm-level copy number, both cohorts
    dp-data labels                           the PAM50 / TCGA-CDR label tables
    dp-data headline-artifacts               build the ~0.8 MB reproduction bundle
    dp-data verify-artifacts --bundle DIR    check a downloaded bundle

CPTAC acquisition is deliberately NOT here: it is `dp-cptac phase=0`, because it
is the first phase of an ordered pipeline whose later phases check its outputs
(DESIGN-ADDENDUM A12).

Each subcommand composes one option of the `acquire` config group and dispatches
the existing `tools/` script with explicit flags, so `python tools/<script>.py`
keeps working unchanged and there is exactly one set of defaults per acquisition.

WHY `headline-artifacts` EXISTS. The external half of the headline table needs
four inputs totalling about 758 KB, and needs no GPU, no slide, and no gated
download. Three of them are gitignored with no release mechanism, so today a
stranger cannot run even the cheap path. This command assembles the bundle and a
checksum manifest; publishing it is an author decision (where, and under what
licence) that no script can make.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig

from .. import schema
from ..paths import assert_paths_absolute
from .config import compose_config

__all__ = [
    "BUNDLE_README",
    "bundle_entries",
    "sha256",
    "write_manifest",
    "read_manifest",
    "main",
]

#: Which `acquire` group option each subcommand composes.
CONFIG_OPTION = {
    "embeddings": "embeddings",
    "cnv": "cnv",
    "labels": "labels",
    "headline-artifacts": "headline_artifacts",
    "verify-artifacts": "headline_artifacts",
}

BUNDLE_README = "README.md"


@dataclass(frozen=True)
class BundleFile:
    """One file in the reproduction bundle."""

    #: Path inside the bundle. Also the path it belongs at in a default clone, so
    #: `cp -a <bundle>/. <clone>/` installs the whole thing.
    destination: str
    #: The file on this machine.
    source: Path
    #: Which `paths.*` key owns its directory, for an installation that moved
    #: `.datasets` or `.scratch` elsewhere.
    paths_key: str
    why: str


def bundle_entries(cfg: DictConfig) -> tuple[list[BundleFile], list[str]]:
    """The bundle contents, resolved against this machine. Returns `(files, missing)`.

    The list is code, not config: it is the definition of "what reproduces the
    external table", and every entry names why it is in there. Adding to it is a
    claim about reproducibility, so it should be a reviewable change.
    """
    paths = cfg.paths
    cnv_dir = Path(str(paths.cnv_dir))
    predictions = (
        Path(str(paths.cptac_validation_dir)) / "results" / "predictions"
    )
    labels_dir = Path(str(paths.labels_dir))
    baseline = Path(str(paths.results_root)) / "pam50_final_s1"

    wanted: list[tuple[str, list[Path], str, str]] = [
        (
            ".datasets/cnv/tcga_brca_cna_arm.csv",
            [cnv_dir / "tcga_brca_cna_arm.csv"],
            "paths.cnv_dir",
            "981 x 39 arm-level log2 medians; the CNV arm's TCGA training input.",
        ),
        (
            ".datasets/cnv/cptac_brca_cna_arm.csv",
            [cnv_dir / "cptac_brca_cna_arm.csv"],
            "paths.cnv_dir",
            "114 x 39; the CNV arm's external input.",
        ),
        (
            ".datasets/cnv/reference/gene_arm_hg38.csv",
            [
                labels_dir / "reference" / "gene_arm_hg38.csv",
                Path(str(paths.cnv_reference_dir)) / "gene_arm_hg38.csv",
            ],
            "paths.cnv_reference_dir",
            "Hugo symbol -> chromosome arm. The only pin on the 39-feature space: "
            "UCSC refGene is a live table, so this cannot be re-derived reproducibly.",
        ),
        (
            ".scratch/cptac_validation/results/predictions/ensemble_predictions.csv",
            [predictions / "ensemble_predictions.csv"],
            "paths.cptac_validation_dir",
            "378 slides -> 114 cases, 10-fold ensemble softmax. This 46 KB file "
            "carries the entire 66 GB + 34 GB gated, GPU-trained upstream.",
        ),
        (
            "tools/data/tcga_brca_pam50_labels.csv",
            [labels_dir / "tcga_brca_pam50_labels.csv"],
            "paths.labels_dir",
            "981 PAM50 calls from cBioPortal. Already git-tracked; included so the "
            "bundle stands alone.",
        ),
    ]

    files: list[BundleFile] = []
    missing: list[str] = []
    for destination, candidates, paths_key, why in wanted:
        source = next((c for c in candidates if c.is_file()), None)
        if source is None:
            missing.append(f"{destination} (looked in {', '.join(str(c) for c in candidates)})")
            continue
        files.append(BundleFile(destination, source, paths_key, why))

    fold_results = sorted(baseline.glob("split_*_results.pkl"))
    if not fold_results:
        missing.append(f"{baseline}/split_*_results.pkl")
    for source in fold_results:
        files.append(
            BundleFile(
                f".scratch/results/pam50_final_s1/{source.name}",
                source,
                "paths.results_root",
                "Per-fold out-of-fold WSI probabilities; the internal (TCGA) half of "
                "the comparison, read by tools/evaluate_cnv_wsi_fusion.py --internal.",
            )
        )
    return files, missing


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(target: Path, files: Sequence[BundleFile], digests: Sequence[str]) -> None:
    """Write a `sha256sum -c`-compatible manifest with provenance comments."""
    lines = [
        "# SHA256 of the dp-code headline-artifact bundle.",
        "#",
        "# Verify a downloaded bundle:",
        "#     dp-data verify-artifacts --bundle <dir>",
        "# or, with coreutils only, from inside the bundle directory:",
        "#     sha256sum -c MANIFEST.sha256",
        "#",
        "# These files reproduce the external half of the headline table with no GPU,",
        "# no slide and no gated download. Every one of them is an OUTPUT of a step",
        "# that does need those things; nothing here is re-derivable from this bundle.",
        "#",
    ]
    previous_why = None
    for entry in files:
        size = entry.source.stat().st_size
        lines.append(f"# {entry.destination}")
        lines.append(f"#   {size} bytes | belongs under {entry.paths_key}")
        if entry.why != previous_why:  # the ten fold pickles share one explanation
            lines.append(f"#   {entry.why}")
            previous_why = entry.why
    lines.append("#")
    for entry, digest in zip(files, digests):
        lines.append(f"{digest}  {entry.destination}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, str]:
    """Parse a `sha256sum` manifest into `{relative path: digest}`."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, _, name = stripped.partition("  ")
        if not name:
            digest, _, name = stripped.partition(" ")
        if digest and name:
            entries[name.strip().lstrip("*")] = digest.strip()
    return entries


def _bundle_readme(files: Sequence[BundleFile], total: int) -> str:
    lines = [
        "# dp-code — headline-artifact bundle",
        "",
        f"{len(files)} files, {total / 1024:.0f} KB. Everything needed to reproduce the",
        "external half of the headline table in",
        "`docs/cnv-wsi-fusion-external-validation.md`, with **no GPU, no whole-slide",
        "image and no gated download**.",
        "",
        "## Install",
        "",
        "```bash",
        "cd <your clone of dp-code>",
        "cp -a <this directory>/. .          # the layout already matches a default clone",
        "sha256sum -c --ignore-missing MANIFEST.sha256",
        "```",
        "",
        "If `DP_DATA_ROOT`, `DP_SCRATCH_ROOT` or `DP_RESULTS_ROOT` moved those trees,",
        "copy each file to the directory named by its `paths.*` key in the manifest",
        "instead; `dp-config show` prints where each key resolves on your machine.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -e .",
        "python tools/evaluate_cnv_wsi_fusion.py             # TCGA -> CPTAC external",
        "python tools/evaluate_cnv_wsi_fusion.py --internal  # adds the TCGA head-to-head",
        "```",
        "",
        "Report the CNV-alone arm every time fusion is reported: fusion's edge over CNV",
        "alone is marginal, and omitting it reproduces the selective reporting the",
        "project's own literature survey criticises. The baseline to beat is the",
        "equal-weight probability mean, not the WSI-only model.",
        "",
        "## Contents",
        "",
    ]
    # The ten per-fold pickles share one explanation; listing it ten times buries
    # the four files that differ.
    groups: list[tuple[str, list[str]]] = []
    for entry in files:
        if groups and groups[-1][0] == entry.why:
            groups[-1][1].append(entry.destination)
        else:
            groups.append((entry.why, [entry.destination]))
    for why, destinations in groups:
        if len(destinations) == 1:
            lines.append(f"- `{destinations[0]}` — {why}")
        else:
            lines.append(
                f"- `{destinations[0]}` … `{destinations[-1]}` "
                f"({len(destinations)} files) — {why}"
            )
    lines.append("")
    lines.append("## What this bundle is not")
    lines.append("")
    lines.append(
        "It contains no patient-level imaging, no whole-slide image, and nothing from"
    )
    lines.append(
        "the private institutional cohort. The CPTAC and TCGA-BRCA inputs it derives"
    )
    lines.append("from are public.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_embeddings(cfg: DictConfig, args: argparse.Namespace) -> int:
    node = cfg.acquire
    command = [
        sys.executable,
        str(Path(str(cfg.paths.repo_root)) / "tools" / "download_embeddings.py"),
        "--cohort",
        str(node.cohort),
    ]
    if node.output_dir is not None:
        command += ["--output_dir", str(node.output_dir)]
    if node.expected_h5 is not None:
        command += ["--expected_h5", str(node.expected_h5)]
    if bool(node.download_only):
        command.append("--download_only")
    if bool(node.dry_run):
        command.append("--dry_run")
    return _dispatch(cfg, command, args.dry_run)


def cmd_cnv(cfg: DictConfig, args: argparse.Namespace) -> int:
    node = cfg.acquire
    command = [
        sys.executable,
        str(Path(str(cfg.paths.repo_root)) / "tools" / "download_cnv_mutations.py"),
        "--what",
        str(node.what),
        "--representation",
        str(node.representation),
        "--out",
        str(node.out),
    ]
    if bool(node.validate_arms):
        command.append("--validate-arms")
    if bool(node.all_cases):
        command.append("--all-cases")
    if node.top_mutated is not None:
        command += ["--top-mutated", str(node.top_mutated)]
    return _dispatch(cfg, command, args.dry_run)


def cmd_labels(cfg: DictConfig, args: argparse.Namespace) -> int:
    tools = Path(str(cfg.paths.repo_root)) / "tools"
    scripts = []
    if bool(cfg.acquire.pam50):
        scripts.append(tools / "fetch_pam50_labels.py")
    if bool(cfg.acquire.tcga_cdr):
        scripts.append(tools / "fetch_tcga_labels.py")
    if not scripts:
        print("Nothing selected: set acquire.pam50=true or acquire.tcga_cdr=true.")
        return 1
    print(
        "NOTE: both outputs live in the git-tracked tools/data/. Check `git diff` "
        "afterwards — a changed label table changes every published number.",
    )
    for script in scripts:
        status = _dispatch(cfg, [sys.executable, str(script)], args.dry_run)
        if status != 0:
            return status
    return 0


def cmd_headline_artifacts(cfg: DictConfig, args: argparse.Namespace) -> int:
    node = cfg.acquire
    output_dir = Path(args.output_dir or str(node.output_dir))
    files, missing = bundle_entries(cfg)

    print(f"bundle : {output_dir}")
    for entry in files:
        print(f"  {entry.source}  ->  {entry.destination}")
    if missing:
        print()
        sys.stdout.flush()
        print("Cannot assemble the bundle; these inputs are absent:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        print(
            "\nEvery one of them is an output of a GPU or gated step (see "
            "`dp-cptac`, `dp-train`, `dp-data cnv`). The bundle exists precisely so "
            "that other people do not have to run those.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        total = sum(entry.source.stat().st_size for entry in files)
        print(f"\n--dry-run: {len(files)} files, {total / 1024:.0f} KB, nothing copied.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    digests = []
    total = 0
    for entry in files:
        destination = output_dir / entry.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        # copy2, never move: every source is either gitignored and unrecoverable
        # or a tracked input.
        shutil.copy2(entry.source, destination)
        digests.append(sha256(destination))
        total += destination.stat().st_size

    manifest = output_dir / str(node.manifest_name)
    write_manifest(manifest, files, digests)
    (output_dir / BUNDLE_README).write_text(_bundle_readme(files, total), encoding="utf-8")

    print(f"\nwrote {len(files)} files ({total / 1024:.0f} KB)")
    print(f"      {manifest}")
    print(f"      {output_dir / BUNDLE_README}")
    print(
        "\nTO PUBLISH (an author decision — this command deliberately does not do it):\n"
        f"  1. archive it:   tar -czf headline-artifacts.tar.gz -C {output_dir} .\n"
        "  2. deposit the archive where it can be cited (Zenodo gets a DOI; a GitHub\n"
        "     release does not; institutional storage may not outlive the thesis),\n"
        "  3. commit the manifest so a stranger can verify a download against a\n"
        "     checksum that came with the clone:\n"
        f"       cp {manifest} {node.tracked_manifest}\n"
        "  4. put the download URL in REPRODUCING.md next to `dp-data verify-artifacts`.\n"
        "\nCheck the licence terms of TCGA-BRCA and CPTAC-BRCA before redistributing:\n"
        "these are derived data (arm-level medians and model outputs), not primary\n"
        "patient data, but the decision is yours to make and record."
    )
    return 0


def cmd_verify_artifacts(cfg: DictConfig, args: argparse.Namespace) -> int:
    node = cfg.acquire
    bundle = Path(args.bundle or str(node.output_dir))
    manifest = Path(args.manifest) if args.manifest else Path(str(node.tracked_manifest))

    if not manifest.is_file():
        fallback = bundle / str(node.manifest_name)
        print(
            f"No tracked manifest at {manifest}.\n"
            "That file is written by the publish step of `dp-data headline-artifacts` "
            "(step 3) and is what lets a download be checked against a checksum that "
            "arrived with the clone. It has not been published yet.",
            file=sys.stderr,
        )
        if not fallback.is_file():
            return 1
        print(
            f"Falling back to the bundle's own manifest, {fallback}. This verifies "
            "the bundle is internally consistent, NOT that it is the right bundle.",
            file=sys.stderr,
        )
        manifest = fallback

    expected = read_manifest(manifest)
    if not expected:
        print(f"{manifest} lists no files.", file=sys.stderr)
        return 1

    print(f"bundle   : {bundle}")
    print(f"manifest : {manifest} ({len(expected)} files)")
    failures = 0
    for name, digest in expected.items():
        path = bundle / name
        if not path.is_file():
            print(f"  MISSING  {name}")
            failures += 1
            continue
        actual = sha256(path)
        if actual != digest:
            print(f"  MISMATCH {name}")
            print(f"           expected {digest}")
            print(f"           actual   {actual}")
            failures += 1
        else:
            print(f"  OK       {name}")
    if failures:
        print(f"\n{failures} of {len(expected)} files failed verification.", file=sys.stderr)
        return 1
    print(f"\nall {len(expected)} files verified")
    return 0


def _dispatch(cfg: DictConfig, command: Sequence[str], dry_run: bool) -> int:
    repo = Path(str(cfg.paths.repo_root))
    print("command : " + " ".join(command))
    print(f"cwd     : {repo}")
    sys.stdout.flush()
    if dry_run:
        print("\n--dry-run: nothing dispatched.")
        return 0
    return subprocess.run(list(command), cwd=str(repo)).returncode


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

COMMANDS = {
    "embeddings": cmd_embeddings,
    "cnv": cmd_cnv,
    "labels": cmd_labels,
    "headline-artifacts": cmd_headline_artifacts,
    "verify-artifacts": cmd_verify_artifacts,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(
            name, help=help_text, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        sub.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. acquire.cohort=cptac-brca")
        sub.add_argument("--config", default=None, help="Which `acquire` group option to compose.")
        sub.add_argument(
            "--dry-run", action="store_true", help="Print what would happen, then stop."
        )
        return sub

    add("embeddings", "GATED UNI2-h WSI features for one cohort")
    add("cnv", "arm-level copy number for both cohorts")
    add("labels", "the PAM50 / TCGA-CDR label tables (both are git-tracked)")

    build = add("headline-artifacts", "assemble the ~0.8 MB reproduction bundle")
    build.add_argument("--output_dir", default=None, help="Override acquire.output_dir.")

    verify = add("verify-artifacts", "check a downloaded bundle against a manifest")
    verify.add_argument("--bundle", default=None, help="The bundle directory to check.")
    verify.add_argument(
        "--manifest",
        default=None,
        help="Manifest to check against. Default: acquire.tracked_manifest.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from hydra.errors import HydraException
    from omegaconf.errors import OmegaConfBaseException

    args = build_parser().parse_args(argv)
    option = args.config or CONFIG_OPTION[args.command]
    overrides = list(args.overrides)

    try:
        cfg = compose_config([f"+acquire={option}", *overrides])
        schema.reject_appended_overrides(
            overrides, allow=bool(cfg.run.allow_config_surgery)
        )
        assert_paths_absolute(cfg)
        if not isinstance(cfg.acquire, DictConfig):
            raise RuntimeError(
                f"acquire={option} did not compose a config group; got "
                f"{cfg.acquire!r}. Is dpcode/conf/acquire/{option}.yaml installed?"
            )
        return COMMANDS[args.command](cfg, args)
    except (
        HydraException,
        OmegaConfBaseException,
        ValueError,
        KeyError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
