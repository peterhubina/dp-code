#!/usr/bin/env python3
"""Write the arm-level CNV matrices in the shape CLAM's tabular branch expects.

    python tools/make_cnv_tabular.py                  # both cohorts (default)
    python tools/make_cnv_tabular.py --cohort tcga    # TCGA only: no CPTAC download needed

``--tabular_csv`` wants ``case_id``, ``label``, then one column per feature -- the same contract the
RNA branch is fed with. This turns ``.datasets/cnv/{tcga,cptac}_brca_cna_arm.csv`` into that shape
so the fusion ladder can swap RNA for CNV with a flag rather than a code change.

Three files land in ``.scratch/cnv-tabular/``:

  TCGA_BRCA_CNV_arm_4class_clam.csv   training/CV input, restricted to cases in CLAM's dataset_csv
  CPTAC_BRCA_CNV_arm_4class_clam.csv  external-validation input
  chromosome_groups.csv               token grouping for ``--fusion_mode coattn``

It then checks coverage and exits non-zero if any case in the existing splits lacks copy number,
because ``multimodal_dataset.py`` raises on a training case with no tabular row. Coverage is in
fact complete -- all 910 non-Normal cases have CNV -- which is what lets the ladder reuse
``splits/tcga_brca_subtyping_100`` and treat the existing ``pam50_final_s1`` WSI-only run as a
directly comparable baseline instead of retraining it.

``--cohort`` exists because the CPTAC table sits behind the whole gated CPTAC acquisition chain,
while the TCGA table needs only ``.datasets/cnv/`` and two tracked CSVs. Building the TCGA side
used to require the CPTAC manifest to be present, which blocked anyone reproducing the internal
half of the thesis before finishing the external half. The default is still ``both``, so the
documented invocation behaves exactly as before.

Labels follow ``tcga_brca_subtyping``: four classes, Normal-like ignored. CPTAC's 114-case subset
has no Normal-like, so the two cohorts already agree.

The coattn grouping is by chromosome rather than by arm. ``--tabular_group_spec prefix`` would give
39 singleton tokens, which is under the 64-token cap but says nothing biological; one token per
chromosome carries each arm pair together, which is how large-scale copy number is actually read.
"""

import argparse
from pathlib import Path

import pandas as pd

# `tools.pam50_arms` rather than a bare `pam50_arms` — see the note in
# evaluate_cnv_wsi_fusion.py. The direct `python tools/make_cnv_tabular.py` invocation is
# unchanged.
try:
    from tools.pam50_arms import (CLAM_DATASET_CSV, CLAM_SPLITS, CNV_TABULAR_DIR, CPTAC_ARMS,
                                  CPTAC_LABELS, REPO, TCGA_ARMS, TCGA_LABELS, case_of)
except ModuleNotFoundError as exc:  # pragma: no cover - install error, not a code path
    raise ModuleNotFoundError(
        f"{exc}. Run `pip install -e .` from the repository root once; after that this script "
        "runs from any working directory."
    ) from exc

CLASSES = ["LumA", "LumB", "Basal", "Her2"]          # label_dict order in CLAM's main.py

#: Folds in ``splits/tcga_brca_subtyping_100``. Fixed by the split set on disk, not a choice made
#: here -- the coverage check has to read every fold the ladder will train on.
N_SPLIT_FOLDS = 10

COHORT_FILES = {
    "TCGA": "TCGA_BRCA_CNV_arm_4class_clam.csv",
    "CPTAC": "CPTAC_BRCA_CNV_arm_4class_clam.csv",
}


def display(path: Path) -> str:
    """Repo-relative when the output is inside the repo, absolute otherwise.

    ``--out`` may legitimately point outside the repository (a scratch directory, a test's
    ``tmp_path``); the previous unconditional ``relative_to(REPO)`` raised ``ValueError`` after
    the files had already been written.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def build(arms_path, labels, name, keep_cases=None):
    arms = pd.read_csv(arms_path, index_col=0)
    arms.index.name = "case_id"
    labelled = arms.index.intersection(labels.index)
    out = arms.loc[labelled].copy()
    out.insert(0, "label", labels.loc[labelled])
    out = out[out["label"].isin(CLASSES)]
    if keep_cases is not None:
        before = len(out)
        out = out[out.index.isin(keep_cases)]
        print(f"  {name}: {before} labelled -> {len(out)} also present in CLAM's dataset_csv")
    dropped = len(arms) - len(out)
    print(f"  {name}: {len(out)} cases x {arms.shape[1]} arms "
          f"({dropped} dropped: unlabelled, Normal-like, or no slides)")
    print(f"    {out['label'].value_counts().reindex(CLASSES).to_dict()}")
    return out.reset_index()


def coverage_check(tcga: pd.DataFrame, clam: pd.DataFrame) -> int:
    """Would the ladder be able to train on the existing splits? Non-zero if not.

    ``multimodal_dataset.py`` raises when a training case has no tabular row, so this decides
    whether the ladder can reuse the existing splits -- and therefore whether the WSI-only run
    already on disk is a valid baseline, or has to be retrained on a new case set.
    """
    with_cnv = set(tcga["case_id"])
    non_normal = set(clam.loc[clam["label"] != "Normal", "case_id"])
    missing = sorted(non_normal - with_cnv)
    print(f"\ncoverage: {len(non_normal - set(missing))}/{len(non_normal)} non-Normal CLAM cases "
          f"have CNV")
    if missing:
        print(f"  !! {len(missing)} cases would make training raise, e.g. {missing[:5]}")
        print("  !! the ladder needs a restricted dataset_csv and fresh splits; the existing "
              "WSI-only run is NOT a valid baseline")
        return 1

    stale = set()
    for fold in range(N_SPLIT_FOLDS):
        split = pd.read_csv(CLAM_SPLITS / f"splits_{fold}.csv")
        for column in ("train", "val", "test"):
            stale |= {case_of(s) for s in split[column].dropna()} - with_cnv
    if stale:
        print(f"  !! {len(stale)} cases in the existing splits have no CNV: {sorted(stale)[:5]}")
        return 1
    print("  existing splits (tcga_brca_subtyping_100) are fully covered -- the ladder can reuse\n"
          "  them, so pam50_final_s1 is a directly comparable WSI-only baseline")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=["tcga", "cptac", "both"], default="both",
                    help="which tables to write; 'tcga' needs no CPTAC data at all "
                         "(default: %(default)s)")
    ap.add_argument("--out", type=Path, default=CNV_TABULAR_DIR)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    want = {"TCGA": args.cohort in ("tcga", "both"), "CPTAC": args.cohort in ("cptac", "both")}

    print("building CLAM-format CNV tabular inputs")
    frames, clam = {}, None
    if want["TCGA"]:
        clam = pd.read_csv(CLAM_DATASET_CSV)
        tcga_labels = (pd.read_csv(TCGA_LABELS)
                       .drop_duplicates("case_id").set_index("case_id")["label"])
        frames["TCGA"] = build(TCGA_ARMS, tcga_labels, "TCGA", keep_cases=set(clam["case_id"]))
    if want["CPTAC"]:
        cptac_labels = (pd.read_csv(CPTAC_LABELS)
                        .drop_duplicates("case_id").set_index("case_id")["label_name"])
        frames["CPTAC"] = build(CPTAC_ARMS, cptac_labels, "CPTAC")

    for name, fname in COHORT_FILES.items():
        if name in frames:
            path = args.out / fname
            frames[name].to_csv(path, index=False)
            print(f"  wrote {display(path)}")

    columns = {name: [c for c in f.columns if c not in ("case_id", "label")]
               for name, f in frames.items()}
    if len(columns) > 1:
        assert len(set(map(tuple, columns.values()))) == 1, "cohorts disagree on arm columns"
    arms = next(iter(columns.values()))

    groups = {}
    for arm in arms:
        groups.setdefault(f"chr{arm[:-1]}", []).append(arm)
    width = max(len(v) for v in groups.values())
    spec = pd.DataFrame({k: v + [None] * (width - len(v))
                         for k, v in sorted(groups.items(), key=lambda kv: int(kv[0][3:]))})
    spec_path = args.out / "chromosome_groups.csv"
    spec.to_csv(spec_path, index=False)
    print(f"  wrote {display(spec_path)}  {len(groups)} chromosome tokens "
          f"covering {len(arms)} arms")

    if not want["TCGA"]:
        print("\ncoverage check skipped: it is a statement about the TCGA training splits, "
              "so it needs --cohort tcga or both")
        return 0
    return coverage_check(frames["TCGA"], clam)


if __name__ == "__main__":
    raise SystemExit(main())
