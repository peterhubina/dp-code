#!/usr/bin/env python3
"""Write the arm-level CNV matrices in the shape CLAM's tabular branch expects.

    python tools/make_cnv_tabular.py

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

Labels follow ``tcga_brca_subtyping``: four classes, Normal-like ignored. CPTAC's 114-case subset
has no Normal-like, so the two cohorts already agree.

The coattn grouping is by chromosome rather than by arm. ``--tabular_group_spec prefix`` would give
39 singleton tokens, which is under the 64-token cap but says nothing biological; one token per
chromosome carries each arm pair together, which is how large-scale copy number is actually read.
"""

import argparse
from pathlib import Path

import pandas as pd

from pam50_arms import REPO, case_of

CLASSES = ["LumA", "LumB", "Basal", "Her2"]          # label_dict order in CLAM's main.py


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / ".scratch/cnv-tabular")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clam = pd.read_csv(REPO / "project/CLAM/dataset_csv/tcga_brca_subtyping.csv")
    tcga_labels = (pd.read_csv(REPO / "tools/data/tcga_brca_pam50_labels.csv")
                   .drop_duplicates("case_id").set_index("case_id")["label"])
    cptac_labels = (pd.read_csv(REPO / ".datasets/cptac-brca/cptac_brca_pam50_dataset.csv")
                    .drop_duplicates("case_id").set_index("case_id")["label_name"])

    print("building CLAM-format CNV tabular inputs")
    tcga = build(REPO / ".datasets/cnv/tcga_brca_cna_arm.csv", tcga_labels, "TCGA",
                 keep_cases=set(clam["case_id"]))
    cptac = build(REPO / ".datasets/cnv/cptac_brca_cna_arm.csv", cptac_labels, "CPTAC")

    for frame, fname in [(tcga, "TCGA_BRCA_CNV_arm_4class_clam.csv"),
                         (cptac, "CPTAC_BRCA_CNV_arm_4class_clam.csv")]:
        path = args.out / fname
        frame.to_csv(path, index=False)
        print(f"  wrote {path.relative_to(REPO)}")

    arms = [c for c in tcga.columns if c not in ("case_id", "label")]
    assert list(arms) == [c for c in cptac.columns if c not in ("case_id", "label")], \
        "cohorts disagree on arm columns"

    groups = {}
    for arm in arms:
        groups.setdefault(f"chr{arm[:-1]}", []).append(arm)
    width = max(len(v) for v in groups.values())
    spec = pd.DataFrame({k: v + [None] * (width - len(v))
                         for k, v in sorted(groups.items(), key=lambda kv: int(kv[0][3:]))})
    spec_path = args.out / "chromosome_groups.csv"
    spec.to_csv(spec_path, index=False)
    print(f"  wrote {spec_path.relative_to(REPO)}  {len(groups)} chromosome tokens "
          f"covering {len(arms)} arms")

    # Coverage check. multimodal_dataset.py raises when a training case has no tabular row, so
    # this decides whether the ladder can reuse the existing splits -- and therefore whether the
    # WSI-only run already on disk is a valid baseline, or has to be retrained on a new case set.
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
    for fold in range(10):
        split = pd.read_csv(REPO / f"project/CLAM/splits/tcga_brca_subtyping_100/splits_{fold}.csv")
        for column in ("train", "val", "test"):
            stale |= {case_of(s) for s in split[column].dropna()} - with_cnv
    if stale:
        print(f"  !! {len(stale)} cases in the existing splits have no CNV: {sorted(stale)[:5]}")
        return 1
    print("  existing splits (tcga_brca_subtyping_100) are fully covered -- the ladder can reuse\n"
          "  them, so pam50_final_s1 is a directly comparable WSI-only baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
