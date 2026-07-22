#!/usr/bin/env python
"""Site-aware cross-validation splits for the binary ER-status CLAM task.

Test folds hold out whole TCGA tissue-submitting SITES (barcode chars 6-7, i.e.
``case_id.split('-')[1]``) so that no site leaks across the train/test boundary
-- the Howard-2021 site-holdout protocol -- while every fold stays ER-balanced.
All slides of a case always travel together.

TCGA-BRCA is dominated by a few large sites (BH=138, A2=100, ...) over a long
tail of single-case sites, so an off-the-shelf StratifiedGroupKFold yields wildly
uneven, sometimes single-class folds. Instead we pack whole sites into k
size-balanced folds greedily (largest site to the least-loaded fold), which keeps
every fold both-class and near-equal in size. The validation set is an internal
model-selection set, not the generalisation measurement, so it is carved from the
training pool by ER-stratified CASE sampling (both classes guaranteed); it may
share sites with train but never with test, because it is drawn from the
test-excluded pool.

Two artifact sets are produced under ``project/CLAM/splits/``:
  * ``tcga_brca_er_100/``  -- 10 size-balanced site-holdout folds (splits_0..9)
  * ``tcga_brca_er_lsgo/`` -- one leave-site-groups-out split (splits_0)

Each fold writes three CSVs matching what CLAM's create_splits_seq.py emits:
  * ``splits_{i}.csv``            NaN-padded slide_id columns (train,val,test);
                                  this is the ONLY file CLAM reads at train time.
  * ``splits_{i}_bool.csv``       one-hot membership indexed by slide_id.
  * ``splits_{i}_descriptor.csv`` per-class slide counts per partition.

Run from the repo root (after the dataset_csv exists)::

    python tools/make_er_site_splits.py

With no dataset_csv present, or with ``--self-test``, a synthetic cohort is
generated and the full split routine is exercised and asserted instead.
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET_CSV = os.path.join(
    REPO_ROOT, "project", "CLAM", "dataset_csv", "tcga_brca_er.csv"
)
SPLITS_ROOT = os.path.join(REPO_ROOT, "project", "CLAM", "splits")
KFOLD_DIR = os.path.join(SPLITS_ROOT, "tcga_brca_er_100")
LSGO_DIR = os.path.join(SPLITS_ROOT, "tcga_brca_er_lsgo")

ER_LABELS = ("ER-negative", "ER-positive")
COLUMN_KEYS = ("train", "val", "test")

DEFAULT_SEED = 1
K_FOLDS = 10
INNER_VAL_SPLITS = 9   # ~1/9 ~= 11% of the non-test pool becomes validation
LSGO_TEST_SPLITS = 5   # one of 5 balanced site-packs (~20% of cases) is the test


# --------------------------------------------------------------------------- #
# Core data structures
# --------------------------------------------------------------------------- #
def site_of(case_id):
    """TCGA tissue-source site: barcode chars 6-7, e.g. TCGA-3C-AALI -> '3C'."""
    return case_id.split("-")[1]


def load_cases(df):
    """Collapse a (case_id, slide_id, label) frame to one row per case.

    Returns the per-case frame (case_id, site, label) and a case_id -> sorted
    slide_id list mapping. A case with inconsistent labels is a data error.
    """
    df = df.copy()
    df["case_id"] = df["case_id"].astype(str)
    df["slide_id"] = df["slide_id"].astype(str)
    df["label"] = df["label"].astype(str)

    slide_map = {}
    rows = []
    for case_id, group in df.groupby("case_id", sort=True):
        labels = set(group["label"])
        if len(labels) != 1:
            raise ValueError(f"case {case_id} has conflicting labels: {labels}")
        label = labels.pop()
        if label not in ER_LABELS:
            raise ValueError(f"case {case_id} has unexpected label {label!r}")
        slide_map[case_id] = sorted(group["slide_id"])
        rows.append({"case_id": case_id, "site": site_of(case_id), "label": label})

    cases = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    return cases, slide_map


def expand_slides(case_ids, slide_map):
    """All slide_ids for a set of cases, deterministically ordered."""
    slides = []
    for case_id in sorted(case_ids):
        slides.extend(slide_map[case_id])
    return slides


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def balanced_site_folds(cases, k):
    """Pack whole sites into k size-balanced folds; return a list of case-sets.

    Sites are placed largest-first into the currently least-loaded fold, so no
    site is split and fold sizes stay close despite the dominant-site skew. The
    ordering (by descending size, site name as tie-break) is deterministic.
    """
    site_members = cases.groupby("site")["case_id"].agg(list)
    ordered_sites = sorted(site_members.index, key=lambda s: (-len(site_members[s]), s))

    fold_cases = [set() for _ in range(k)]
    fold_load = [0] * k
    for site in ordered_sites:
        target = min(range(k), key=lambda f: (fold_load[f], f))
        members = site_members[site]
        fold_cases[target].update(members)
        fold_load[target] += len(members)
    return fold_cases


def carve_validation(pool, seed):
    """Split a per-case pool into (train_cases, val_cases) by ER-stratified case
    sampling. Validation is NOT site-grouped -- see the module docstring for why.
    """
    inner = StratifiedKFold(n_splits=INNER_VAL_SPLITS, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(pool, y=pool["label"]))
    train_cases = set(pool.iloc[train_idx]["case_id"])
    val_cases = set(pool.iloc[val_idx]["case_id"])
    return train_cases, val_cases


def kfold_partitions(cases, k, seed):
    """Yield (fold_index, train_cases, val_cases, test_cases) for each of k folds."""
    test_folds = balanced_site_folds(cases, k)
    for fold, test_cases in enumerate(test_folds):
        pool = cases[~cases["case_id"].isin(test_cases)].reset_index(drop=True)
        train_cases, val_cases = carve_validation(pool, seed)
        yield fold, train_cases, val_cases, test_cases


def lsgo_partition(cases, seed):
    """One leave-site-groups-out split: ~20% whole-site test (both ER classes via
    balanced packing), ER-stratified case validation, remainder train.
    """
    test_cases = balanced_site_folds(cases, LSGO_TEST_SPLITS)[0]
    pool = cases[~cases["case_id"].isin(test_cases)].reset_index(drop=True)
    train_cases, val_cases = carve_validation(pool, seed)
    return train_cases, val_cases, test_cases


# --------------------------------------------------------------------------- #
# Assertions (raise on any violation)
# --------------------------------------------------------------------------- #
def assert_partition_valid(cases, train_cases, val_cases, test_cases, tag):
    parts = {"train": train_cases, "val": val_cases, "test": test_cases}
    label_of = dict(zip(cases["case_id"], cases["label"]))
    site_of_case = dict(zip(cases["case_id"], cases["site"]))

    # (a) no case in more than one partition
    for a in COLUMN_KEYS:
        for b in COLUMN_KEYS:
            if a < b:
                overlap = parts[a] & parts[b]
                assert not overlap, (
                    f"[{tag}] case leak between {a}/{b}: {sorted(overlap)[:5]}"
                )

    # (b) test sites are disjoint from train and val sites (Howard test holdout).
    #     train and val may share sites: validation is an internal selection set.
    site_sets = {
        name: {site_of_case[c] for c in members} for name, members in parts.items()
    }
    trainval_sites = site_sets["train"] | site_sets["val"]
    site_leak = site_sets["test"] & trainval_sites
    assert not site_leak, (
        f"[{tag}] SITE leak: test shares sites with train/val: {sorted(site_leak)}"
    )

    # (c) every partition carries both ER classes, >=1 each
    for name, members in parts.items():
        present = {label_of[c] for c in members}
        for cls in ER_LABELS:
            assert cls in present, f"[{tag}] partition {name} missing class {cls}"


def assert_test_cover(cases, test_case_sets):
    """(d) across all test folds every case appears in exactly one test fold."""
    all_cases = set(cases["case_id"])
    seen = {}
    for fold, members in enumerate(test_case_sets):
        for c in members:
            assert c not in seen, (
                f"case {c} in test folds {seen[c]} and {fold}"
            )
            seen[c] = fold
    assert set(seen) == all_cases, (
        f"test folds are not a full cover: "
        f"{len(all_cases - set(seen))} cases never tested"
    )


# --------------------------------------------------------------------------- #
# CSV writers (mirror CLAM's save_splits artifacts)
# --------------------------------------------------------------------------- #
def write_split_csv(path, train_slides, val_slides, test_slides):
    series = [
        pd.Series(list(train_slides), dtype=object).reset_index(drop=True),
        pd.Series(list(val_slides), dtype=object).reset_index(drop=True),
        pd.Series(list(test_slides), dtype=object).reset_index(drop=True),
    ]
    df = pd.concat(series, axis=1)
    df.columns = list(COLUMN_KEYS)
    df.to_csv(path)


def write_bool_csv(path, train_slides, val_slides, test_slides):
    order = list(train_slides) + list(val_slides) + list(test_slides)
    counts = [len(train_slides), len(val_slides), len(test_slides)]
    one_hot = np.eye(3).astype(bool)
    bool_array = np.repeat(one_hot, counts, axis=0)
    df = pd.DataFrame(bool_array, index=order, columns=list(COLUMN_KEYS))
    df.to_csv(path)


def write_descriptor_csv(path, cases, train_cases, val_cases, test_cases, slide_map):
    label_of = dict(zip(cases["case_id"], cases["label"]))
    data = {name: [] for name in COLUMN_KEYS}
    partitions = {
        "train": train_cases, "val": val_cases, "test": test_cases,
    }
    for cls in ER_LABELS:
        for name in COLUMN_KEYS:
            n_slides = sum(
                len(slide_map[c]) for c in partitions[name] if label_of[c] == cls
            )
            data[name].append(n_slides)
    df = pd.DataFrame(data, index=list(ER_LABELS), columns=list(COLUMN_KEYS))
    df.to_csv(path)


def write_fold(out_dir, index, cases, slide_map, train_cases, val_cases, test_cases):
    train_slides = expand_slides(train_cases, slide_map)
    val_slides = expand_slides(val_cases, slide_map)
    test_slides = expand_slides(test_cases, slide_map)
    write_split_csv(
        os.path.join(out_dir, f"splits_{index}.csv"),
        train_slides, val_slides, test_slides,
    )
    write_bool_csv(
        os.path.join(out_dir, f"splits_{index}_bool.csv"),
        train_slides, val_slides, test_slides,
    )
    write_descriptor_csv(
        os.path.join(out_dir, f"splits_{index}_descriptor.csv"),
        cases, train_cases, val_cases, test_cases, slide_map,
    )
    return train_slides, val_slides, test_slides


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report_partition(tag, cases, slide_map, train_cases, val_cases, test_cases):
    label_of = dict(zip(cases["case_id"], cases["label"]))
    site_of_case = dict(zip(cases["case_id"], cases["site"]))
    parts = {"train": train_cases, "val": val_cases, "test": test_cases}
    print(f"  {tag}")
    for name in COLUMN_KEYS:
        members = parts[name]
        slides = expand_slides(members, slide_map)
        sites = {site_of_case[c] for c in members}
        per_cls = {
            cls: sum(1 for c in members if label_of[c] == cls) for cls in ER_LABELS
        }
        bal = ", ".join(f"{cls}={per_cls[cls]}" for cls in ER_LABELS)
        print(
            f"    {name:5s}: {len(slides):4d} slides / {len(members):4d} cases "
            f"/ {len(sites):3d} sites   cases[{bal}]"
        )


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def generate_kfold(cases, slide_map, out_dir, k, seed, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    test_case_sets = []
    if verbose:
        print(f"[{k}-fold size-balanced site-holdout CV] -> {out_dir}")
    for fold, train_cases, val_cases, test_cases in kfold_partitions(cases, k, seed):
        assert_partition_valid(
            cases, train_cases, val_cases, test_cases, tag=f"kfold[{fold}]"
        )
        write_fold(
            out_dir, fold, cases, slide_map, train_cases, val_cases, test_cases
        )
        test_case_sets.append(test_cases)
        if verbose:
            report_partition(
                f"fold {fold}", cases, slide_map,
                train_cases, val_cases, test_cases,
            )
    assert_test_cover(cases, test_case_sets)
    if verbose:
        print(f"  OK: {k} folds; test folds form a disjoint cover of all cases.")
    return test_case_sets


def generate_lsgo(cases, slide_map, out_dir, seed, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print(f"[leave-site-groups-out] -> {out_dir}")
    train_cases, val_cases, test_cases = lsgo_partition(cases, seed)
    assert_partition_valid(cases, train_cases, val_cases, test_cases, tag="lsgo")
    write_fold(out_dir, 0, cases, slide_map, train_cases, val_cases, test_cases)
    if verbose:
        report_partition(
            "split 0", cases, slide_map, train_cases, val_cases, test_cases
        )
        print("  OK: whole-site test holdout, both ER classes present.")
    return train_cases, val_cases, test_cases


def run_all(cases, slide_map, kfold_dir, lsgo_dir, seed, verbose=True):
    generate_kfold(cases, slide_map, kfold_dir, K_FOLDS, seed, verbose=verbose)
    generate_lsgo(cases, slide_map, lsgo_dir, seed, verbose=verbose)


# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #
def make_synthetic_frame(seed, n_cases=250, n_sites=25, pos_frac=0.78):
    rng = np.random.default_rng(seed)
    # Distinct two-char site codes, TCGA-barcode style (e.g. "0A", "1C").
    sites = [f"{i // 10}{chr(ord('A') + i % 10)}" for i in range(n_sites)]
    rows = []
    for idx in range(n_cases):
        site = sites[rng.integers(0, n_sites)]
        case_id = f"TCGA-{site}-{idx:04d}"
        label = ER_LABELS[1] if rng.random() < pos_frac else ER_LABELS[0]
        n_slides = int(rng.integers(1, 3))  # 1 or 2 slides
        for j in range(1, n_slides + 1):
            slide_id = f"{case_id}-01Z-00-DX{j}.SYNTH{idx:04d}{j}"
            rows.append({"case_id": case_id, "slide_id": slide_id, "label": label})
    return pd.DataFrame(rows)


def self_test(seed):
    print("=== SELF-TEST on synthetic cohort ===")
    frame = make_synthetic_frame(seed)
    cases, slide_map = load_cases(frame)
    n_pos = int((cases["label"] == ER_LABELS[1]).sum())
    n_neg = int((cases["label"] == ER_LABELS[0]).sum())
    print(
        f"synthetic: {len(frame)} slides / {len(cases)} cases / "
        f"{cases['site'].nunique()} sites   "
        f"cases[{ER_LABELS[0]}={n_neg}, {ER_LABELS[1]}={n_pos}]"
    )

    tmp = tempfile.mkdtemp(prefix="er_split_selftest_")
    try:
        kfold_dir = os.path.join(tmp, "tcga_brca_er_100")
        lsgo_dir = os.path.join(tmp, "tcga_brca_er_lsgo")
        test_case_sets = generate_kfold(
            cases, slide_map, kfold_dir, K_FOLDS, seed, verbose=False
        )
        generate_lsgo(cases, slide_map, lsgo_dir, seed, verbose=False)

        # Re-assert (a)-(d) explicitly at the harness level for a loud summary.
        for fold, train_cases, val_cases, test_cases in kfold_partitions(
            cases, K_FOLDS, seed
        ):
            assert_partition_valid(
                cases, train_cases, val_cases, test_cases, tag=f"selftest[{fold}]"
            )
        assert_test_cover(cases, test_case_sets)

        # Prove the on-disk format: read splits_0.csv back and confirm shape +
        # that its slide columns keep test sites disjoint from train/val.
        written = pd.read_csv(
            os.path.join(kfold_dir, "splits_0.csv"), index_col=0
        )
        assert list(written.columns) == list(COLUMN_KEYS), written.columns

        def slides_to_sites(series):
            slides = [s for s in series.dropna().tolist()]
            case_ids = {"-".join(s.split("-")[:3]) for s in slides}
            return {site_of(c) for c in case_ids}, case_ids

        tr_sites, tr_cases = slides_to_sites(written["train"])
        va_sites, va_cases = slides_to_sites(written["val"])
        te_sites, te_cases = slides_to_sites(written["test"])
        assert not (tr_cases & va_cases) and not (tr_cases & te_cases) \
            and not (va_cases & te_cases), "case leak in written splits_0.csv"
        assert not (te_sites & tr_sites) and not (te_sites & va_sites), \
            "test SITE leak in written splits_0.csv"

        n_test_total = sum(len(s) for s in test_case_sets)
        print("asserts passed:")
        print("  (a) no case_id crosses train/val/test         [10 folds + lsgo]")
        print("  (b) test sites disjoint from train/val sites   [10 folds + lsgo]")
        print("  (c) both ER classes present in every partition [10 folds + lsgo]")
        print(
            f"  (d) test folds are a disjoint cover: "
            f"{n_test_total} test cases across {len(test_case_sets)} folds "
            f"== {len(cases)} unique cases"
        )
        print("  on-disk splits_0.csv re-read: 3 cols (train,val,test), no test-site leak")
        print("=== SELF-TEST PASSED ===")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_csv", default=DEFAULT_DATASET_CSV,
                        help="CLAM dataset_csv with case_id,slide_id,label")
    parser.add_argument("--kfold_dir", default=KFOLD_DIR,
                        help="output dir for the 10-fold splits")
    parser.add_argument("--lsgo_dir", default=LSGO_DIR,
                        help="output dir for the leave-site-groups-out split")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed for reproducible splits")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="run only the synthetic self-test and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    # The self-test always proves the split logic (site/case non-leakage).
    self_test(args.seed)

    if args.self_test:
        return 0

    if not os.path.exists(args.dataset_csv):
        print()
        print(f"dataset_csv not found: {args.dataset_csv}")
        print("Build it first, then generate the real splits with:")
        print("    python tools/make_er_dataset_csv.py")
        print("    python tools/make_er_site_splits.py")
        return 0

    print()
    frame = pd.read_csv(args.dataset_csv, dtype=str)
    cases, slide_map = load_cases(frame)
    print(
        f"real cohort: {len(frame)} slides / {len(cases)} cases / "
        f"{cases['site'].nunique()} sites"
    )
    run_all(cases, slide_map, args.kfold_dir, args.lsgo_dir, args.seed, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
