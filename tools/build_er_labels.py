#!/usr/bin/env python3
"""Build the TCGA-BRCA ER-status label and clinicopath tables.

Produces three CSVs under ``tools/data``:

1. ``tcga_brca_er_labels.csv``       -- case_id, label (ER-positive/ER-negative)
2. ``tcga_brca_clinicopath.csv``     -- human-readable clinicopath, one row per ER-labeled case
3. ``tcga_brca_clinicopath_clam.csv``-- CLAM ``--tabular_csv`` table (numeric features, ER label)

The estrogen-receptor call and clinicopath fields come from the cBioPortal
``brca_tcga`` study (the pan-cancer-atlas study does not expose ER-by-IHC). The
local Xena clinical matrix is used as a cross-check and as an offline fallback;
the two are the same underlying GDC clinical data. When both give a definitive
Positive/Negative call and disagree, the cBioPortal call wins.

Clinicopath features carry no receptor status: predicting ER from ER would leak.
TCGA-BRCA has no Nottingham grade, so no grade column is emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

CBIOPORTAL_API = "https://www.cbioportal.org/api"
ER_STUDY = "brca_tcga"

# cBioPortal patient-level attribute ids for the fields we need.
CBIO_ATTRS = {
    "er": "ER_STATUS_BY_IHC",
    "age": "AGE",
    "ajcc_stage": "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "pathologic_t": "AJCC_TUMOR_PATHOLOGIC_PT",
    "pathologic_n": "AJCC_NODES_PATHOLOGIC_PN",
    "pathologic_m": "AJCC_METASTASIS_PATHOLOGIC_PM",
    "histological_type": "HISTOLOGICAL_DIAGNOSIS",
}

# Column names in the local Xena clinical matrix that mirror CBIO_ATTRS.
LOCAL_MATRIX = ".scratch/TCGA-BRCA-rna/TCGA_BRCA_clinicalMatrix.tsv"
LOCAL_COLS = {
    "er": "breast_carcinoma_estrogen_receptor_status",
    "age": "age_at_initial_pathologic_diagnosis",
    "ajcc_stage": "pathologic_stage",
    "pathologic_t": "pathologic_T",
    "pathologic_n": "pathologic_N",
    "pathologic_m": "pathologic_M",
    "histological_type": "histological_type",
}
# Tumour samples first (-01 primary, -06 metastatic); -11 is matched normal.
TUMOUR_SAMPLE_ORDER = ("01", "06")

ER_POSITIVE = "ER-positive"
ER_NEGATIVE = "ER-negative"
CALL_TO_LABEL = {"Positive": ER_POSITIVE, "Negative": ER_NEGATIVE}

# Placeholder tokens that mean "no value" in the GDC/Xena exports.
MISSING_TOKENS = {"", "NA", "[Not Available]", "[Not Applicable]", "[Unknown]", "[Discrepancy]", "null", "None"}

CLINICOPATH_FIELDS = ["age", "ajcc_stage", "pathologic_t", "pathologic_n", "pathologic_m", "histological_type"]


def clean(value):
    """Return a stripped value, or None if it is a missing-data placeholder."""
    if value is None:
        return None
    value = str(value).strip()
    return None if value in MISSING_TOKENS else value


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def fetch_cbioportal(study):
    """Return {case_id: {field: value}} from cBioPortal, or None if unreachable."""
    url = f"{CBIOPORTAL_API}/studies/{study}/clinical-data?clinicalDataType=PATIENT&projection=DETAILED&pageSize=10000000"
    try:
        with urllib.request.urlopen(url, timeout=90) as response:
            records = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[warn] cBioPortal fetch failed ({exc}); falling back to the local matrix.")
        return None

    attr_to_field = {attr: field for field, attr in CBIO_ATTRS.items()}
    cases = defaultdict(dict)
    for record in records:
        field = attr_to_field.get(record["clinicalAttributeId"])
        if field is not None:
            cases[record["patientId"]][field] = clean(record["value"])
    print(f"[info] cBioPortal '{study}': {len(cases)} patients")
    return dict(cases)


def load_local_matrix(path):
    """Return {case_id: {field: value}} from the Xena matrix, one row per case.

    Values are taken from the primary tumour sample when present. Cases whose
    tumour samples carry conflicting definitive ER calls are reported and their
    ER value is left unresolved (None).
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    samples_by_case = defaultdict(dict)
    for row in rows:
        parts = row["sampleID"].split("-")
        case_id = "-".join(parts[:3])
        sample_type = parts[3][:2] if len(parts) > 3 else "??"
        samples_by_case[case_id][sample_type] = row

    cases, conflicts = {}, []
    for case_id, samples in samples_by_case.items():
        preferred = _preferred_sample(samples)
        fields = {field: clean(preferred.get(col)) for field, col in LOCAL_COLS.items()}

        tumour_calls = {
            clean(samples[st].get(LOCAL_COLS["er"]))
            for st in TUMOUR_SAMPLE_ORDER
            if st in samples
        }
        tumour_calls &= set(CALL_TO_LABEL)
        primary_call = clean(samples.get("01", {}).get(LOCAL_COLS["er"]))
        if len(tumour_calls) > 1 and primary_call not in CALL_TO_LABEL:
            conflicts.append(case_id)
            fields["er"] = None

        cases[case_id] = fields

    print(f"[info] local matrix: {len(cases)} cases ({len(conflicts)} dropped for ER conflict across samples)")
    return cases, conflicts


def _preferred_sample(samples):
    """Pick the tumour sample to read clinicopath from, preferring primary tumour."""
    for sample_type in (*TUMOUR_SAMPLE_ORDER, "11"):
        if sample_type in samples:
            return samples[sample_type]
    return next(iter(samples.values()))


# --------------------------------------------------------------------------- #
# ER merge
# --------------------------------------------------------------------------- #
def resolve_er_labels(cbio, local):
    """Merge the two sources into {case_id: label}, preferring cBioPortal.

    Returns the label map plus a stats dict describing drops and disagreements.
    """
    def definitive_call(record):
        return record.get("er") if record and record.get("er") in CALL_TO_LABEL else None

    labels, disagreements, dropped = {}, [], []
    for case_id in sorted(set(cbio) | set(local)):
        cbio_call = definitive_call(cbio.get(case_id))
        local_call = definitive_call(local.get(case_id))
        if cbio_call and local_call and cbio_call != local_call:
            disagreements.append((case_id, cbio_call, local_call))

        chosen = cbio_call or local_call
        if chosen:
            labels[case_id] = CALL_TO_LABEL[chosen]
        else:
            raw = (cbio.get(case_id) or {}).get("er") or (local.get(case_id) or {}).get("er")
            dropped.append((case_id, raw))

    stats = {
        "disagreements": disagreements,
        "dropped": dropped,
        "n_indeterminate": sum(1 for _, raw in dropped if raw == "Indeterminate"),
        "n_missing": sum(1 for _, raw in dropped if raw != "Indeterminate"),
    }
    return labels, stats


def merge_clinicopath(case_ids, cbio, local):
    """Per-case clinicopath, preferring cBioPortal and filling gaps from local."""
    merged = {}
    for case_id in case_ids:
        primary = cbio.get(case_id, {})
        secondary = local.get(case_id, {})
        merged[case_id] = {
            field: primary.get(field) or secondary.get(field)
            for field in CLINICOPATH_FIELDS
        }
    return merged


# --------------------------------------------------------------------------- #
# CLAM one-hot encoding
# --------------------------------------------------------------------------- #
UNKNOWN = "unknown"


def collapse_stage(value):
    if value is None:
        return UNKNOWN
    stage = value.replace("Stage", "").strip().upper()
    for level in ("IV", "III", "II", "I"):
        if stage.startswith(level):
            return level
    return UNKNOWN  # "X" cannot be assessed


def _collapse_tnm(value, levels):
    if value is None:
        return UNKNOWN
    token = value.upper().lstrip("C")  # cM0 -> M0
    for level in levels:
        if token.startswith(level):
            return level
    return UNKNOWN  # TX / NX / MX


def collapse_t(value):
    return _collapse_tnm(value, ("T1", "T2", "T3", "T4"))


def collapse_n(value):
    return _collapse_tnm(value, ("N0", "N1", "N2", "N3"))


def collapse_m(value):
    return _collapse_tnm(value, ("M0", "M1"))


HISTOLOGY_LEVELS = {
    "Infiltrating Ductal Carcinoma": "ductal",
    "Infiltrating Lobular Carcinoma": "lobular",
    "Mixed Histology (please specify)": "mixed",
}


def collapse_histology(value):
    if value is None:
        return UNKNOWN
    return HISTOLOGY_LEVELS.get(value, "other")


# Each categorical maps to a collapse function and the fixed level set it emits,
# so every case yields the same one-hot columns (including an explicit unknown).
CATEGORICAL_ENCODERS = {
    "ajcc_stage": (collapse_stage, ["I", "II", "III", "IV", UNKNOWN]),
    "pathologic_t": (collapse_t, ["T1", "T2", "T3", "T4", UNKNOWN]),
    "pathologic_n": (collapse_n, ["N0", "N1", "N2", "N3", UNKNOWN]),
    "pathologic_m": (collapse_m, ["M0", "M1", UNKNOWN]),
    "histological_type": (collapse_histology, ["ductal", "lobular", "mixed", "other", UNKNOWN]),
}


def clam_feature_row(clinicopath):
    """Return an ordered dict of numeric CLAM features for one case."""
    features = {}
    age = clinicopath.get("age")
    features["age"] = float(age) if age is not None else ""  # blank -> NaN, mean-imputed by CLAM
    for field, (collapse, levels) in CATEGORICAL_ENCODERS.items():
        active = collapse(clinicopath.get(field))
        for level in levels:
            features[f"{field}_{level}"] = 1 if level == active else 0
    return features


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(out_dir, labels, clinicopath):
    case_ids = sorted(labels)
    out_dir = Path(out_dir)

    labels_path = out_dir / "tcga_brca_er_labels.csv"
    write_csv(labels_path, ["case_id", "label"], [[c, labels[c]] for c in case_ids])

    clinico_path = out_dir / "tcga_brca_clinicopath.csv"
    header = ["case_id"] + CLINICOPATH_FIELDS
    rows = [[c] + [clinicopath[c].get(f) if clinicopath[c].get(f) is not None else "" for f in CLINICOPATH_FIELDS]
            for c in case_ids]
    write_csv(clinico_path, header, rows)

    clam_path = out_dir / "tcga_brca_clinicopath_clam.csv"
    feature_names = list(clam_feature_row(next(iter(clinicopath.values()))))
    header = ["case_id", "label"] + feature_names
    rows = []
    for case_id in case_ids:
        features = clam_feature_row(clinicopath[case_id])
        rows.append([case_id, labels[case_id]] + [features[name] for name in feature_names])
    write_csv(clam_path, header, rows)

    return labels_path, clinico_path, clam_path, feature_names


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def print_report(labels, stats, clinicopath, source, feature_names, paths):
    counts = Counter(labels.values())
    total = sum(counts.values())
    pos, neg = counts[ER_POSITIVE], counts[ER_NEGATIVE]

    def pct(n):
        return 100 * n / total if total else 0

    with_clinico = sum(
        1 for c in labels if any(clinicopath[c].get(f) is not None for f in CLINICOPATH_FIELDS)
    )

    print("\n" + "=" * 70)
    print("TCGA-BRCA ER label build report")
    print("=" * 70)
    print(f"ER source used            : {source}")
    print(f"Retained ER-labeled cases : {total}")
    print(f"  {ER_POSITIVE:12s}: {pos} ({pct(pos):.1f}%)")
    print(f"  {ER_NEGATIVE:12s}: {neg} ({pct(neg):.1f}%)")
    print(f"Cases with clinicopath    : {with_clinico} (all are ER-matched)")
    print(f"Dropped (no ER call)      : {len(stats['dropped'])} "
          f"= {stats['n_missing']} missing/NaN + {stats['n_indeterminate']} Indeterminate")
    print(f"cBioPortal<->local ER disagreements: {len(stats['disagreements'])}")
    for case_id, cbio_call, local_call in stats["disagreements"]:
        print(f"    {case_id}: cBioPortal={cbio_call} local={local_call}")
    print(f"Grade column              : omitted (TCGA-BRCA has no Nottingham grade)")
    print(f"CLAM feature columns ({len(feature_names)}): {', '.join(feature_names)}")
    for path in paths:
        print(f"  wrote {path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", default=None, help="defaults to <repo-root>/tools/data")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "tools" / "data"

    cbio = fetch_cbioportal(ER_STUDY)
    local, _conflicts = load_local_matrix(repo_root / LOCAL_MATRIX)

    if cbio is None:
        cbio, source = {}, "local Xena matrix (cBioPortal unreachable)"
    else:
        source = f"cBioPortal '{ER_STUDY}' (cross-checked against local Xena matrix)"

    labels, stats = resolve_er_labels(cbio, local)
    clinicopath = merge_clinicopath(labels.keys(), cbio, local)

    *paths, feature_names = write_outputs(out_dir, labels, clinicopath)
    print_report(labels, stats, clinicopath, source, feature_names, paths)


if __name__ == "__main__":
    main()
