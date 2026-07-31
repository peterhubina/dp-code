"""
CPTAC-BRCA clinicopath table in the TCGA training schema
========================================================
Emits the same 24 numeric columns, in the same order, that
tools/build_er_labels.py wrote for TCGA, so the trained `er_wsi_clinpath_gated`
fold checkpoints can be applied to CPTAC without retraining. The per-fold
standardisation is reloaded from s_{fold}_tabular_transform.json, which stores
selected_feature_names -- so columns are matched by NAME at inference, but this
script still emits them in training order to make diffs readable.

Stage / T / N / M reuse the exact collapse functions from build_er_labels.py
rather than reimplementing them; a divergence there would be invisible in the
output and would silently shift the encoding.

Histology cannot be shared: TCGA's collapse_histology is an exact-string lookup
over GDC's controlled vocabulary, and CPTAC records free text (including the
"Inflitrating" misspelling, which appears in the source data). A naive reuse
would send EVERY CPTAC case to "other". The vocabulary is small and closed, so
it is mapped explicitly below and any unseen string raises rather than silently
degrading to "other".

Caveat worth carrying into the writeup: stage, pT and pM are eligibility-shifted
in CPTAC, which enrolled stage IIA-IIIC only (Krug 2020, PMID 33212010). They
are emitted because the trained model expects them, not because they transport.

    python tools/cptac/prepare_cptac_clinicopath.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_er_labels import (CATEGORICAL_ENCODERS, UNKNOWN,  # noqa: E402
                             collapse_histology)

# CPTAC free-text histology -> the GDC controlled string TCGA's collapse
# function expects. Rationale per non-obvious entry:
#   "IDC and DCIS"            invasive component is ductal; DCIS is in situ
#   "... and Mucinous ..."    two named invasive types -> mixed, as TCGA would
#   "... papillary ..."       invasive ductal with papillary features -> ductal
CPTAC_HISTOLOGY = {
    "Inflitrating Ductal Carcinoma": "Infiltrating Ductal Carcinoma",
    "Inflitrating Lobular Carcinoma": "Infiltrating Lobular Carcinoma",
    "Infiltrating Ductal and Lobular Carcinoma": "Mixed Histology (please specify)",
    "Mixed ductal and lobular features": "Mixed Histology (please specify)",
    "Infiltrating Ductal and Mucinous Carcinoma": "Mixed Histology (please specify)",
    "IDC and DCIS": "Infiltrating Ductal Carcinoma",
    "invasive ductal carcinoma with extensive high-grade DCIS with comedonecrosis "
    "and features of papillary carcinoma": "Infiltrating Ductal Carcinoma",
    "Mucinous Carcinoma": "Mucinous Carcinoma",      # not in TCGA's dict -> "other"
    "Other (specify)": "Other, specify",             # not in TCGA's dict -> "other"
}

MISSING = {"", "nan", "NA", "Not Reported/ Unknown", "Not Applicable", "Unknown",
           "Staging is not applicable or unknown", "None"}


def parse_args():
    parser = argparse.ArgumentParser(description="CPTAC clinicopath table in TCGA schema")
    parser.add_argument("--pancancer_csv", type=str,
                        default=".datasets/cptac-brca/clinical/cptac_pancancer_clinical_breast.csv")
    parser.add_argument("--labels_csv", type=str,
                        default=".datasets/cptac-brca/clinical/cbioportal_labels.csv")
    parser.add_argument("--dataset_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_er_dataset.csv")
    parser.add_argument("--reference_csv", type=str,
                        default="tools/data/tcga_brca_clinicopath_clam.csv",
                        help="TCGA table whose column order and dtypes are mirrored")
    parser.add_argument("--out_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_er_clinicopath_clam.csv")
    return parser.parse_args()


def clean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text in MISSING else text


def map_histology(value):
    text = clean(value)
    if text is None:
        return None
    if text not in CPTAC_HISTOLOGY:
        raise SystemExit(
            f"unmapped CPTAC histology string: {text!r}. Add it to CPTAC_HISTOLOGY "
            "with a documented rationale rather than letting it fall through to 'other'."
        )
    return CPTAC_HISTOLOGY[text]


def main():
    args = parse_args()

    manifest = pd.read_csv(args.dataset_csv)
    cases = manifest[["case_id", "label_name"]].drop_duplicates("case_id")
    print(f"ER manifest: {len(cases)} cases")

    pan = pd.read_csv(args.pancancer_csv, low_memory=False).drop_duplicates("case_id")
    labels = pd.read_csv(args.labels_csv).drop_duplicates("case_id")
    src = cases.merge(pan, on="case_id", how="left").merge(
        labels[["case_id", "TUMOR_STAGE"]], on="case_id", how="left")
    print(f"joined pan-cancer clinical for {src['consent/age'].notna().sum()}/{len(src)} cases")

    # cBioPortal TUMOR_STAGE carries substages (Stage IIA); the pan-cancer table
    # only has Stage II. Prefer the finer one, fall back to the coarse one --
    # collapse_stage reduces both to I/II/III/IV anyway.
    raw = pd.DataFrame({
        "case_id": src["case_id"],
        "label": src["label_name"],
        "age": src["consent/age"],
        "ajcc_stage": src["TUMOR_STAGE"].where(
            src["TUMOR_STAGE"].notna(), src["baseline/tumor_stage_pathological"]),
        "pathologic_t": src["baseline/pathologic_staging_primary_tumor_pt"],
        "pathologic_n": src["baseline/pathologic_staging_regional_lymph_nodes_pn"],
        "pathologic_m": src["baseline/pathologic_staging_distant_metastasis_pm"],
        "histological_type": src["baseline/histologic_type"].map(map_histology),
    })

    rows = []
    for record in raw.to_dict("records"):
        age = clean(record["age"])
        features = {"case_id": record["case_id"], "label": record["label"],
                    "age": float(age) if age is not None else ""}
        for field, (collapse, levels) in CATEGORICAL_ENCODERS.items():
            active = collapse(clean(record[field]))
            for level in levels:
                features[f"{field}_{level}"] = 1 if level == active else 0
        rows.append(features)
    table = pd.DataFrame(rows)

    reference = pd.read_csv(args.reference_csv, nrows=1)
    expected = list(reference.columns)
    if list(table.columns) != expected:
        missing = set(expected) - set(table.columns)
        extra = set(table.columns) - set(expected)
        raise SystemExit(f"column mismatch vs {args.reference_csv}: missing={missing} extra={extra}")
    table = table[expected]

    print("\n=== collapsed level distributions (CPTAC vs TCGA training table) ===")
    tcga = pd.read_csv(args.reference_csv)
    for field, (_, levels) in CATEGORICAL_ENCODERS.items():
        cols = [f"{field}_{level}" for level in levels]
        cptac_share = (table[cols].sum() / len(table) * 100).round(1)
        tcga_share = (tcga[cols].sum() / len(tcga) * 100).round(1)
        comparison = pd.DataFrame({"CPTAC %": cptac_share, "TCGA %": tcga_share})
        print(f"\n{field}:")
        print(comparison.to_string())

    blank_age = (table["age"] == "").sum()
    print(f"\nage: {len(table) - blank_age}/{len(table)} present "
          f"({blank_age} blank -> NaN -> training mean after standardisation)")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}  ({len(table)} cases x {len(table.columns)} columns)")


if __name__ == "__main__":
    main()
