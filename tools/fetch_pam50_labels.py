"""Fetch PAM50 molecular subtype labels from cBioPortal for TCGA-BRCA."""
import json
import os
import urllib.request

import pandas as pd


CBIOPORTAL_URL = (
    "https://www.cbioportal.org/api/studies/"
    "brca_tcga_pan_can_atlas_2018/clinical-data"
    "?clinicalDataType=PATIENT&projection=SUMMARY"
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "tcga_brca_pam50_labels.csv")

SUBTYPE_MAP = {
    "BRCA_LumA": "LumA",
    "BRCA_LumB": "LumB",
    "BRCA_Basal": "Basal",
    "BRCA_Her2": "Her2",
    "BRCA_Normal": "Normal",
}


def main():
    print("Fetching clinical data from cBioPortal...")
    req = urllib.request.Request(CBIOPORTAL_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    subtypes = [
        {"case_id": d["patientId"], "pam50_raw": d["value"]}
        for d in data
        if d.get("clinicalAttributeId") == "SUBTYPE"
    ]
    df = pd.DataFrame(subtypes)
    df["label"] = df["pam50_raw"].map(SUBTYPE_MAP)
    df = df.dropna(subset=["label"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df[["case_id", "label"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} PAM50 labels to {OUTPUT_PATH}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
