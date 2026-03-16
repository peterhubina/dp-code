"""
Fetch TCGA-BRCA clinical labels (DFI) from the TCGA-CDR supplemental table.
Liu et al. 2018, Cell - "An Integrated TCGA Pan-Cancer Clinical Data Resource"
Saves: tools/data/tcga_brca_labels.csv with columns: case_id, DFI, DFI_STATUS, label
"""

import os
import pandas as pd
import urllib.request

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "tcga_brca_labels.csv")

# TCGA-CDR supplemental table (Liu et al. 2018 Cell) hosted on GDC/Synapse
# This URL points to the published Excel file from the paper
CDR_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S0092867418302290-mmc1.xlsx"
CDR_LOCAL = os.path.join(OUTPUT_DIR, "TCGA-CDR-SupplementalTableS1.xlsx")


def download_cdr(url: str, dest: str) -> None:
    print(f"Downloading TCGA-CDR from {url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"Saved to {dest}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CDR_LOCAL):
        download_cdr(CDR_URL, CDR_LOCAL)
    else:
        print(f"Using cached CDR file: {CDR_LOCAL}")

    print("Reading CDR Excel file ...")
    # The sheet is named "TCGA-CDR" in the supplemental file
    try:
        df = pd.read_excel(CDR_LOCAL, sheet_name="TCGA-CDR", engine="openpyxl")
    except Exception:
        # Some versions use the first sheet
        df = pd.read_excel(CDR_LOCAL, sheet_name=0, engine="openpyxl")

    print(f"Total CDR rows: {len(df)}")
    print(f"Columns: {list(df.columns)[:20]}")

    # Filter to BRCA
    brca = df[df["type"] == "BRCA"].copy()
    print(f"BRCA patients: {len(brca)}")

    # Keep relevant columns
    # DFI = 1 means event (recurrence/progression), 0 = censored
    # DFI.time = days to event or censoring
    keep_cols = ["bcr_patient_barcode", "DFI", "DFI.time"]
    available = [c for c in keep_cols if c in brca.columns]
    brca = brca[available].copy()

    # Rename for clarity
    rename_map = {
        "bcr_patient_barcode": "case_id",
        "DFI": "DFI_STATUS",
        "DFI.time": "DFI",
    }
    brca = brca.rename(columns={k: v for k, v in rename_map.items() if k in brca.columns})

    # Drop rows with missing DFI_STATUS
    before = len(brca)
    brca = brca.dropna(subset=["DFI_STATUS"])
    print(f"Dropped {before - len(brca)} rows with missing DFI_STATUS")

    # Convert DFI_STATUS to int
    brca["DFI_STATUS"] = brca["DFI_STATUS"].astype(int)

    # Create label column
    brca["label"] = brca["DFI_STATUS"].map({1: "recurrence", 0: "no_recurrence"})

    print(f"\nFinal BRCA rows: {len(brca)}")
    print(f"Label distribution:\n{brca['label'].value_counts()}")
    recurrence_rate = brca["DFI_STATUS"].mean() * 100
    print(f"Recurrence rate: {recurrence_rate:.1f}%")

    brca.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
