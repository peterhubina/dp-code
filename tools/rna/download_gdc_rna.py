#!/usr/bin/env python3
"""Download TCGA-BRCA primary-tumor STAR-Counts from GDC.

The RNA branch of the fusion head was originally trained on UCSC Xena HiSeqV2
(log2 RSEM normalized_count, hg19, ~20.5k HUGO symbols). CPTAC RNA comes from
GDC as STAR counts (hg38, GENCODE v36, Ensembl ids), so the two are not on a
common scale and a model trained on one cannot be applied to the other without
a cross-pipeline mapping.

GDC also hosts TCGA-BRCA under the *same* STAR-Counts workflow, so pulling it
here lets both cohorts come out of one pipeline and removes the mismatch by
construction rather than by normalisation. That is the Tier-1 route in
tools/rna/build_gdc_expression.py.

    python tools/rna/download_gdc_rna.py                 # TCGA-BRCA primary tumour
    python tools/rna/download_gdc_rna.py --workers 12
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

GDC = "https://api.gdc.cancer.gov"
ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="TCGA-BRCA")
    parser.add_argument("--sample_type", default="Primary Tumor")
    parser.add_argument("--out_dir", default=str(ROOT / ".datasets/tcga-brca/rna"))
    parser.add_argument("--manifest", default=str(ROOT / ".datasets/tcga-brca/rna_manifest.csv"))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def fetch_manifest(project, sample_type):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": [sample_type]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    response = requests.post(
        f"{GDC}/files",
        json={
            "filters": filters,
            "size": "20000",
            "fields": "file_id,file_name,file_size,md5sum,"
            "cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type",
        },
        timeout=600,
    )
    response.raise_for_status()

    records = []
    for hit in response.json()["data"]["hits"]:
        case = (hit.get("cases") or [{}])[0]
        cid = case.get("submitter_id", "unknown")
        samples = [s for s in (case.get("samples") or []) if s.get("sample_type") == sample_type]
        sample_barcode = samples[0].get("submitter_id", "") if samples else ""
        records.append(
            {
                "case_id": cid,
                "sample_barcode": sample_barcode,
                "filename": f"{cid}.{sample_barcode or 'NA'}.star_counts.tsv",
                "file_id": hit["file_id"],
                "gdc_file_name": hit["file_name"],
                "file_size": hit["file_size"],
                "md5sum": hit.get("md5sum", ""),
            }
        )
    return records


def dedupe_one_per_case(records):
    """TCGA cases can carry several primary-tumour aliquots; keep one, deterministically."""
    by_case = {}
    for record in sorted(records, key=lambda r: (r["case_id"], r["sample_barcode"], r["file_id"])):
        by_case.setdefault(record["case_id"], record)
    dropped = len(records) - len(by_case)
    if dropped:
        print(f"[rna] {dropped} extra aliquots dropped, keeping the first per case "
              f"(sorted by sample barcode then file id)")
    return list(by_case.values())


def download(record, dest_dir, session):
    target = dest_dir / record["filename"]
    if target.exists() and target.stat().st_size == record["file_size"]:
        return "skipped", 0
    response = session.get(f"{GDC}/data/{record['file_id']}", timeout=600)
    response.raise_for_status()
    target.write_bytes(response.content)
    if target.stat().st_size != record["file_size"]:
        raise IOError(f"{record['filename']}: size mismatch")
    return "downloaded", record["file_size"]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = fetch_manifest(args.project, args.sample_type)
    print(f"[rna] {len(records)} {args.sample_type} STAR-Counts files for {args.project}")
    records = dedupe_one_per_case(records)
    print(f"[rna] {len(records)} cases, {sum(r['file_size'] for r in records) / 1e9:.1f} GB")

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"[rna] wrote {args.manifest}")

    session = requests.Session()
    tallies = {"downloaded": 0, "skipped": 0}
    failures = []
    progress = tqdm(total=len(records), desc="[rna] downloading", unit="file")
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download, r, out_dir, session): r for r in records}
        for future in as_completed(futures):
            try:
                status, _ = future.result()
                tallies[status] += 1
            except Exception as exc:
                failures.append((futures[future]["filename"], exc))
            progress.update()
            progress.set_postfix(new=tallies["downloaded"], failed=len(failures))
    progress.close()

    print(f"[rna] downloaded {tallies['downloaded']}, already complete {tallies['skipped']}")
    if failures:
        print(f"[rna] {len(failures)} failed (re-run to retry):", file=sys.stderr)
        for name, exc in failures[:20]:
            print(f"   {name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
