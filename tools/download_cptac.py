#!/usr/bin/env python3
"""Download the CPTAC-BRCA multimodal cohort: WSIs, RNA-seq, and clinical labels.

The three modalities live in three different places, none of which talk to each
other, so this pulls each and joins them on the CPTAC case id (``01BR001``):

  wsi       TCIA PathDB. TCIA recommends the Aspera Faspex browser plugin, which
            needs a GUI, but PathDB also exposes every slide as a direct file URL
            that works headless. Range requests are honoured, so slide downloads
            resume and are only re-fetched when the local size is wrong.
  rna       GDC. The prospective breast cohort is registered under project
            CPTAC-2 (not CPTAC-3), and its STAR-Counts files are open access —
            no token, no dbGaP application.
  clinical  cBioPortal study brca_cptac_2020 (Krug et al. 2020) for PAM50 and
            receptor status, plus the CPTAC pan-cancer clinical table mirrored on
            Zenodo for recurrence and survival. GDC and PDC clinical carry
            neither PAM50 nor receptor status, so neither is enough on its own.

START HERE, and note that this is NOT what ``--modality clinical`` does::

    python tools/download_cptac.py --modality all --cohort-only --dry-run

That is the only invocation that produces the complete cohort metadata, and it
transfers no slide and no RNA file. ``cohort.csv`` is written only when more than
one modality is requested (``if len(wants) > 1``, near the bottom of ``main``),
so ``--modality clinical`` alone — which this docstring used to recommend as the
place to start — cannot produce it, while
``tools/cptac/prepare_cptac_manifest.py`` requires it. ``--dry-run`` still writes
``wsi_manifest.csv`` and ``rna_manifest.csv`` (both are written before the
dry-run check in their respective fetchers), and the three files it produces are
byte-identical to the ones a full download produces. ``dp-cptac phase=0`` is
exactly this command.

The bulk downloads, when you actually want the primary data::

    python tools/download_cptac.py --modality clinical    # ~100 KB
    python tools/download_cptac.py --modality rna         # 548 MB
    python tools/download_cptac.py --modality wsi --workers 8
    python tools/download_cptac.py --modality all --cohort-only

``--cohort-only`` drops slides whose case lacks RNA or a PAM50 label, which is the
difference between 654 slides / 114 GB and 391 slides / 68 GB.

Nothing in this repository opens a ``.svs``: the WSI features arrive
pre-extracted from HuggingFace, and only ``filename`` and ``mpp_x`` from
``wsi_manifest.csv`` are ever read. The 68 GB of slides is acquirable but
unnecessary for every number the project reports.

Only CPTAC-BRCA has the RNA and clinical wiring; ``--collection`` fetches WSIs
for any other CPTAC collection (CPTAC-LUAD, CPTAC-CCRCC, …).
"""

import argparse
import csv
import gzip
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

PATHDB = "https://pathdb.cancerimagingarchive.net"
GDC = "https://api.gdc.cancer.gov"
CBIOPORTAL = "https://www.cbioportal.org/api"
CBIO_STUDY = "brca_cptac_2020"
# Zenodo record behind the `cptac` package. The package itself needs a compiler
# for a pyranges dependency and will not build in this image, but its data is
# plain HTTP.
ZENODO = "https://zenodo.org/api/records/8394329/files"
PANCANCER_CLINICAL = "mssm-all_cancers-clinical-clinical_Pan-cancer.May2022.tsv.gz"

CHUNK_SIZE = 1 << 20
LABEL_ATTRS = [
    "PAM50",
    "ER_UPDATED_CLINICAL_STATUS",
    "PR_CLINICAL_STATUS",
    "ERBB2_UPDATED_CLINICAL_STATUS",
    "ERBB2_PROTEOGENOMIC_STATUS",
    "TNBC_UPDATED_CLINICAL_STATUS",
    "TUMOR_STAGE",
    "CD3_TILS_STATUS",
]


def case_id(raw):
    """cBioPortal prefixes the CPTAC ids with X (X01BR001); PathDB and GDC don't."""
    return raw[1:] if raw.startswith("X") and raw[1:2].isdigit() else raw


# --------------------------------------------------------------------------- WSI


def resolve_collection_id(name):
    response = requests.get(f"{PATHDB}/collections?_format=json", timeout=120)
    response.raise_for_status()
    matches = {
        item["name"][0]["value"]: item["tid"][0]["value"] for item in response.json()
    }
    if name not in matches:
        near = sorted(k for k in matches if name.lower() in k.lower())
        raise SystemExit(
            f"No PathDB collection named {name!r}."
            + (f" Did you mean: {', '.join(near)}?" if near else "")
        )
    return matches[name]


def fetch_wsi_manifest(collection_id):
    """Page through the collection and return one record per slide."""
    slides, page = [], 0
    while True:
        response = requests.get(
            f"{PATHDB}/listofimages/{collection_id}?_format=json&page={page}",
            timeout=300,
        )
        response.raise_for_status()
        items = response.json()
        if not items:
            return slides

        for item in items:
            url = item["field_wsiimage"][0]["url"].replace("http://", "https://")
            slides.append(
                {
                    "collection": item["studyid"][0]["value"],
                    "case_id": item["clinicaltrialsubjectid"][0]["value"],
                    "slide_id": item["nid"][0]["value"],
                    "filename": url.rsplit("/", 1)[-1],
                    "url": url,
                    "width": item["imagedvolumewidth"][0]["value"],
                    "height": item["imagedvolumeheight"][0]["value"],
                    "mpp_x": _first(item, "referencepixelphysicalvaluex"),
                    "mpp_y": _first(item, "referencepixelphysicalvaluey"),
                }
            )
        page += 1


def _first(item, key):
    values = item.get(key) or [{}]
    return values[0].get("value", "")


def remote_size(url, session):
    """PathDB omits Content-Length on HEAD, so read it off a one-byte range."""
    response = session.get(url, headers={"Range": "bytes=0-0"}, timeout=120)
    response.raise_for_status()
    return int(response.headers["Content-Range"].rsplit("/", 1)[-1])


def download_slide(slide, dest_dir, session):
    target = dest_dir / slide["filename"]
    expected = remote_size(slide["url"], session)
    have = target.stat().st_size if target.exists() else 0

    if have == expected:
        return "skipped", 0
    if have > expected:
        target.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(
        slide["url"], headers=headers, stream=True, timeout=600
    ) as response:
        response.raise_for_status()
        with open(target, "ab" if have else "wb") as handle:
            for chunk in response.iter_content(CHUNK_SIZE):
                handle.write(chunk)

    written = target.stat().st_size
    if written != expected:
        raise IOError(f"{slide['filename']}: got {written} bytes, expected {expected}")
    return ("resumed" if have else "downloaded"), expected - have


def cohort_case_ids(rna=None, labels=None):
    """Cases that carry every modality: WSI is filtered against this, so RNA + PAM50."""
    rna = fetch_rna_manifest() if rna is None else rna
    labels = fetch_cbioportal_labels() if labels is None else labels
    return {r["case_id"] for r in rna} & {r["case_id"] for r in labels if r.get("PAM50")}


def fetch_wsi(args, session, keep=None):
    collection_id = resolve_collection_id(args.collection)
    slides = fetch_wsi_manifest(collection_id)
    print(
        f"[wsi] {args.collection} (PathDB {collection_id}): "
        f"{len(slides)} slides, {len({s['case_id'] for s in slides})} cases"
    )
    if keep is not None:
        dropped = [s for s in slides if s["case_id"] not in keep]
        slides = [s for s in slides if s["case_id"] in keep]
        print(f"[wsi] --cohort-only: keeping {len(slides)} slides from "
              f"{len({s['case_id'] for s in slides})} cases with RNA + PAM50; "
              f"skipping {len(dropped)} slides from "
              f"{len({s['case_id'] for s in dropped})} cases")
    if args.limit:
        slides = slides[: args.limit]
        print(f"[wsi] --limit {args.limit}: truncated to {len(slides)} slides")

    wsi_dir = args.output / "wsi"
    wsi_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "wsi_manifest.csv", slides)

    if args.dry_run:
        with ThreadPoolExecutor(args.workers * 2) as pool:
            sizes = list(
                tqdm(
                    pool.map(lambda s: remote_size(s["url"], session), slides),
                    total=len(slides),
                    desc="[wsi] sizing",
                )
            )
        on_disk = sum(
            (wsi_dir / s["filename"]).stat().st_size
            for s in slides
            if (wsi_dir / s["filename"]).exists()
        )
        print(f"[wsi] collection {sum(sizes) / 1e9:.1f} GB, on disk {on_disk / 1e9:.1f} GB, "
              f"remaining {(sum(sizes) - on_disk) / 1e9:.1f} GB")
        return slides

    run_downloads(
        slides, lambda s: download_slide(s, wsi_dir, session), "[wsi]", "slide",
        args.workers,
    )
    return slides


# --------------------------------------------------------------------------- RNA


def fetch_rna_manifest():
    """Open-access STAR-Counts for the CPTAC prospective breast cohort."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["CPTAC-2"]}},
            {"op": "in", "content": {"field": "cases.primary_site", "value": ["Breast"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    response = requests.post(
        f"{GDC}/files",
        json={
            "filters": filters,
            "size": "5000",
            "fields": "file_id,file_name,file_size,md5sum,"
            "cases.submitter_id,cases.samples.sample_type",
        },
        timeout=300,
    )
    response.raise_for_status()

    records = []
    for hit in response.json()["data"]["hits"]:
        case = (hit.get("cases") or [{}])[0]
        samples = case.get("samples") or [{}]
        sample_type = (samples[0].get("sample_type") or "unknown").replace(" ", "_")
        cid = case.get("submitter_id", "unknown")
        records.append(
            {
                "case_id": cid,
                "sample_type": sample_type,
                "filename": f"{cid}.{sample_type}.star_counts.tsv",
                "file_id": hit["file_id"],
                "gdc_file_name": hit["file_name"],
                "file_size": hit["file_size"],
                "md5sum": hit.get("md5sum", ""),
            }
        )
    return records


def download_rna_file(record, dest_dir, session):
    target = dest_dir / record["filename"]
    if target.exists() and target.stat().st_size == record["file_size"]:
        return "skipped", 0

    response = session.get(f"{GDC}/data/{record['file_id']}", timeout=600)
    response.raise_for_status()
    target.write_bytes(response.content)
    if target.stat().st_size != record["file_size"]:
        raise IOError(f"{record['filename']}: size mismatch")
    return "downloaded", record["file_size"]


def build_tpm_matrix(records, rna_dir, out_path):
    """Collapse the per-case STAR-Counts files into one genes x cases TPM matrix.

    The layout mirrors the TCGA expression matrix that tools/rna/ produces, so the
    fusion head can consume either. Note the values are GDC STAR TPM while the
    TCGA side is Xena log2 RSEM — they are NOT on the same scale and need
    harmonising before a model trained on one is applied to the other.
    """
    columns, genes = {}, None
    for record in tqdm(records, desc="[rna] merging", unit="file"):
        path = rna_dir / record["filename"]
        if not path.exists():
            continue
        names, values = [], []
        with open(path) as handle:
            next(handle)  # "# gene-model: GENCODE v36"
            header = next(handle).rstrip("\n").split("\t")
            gene_col, tpm_col = header.index("gene_name"), header.index("tpm_unstranded")
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if parts[0].startswith("N_"):  # STAR summary rows
                    continue
                names.append(parts[gene_col])
                values.append(parts[tpm_col])
        if genes is None:
            genes = names
        elif names != genes:
            print(f"[rna] gene order differs in {record['filename']}, skipping merge column",
                  file=sys.stderr)
            continue
        columns[record["case_id"]] = values

    if not columns:
        print("[rna] no files to merge", file=sys.stderr)
        return
    with gzip.open(out_path, "wt") as handle:
        case_ids = sorted(columns)
        handle.write("gene_name\t" + "\t".join(case_ids) + "\n")
        for i, gene in enumerate(genes):
            handle.write(gene + "\t" + "\t".join(columns[c][i] for c in case_ids) + "\n")
    print(f"[rna] merged matrix -> {out_path}  ({len(genes)} genes x {len(case_ids)} cases)")


def fetch_rna(args, session):
    records = fetch_rna_manifest()
    print(f"[rna] {len(records)} open-access STAR-Counts files, "
          f"{len({r['case_id'] for r in records})} cases")

    rna_dir = args.output / "rna"
    rna_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "rna_manifest.csv", records)

    if args.dry_run:
        print(f"[rna] total {sum(r['file_size'] for r in records) / 1e6:.0f} MB")
        return records

    run_downloads(
        records, lambda r: download_rna_file(r, rna_dir, session), "[rna]", "file",
        args.workers,
    )
    build_tpm_matrix(records, rna_dir, args.output / "rna" / "tpm_matrix.tsv.gz")
    return records


# ---------------------------------------------------------------------- clinical


def fetch_cbioportal_labels():
    """PAM50 and receptor status from Krug et al. 2020 via cBioPortal."""
    merged = {}
    for level in ("PATIENT", "SAMPLE"):
        response = requests.get(
            f"{CBIOPORTAL}/studies/{CBIO_STUDY}/clinical-data",
            params={"clinicalDataType": level, "projection": "DETAILED", "pageSize": 100000},
            timeout=300,
        )
        response.raise_for_status()
        for entry in response.json():
            attr = entry["clinicalAttributeId"]
            if attr not in LABEL_ATTRS:
                continue
            cid = case_id(entry.get("patientId") or entry.get("sampleId"))
            # the source mixes 'Negative' and 'negative' for HER2/TNBC
            merged.setdefault(cid, {"case_id": cid})[attr] = entry["value"].capitalize() \
                if attr.endswith("STATUS") else entry["value"]
    return [merged[c] for c in sorted(merged)]


def fetch_pancancer_clinical():
    """CPTAC pan-cancer clinical (recurrence + survival), filtered to breast."""
    response = requests.get(f"{ZENODO}/{PANCANCER_CLINICAL}/content", timeout=300)
    response.raise_for_status()
    text = gzip.decompress(response.content).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = [r for r in reader if str(r.get("tumor_code", "")).upper() == "BR"]
    for row in rows:
        row["case_id"] = row.get("case_id", "")
    return rows


def fetch_clinical(args):
    clinical_dir = args.output / "clinical"
    clinical_dir.mkdir(parents=True, exist_ok=True)

    labels = fetch_cbioportal_labels()
    write_csv(clinical_dir / "cbioportal_labels.csv", labels)
    have_pam50 = sum(1 for r in labels if r.get("PAM50"))
    print(f"[clinical] cBioPortal {CBIO_STUDY}: {len(labels)} cases, {have_pam50} with PAM50"
          f" -> {clinical_dir / 'cbioportal_labels.csv'}")

    pancancer = fetch_pancancer_clinical()
    write_csv(clinical_dir / "cptac_pancancer_clinical_breast.csv", pancancer)
    print(f"[clinical] CPTAC pan-cancer clinical: {len(pancancer)} breast cases"
          f" -> {clinical_dir / 'cptac_pancancer_clinical_breast.csv'}")
    return labels, pancancer


# ------------------------------------------------------------------------ shared


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list({k: None for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_downloads(items, worker, tag, unit, workers):
    tallies = {"downloaded": 0, "resumed": 0, "skipped": 0}
    transferred, failures = 0, []
    progress = tqdm(total=len(items), desc=f"{tag} downloading", unit=unit)
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            try:
                status, size = future.result()
                tallies[status] += 1
                transferred += size
            except Exception as exc:
                failures.append((futures[future]["filename"], exc))
            progress.update()
            progress.set_postfix(new=tallies["downloaded"], failed=len(failures))
    progress.close()
    print(f"{tag} downloaded {tallies['downloaded']}, resumed {tallies['resumed']}, "
          f"already complete {tallies['skipped']} — {transferred / 1e9:.2f} GB transferred")
    if failures:
        print(f"{tag} {len(failures)} failed (re-run to retry):", file=sys.stderr)
        for name, exc in failures[:20]:
            print(f"   {name}: {exc}", file=sys.stderr)
    return failures


def write_cohort(args, slides, rna, labels, pancancer):
    """One row per case: what modalities it has, plus its labels."""
    by_case = {}
    for slide in slides or []:
        entry = by_case.setdefault(slide["case_id"], {"case_id": slide["case_id"]})
        entry["n_slides"] = entry.get("n_slides", 0) + 1
    for record in rna or []:
        by_case.setdefault(record["case_id"], {"case_id": record["case_id"]})["rna_file"] = \
            record["filename"]
    for row in labels or []:
        by_case.setdefault(row["case_id"], {"case_id": row["case_id"]}).update(
            {k: v for k, v in row.items() if k != "case_id"}
        )
    for row in pancancer or []:
        entry = by_case.setdefault(row["case_id"], {"case_id": row["case_id"]})
        for src, dst in (
            ("Recurrence status (1, yes; 0, no)", "recurrence"),
            ("Overall survival, days", "os_days"),
            ("Survival status (1, dead; 0, alive)", "os_event"),
        ):
            if row.get(src) not in (None, ""):
                entry[dst] = row[src]

    rows = [by_case[c] for c in sorted(by_case)]
    for row in rows:
        row.setdefault("n_slides", 0)
    path = args.output / "cohort.csv"
    write_csv(path, rows)

    complete = [r for r in rows if r.get("n_slides") and r.get("rna_file") and r.get("PAM50")]
    print(f"\n[cohort] {len(rows)} cases -> {path}")
    print(f"[cohort] with WSI: {sum(1 for r in rows if r.get('n_slides'))} | "
          f"with RNA: {sum(1 for r in rows if r.get('rna_file'))} | "
          f"with PAM50: {sum(1 for r in rows if r.get('PAM50'))}")
    print(f"[cohort] WSI + RNA + PAM50: {len(complete)} cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", default="all",
                        choices=["all", "wsi", "rna", "clinical"])
    parser.add_argument("--collection", default="CPTAC-BRCA")
    parser.add_argument("--output", type=Path, default=Path(".datasets/cptac-brca"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0,
                        help="download only the first N slides (wsi only)")
    parser.add_argument("--cohort-only", action="store_true",
                        help="restrict WSIs to cases that also have RNA and a PAM50 label "
                             "(391 of 654 slides, 68 of 114 GB)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write manifests and report sizes without downloading")
    args = parser.parse_args()

    wants = {"all": {"wsi", "rna", "clinical"}}.get(args.modality, {args.modality})
    if args.collection != "CPTAC-BRCA" and (wants & {"rna", "clinical"} or args.cohort_only):
        if args.modality == "all" and not args.cohort_only:
            print(f"[!] RNA and clinical are wired for CPTAC-BRCA only; "
                  f"fetching WSIs only for {args.collection}.", file=sys.stderr)
            wants = {"wsi"}
        else:
            raise SystemExit(
                f"--modality {args.modality}"
                f"{' with --cohort-only' if args.cohort_only else ''} supports CPTAC-BRCA only."
            )

    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    slides = rna = labels = pancancer = None

    if "clinical" in wants:
        labels, pancancer = fetch_clinical(args)
    if "rna" in wants:
        rna = fetch_rna(args, session)
    if "wsi" in wants:
        # --cohort-only needs the RNA manifest and labels even when those modalities
        # were not requested; both are metadata-only queries, no bulk transfer.
        keep = cohort_case_ids(rna, labels) if args.cohort_only else None
        slides = fetch_wsi(args, session, keep)

    if len(wants) > 1:
        write_cohort(args, slides, rna, labels, pancancer)


if __name__ == "__main__":
    main()
