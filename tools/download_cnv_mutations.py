#!/usr/bin/env python3
"""Download harmonised copy-number and somatic-mutation data for TCGA-BRCA and CPTAC-BRCA.

Both cohorts are pulled from cBioPortal so the two sides go through the *same*
upstream pipeline — GISTIC2 for copy number, the same MAF schema for mutations.
Pulling TCGA from GDC and CPTAC from PDC instead would mean harmonising two
different callers before any modelling could start, which is the mistake the RNA
side of this project already paid for (see the Xena-vs-GDC scale mismatch note).

Two routes, because neither one covers both data types:

  cna        cBioPortal *datahub* flat files. One HTTP GET per cohort beats
             paginating 25k genes x 1k samples through the REST API.
  mutations  cBioPortal *REST* API. The datahub MAF is git-LFS backed and the
             repository's LFS budget is currently exhausted (TCGA's file 404s), so
             the API is the only complete route. ``POST /mutations/fetch`` with no
             gene filter returns the whole profile: 84,226 rows for TCGA.

Copy number comes in three representations and they are *not* interchangeable:

  gistic  ``data_cna.txt`` — GISTIC2 discrete gene-level call, -2 deep deletion …
          +2 high amplification. The most processed product in the chain, and the
          one Amer et al. 2025 (arXiv:2509.03408) used, so it is the right input
          for reproducing that baseline.
  log2    ``data_log2_cna.txt`` — continuous gene-level log2 ratio, one step less
          processed. Strictly more information than ``gistic``: thresholding to five
          integers throws away magnitude.
  arm     39 chromosome arms, median log2 per arm, derived here. This is the
          representation a *cheap* assay actually delivers. Shallow whole-genome
          sequencing at 0.1-0.5x resolves arm- and segment-scale changes but cannot
          call focal gene-level amplifications, so a model trained on 19,755 GISTIC
          calls does not transfer to a setting where sWGS is the only affordable
          assay. If the claim is "this works where RNA-seq is out of reach", the
          feature vector has to be one that route can produce.

TCGA additionally ships ``data_cna_hg19.seg`` (DNAcopy segments) and
``data_armlevel_cna.txt`` (official Gain/Loss/Unchanged arm calls). CPTAC ships
neither, which is why arm-level is derived from gene-level log2 for both cohorts
by the same code — ``--validate-arms`` scores that derivation against TCGA's
official calls so the shortcut is measured rather than assumed.

Platform caveat: TCGA copy number is Affymetrix SNP6.0 array -> GISTIC2, CPTAC is
WGS-derived. Same pipeline name, different measurement. Check the arm-level
distributions across cohorts before trusting a transfer result — this is the same
shape of trap as the Xena-vs-GDC RNA scale mismatch already recorded for this repo.

Sample ids are normalised to the case ids this repo already keys on:

  TCGA   ``TCGA-3C-AAAU-01`` -> ``TCGA-3C-AAAU``, primary tumour (``-01``) only
  CPTAC  ``X01BR001``        -> ``01BR001``

    python tools/download_cnv_mutations.py --what all --representation all
    python tools/download_cnv_mutations.py --what cna --representation arm --validate-arms
    python tools/download_cnv_mutations.py --what all --all-cases  # skip WSI filter

``--cohort-only`` (the default) keeps only cases that have UNI2-h features on disk,
so the matrices line up row-for-row with the CLAM datasets: 981 TCGA / 114 CPTAC.
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
DATAHUB = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
CBIOPORTAL = "https://www.cbioportal.org/api"

COHORTS = {
    "tcga": {
        "study": "brca_tcga_pan_can_atlas_2018",
        "sample_list": "brca_tcga_pan_can_atlas_2018_sequenced",
        "mut_profile": "brca_tcga_pan_can_atlas_2018_mutations",
        "labels": REPO / "tools/data/tcga_brca_pam50_labels.csv",
        "label_col": "case_id",
    },
    "cptac": {
        "study": "brca_cptac_2020",
        "sample_list": "brca_cptac_2020_sequenced",
        "mut_profile": "brca_cptac_2020_mutations",
        "labels": REPO / ".datasets/cptac-brca/cptac_brca_pam50_dataset.csv",
        "label_col": "case_id",
    },
}

# Non-silent consequences. Silent/intronic/UTR calls are noise for a subtype model
# and inflate the per-gene mutation rate by roughly a third.
CODING = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site",
    "Nonstop_Mutation", "Splice_Region",
}

# The 39 arms TCGA reports. Acrocentric p-arms (13p, 14p, 15p, 21p, 22p) are heterochromatic
# and carry no assayable genes, so they are absent by construction rather than by filtering.
ARMS = [f"{c}{a}" for c in list(range(1, 23)) for a in ("p", "q")
        if f"{c}{a}" not in {"13p", "14p", "15p", "21p", "22p"}]


def to_case_id(sample_id: str, cohort: str) -> str:
    """cBioPortal sample id -> the case id the rest of this repo keys on."""
    if cohort == "tcga":
        parts = sample_id.split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else sample_id
    return sample_id.lstrip("X")


def is_primary_tumour(sample_id: str, cohort: str) -> bool:
    if cohort != "tcga":
        return True
    parts = sample_id.split("-")
    return len(parts) < 4 or parts[3].startswith("01")


def wsi_cases(cohort: str) -> set[str] | None:
    """Cases with UNI2-h features on disk, or None if the label file is missing."""
    cfg = COHORTS[cohort]
    path = cfg["labels"]
    if not path.exists():
        print(f"  ! {path} not found — keeping all cases", file=sys.stderr)
        return None
    with open(path) as fh:
        return {r[cfg["label_col"]].strip() for r in csv.DictReader(fh)}


def fetch_cna(cohort: str, filename: str, keep: set[str] | None) -> pd.DataFrame:
    """Stream a datahub gene-level CNA matrix into a cases x genes frame."""
    url = f"{DATAHUB}/{COHORTS[cohort]['study']}/{filename}"
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {cohort}/{filename}") as bar:
        for chunk in resp.iter_content(1 << 20):
            buf.write(chunk)
            bar.update(len(chunk))
    buf.seek(0)

    df = pd.read_csv(buf, sep="\t", low_memory=False)
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in df.columns])
    df = df.dropna(subset=["Hugo_Symbol"])
    # A handful of Hugo symbols appear twice; averaging two GISTIC calls would
    # invent a non-integer state, so keep the first and say how many were dropped.
    dupes = df["Hugo_Symbol"].duplicated().sum()
    if dupes:
        print(f"    dropped {dupes} duplicate gene symbols")
        df = df[~df["Hugo_Symbol"].duplicated()]
    df = df.set_index("Hugo_Symbol").T
    df.index.name = "sample_id"

    df = df[[is_primary_tumour(s, cohort) for s in df.index]]
    df.insert(0, "case_id", [to_case_id(s, cohort) for s in df.index])
    before = len(df)
    df = df[~df["case_id"].duplicated()]
    if before != len(df):
        print(f"    collapsed {before - len(df)} extra samples to one per case")
    df = df.set_index("case_id").drop(columns=[], errors="ignore")

    if keep is not None:
        hit = df.index.isin(keep)
        print(f"    {hit.sum()}/{len(keep)} cohort cases have CNA "
              f"({len(df) - hit.sum()} non-cohort cases dropped)")
        df = df[hit]
    return df


def fetch_mutations(cohort: str, keep: set[str] | None) -> pd.DataFrame:
    """Page the whole mutation profile out of the REST API into a tidy frame."""
    cfg = COHORTS[cohort]
    url = f"{CBIOPORTAL}/molecular-profiles/{cfg['mut_profile']}/mutations/fetch"
    body = {"sampleListId": cfg["sample_list"]}

    meta = requests.post(url, json=body, params={"projection": "META"}, timeout=300)
    meta.raise_for_status()
    total = int(meta.headers.get("total-count", 0))

    rows, page, size = [], 0, 50_000
    with tqdm(total=total, unit="mut", desc=f"  {cohort}/mutations") as bar:
        while True:
            resp = requests.post(
                url, json=body,
                params={"projection": "DETAILED", "pageSize": size, "pageNumber": page},
                timeout=600,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for m in batch:
                gene = (m.get("gene") or {}).get("hugoGeneSymbol")
                rows.append({
                    "sample_id": m["sampleId"],
                    "hugo_symbol": gene or m.get("entrezGeneId"),
                    "variant_classification": m.get("mutationType"),
                    "protein_change": m.get("proteinChange"),
                    "chromosome": m.get("chr"),
                    "start_position": m.get("startPosition"),
                })
            bar.update(len(batch))
            page += 1
            if len(batch) < size:
                break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[[is_primary_tumour(s, cohort) for s in df["sample_id"]]]
    df.insert(0, "case_id", [to_case_id(s, cohort) for s in df["sample_id"]])
    if keep is not None:
        df = df[df["case_id"].isin(keep)]
        print(f"    {df['case_id'].nunique()}/{len(keep)} cohort cases have mutations")
    return df


def gene_arm_map(cache: Path) -> dict[str, str]:
    """Hugo symbol -> chromosome arm ('17q'), from UCSC refGene placed into cytoBand intervals."""
    cache.mkdir(parents=True, exist_ok=True)
    hit = cache / "gene_arm_hg38.csv"
    if hit.exists():
        got = pd.read_csv(hit)
        return dict(zip(got["gene"], got["arm"]))

    ucsc = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database"
    bands = pd.read_csv(f"{ucsc}/cytoBand.txt.gz", sep="\t", compression="gzip",
                        names=["chrom", "start", "end", "band", "stain"])
    genes = pd.read_csv(f"{ucsc}/refGene.txt.gz", sep="\t", compression="gzip", header=None,
                        usecols=[2, 4, 5, 12], names=["chrom", "tx_start", "tx_end", "gene"])

    # Autosomes plus X; chrY is excluded because copy number there is confounded by sex and
    # this cohort is 99% female. Unplaced/alt contigs carry no arm.
    ok = {f"chr{c}" for c in list(range(1, 23)) + ["X"]}
    bands, genes = bands[bands["chrom"].isin(ok)], genes[genes["chrom"].isin(ok)]
    bands = bands[bands["band"].str[0].isin(["p", "q"])].copy()
    bands["arm"] = bands["chrom"].str.removeprefix("chr") + bands["band"].str[0]

    # A gene can have several refGene transcripts; collapse to one span first so the midpoint
    # is stable, then place that midpoint in a band interval.
    span = genes.groupby(["gene", "chrom"], as_index=False).agg(
        tx_start=("tx_start", "min"), tx_end=("tx_end", "max"))
    span["mid"] = (span["tx_start"] + span["tx_end"]) // 2

    out = {}
    for chrom, chunk in span.groupby("chrom"):
        bb = bands[bands["chrom"] == chrom].sort_values("start")
        if bb.empty:
            continue
        idx = pd.IntervalIndex.from_arrays(bb["start"], bb["end"], closed="left")
        pos = idx.get_indexer(chunk["mid"])
        for gene, i in zip(chunk["gene"], pos):
            if i >= 0:
                out[gene] = bb["arm"].iloc[i]

    pd.DataFrame({"gene": list(out), "arm": list(out.values())}).to_csv(hit, index=False)
    print(f"    built gene->arm map for {len(out)} genes -> {hit.relative_to(REPO)}")
    return out


def arm_level(log2: pd.DataFrame, gene2arm: dict[str, str]) -> pd.DataFrame:
    """Cases x 39 arms, median gene-level log2 within each arm.

    Median rather than mean: a handful of focal high-amplitude amplicons (ERBB2, MYC) would
    otherwise drag a whole arm's value and reintroduce exactly the focal signal this
    representation is meant to exclude.
    """
    arms = pd.Series({g: gene2arm[g] for g in log2.columns if g in gene2arm})
    usable = log2[arms.index]
    out = usable.T.groupby(arms).median().T
    keep = [a for a in ARMS if a in out.columns]
    return out[keep]


def validate_arms(derived: pd.DataFrame, cache: Path) -> None:
    """Score the derived arm values against TCGA's official Gain/Loss/Unchanged calls."""
    url = f"{DATAHUB}/{COHORTS['tcga']['study']}/data_armlevel_cna.txt"
    official = pd.read_csv(url, sep="\t", low_memory=False)
    official = official.set_index("NAME").drop(columns=["ENTITY_STABLE_ID", "DESCRIPTION"]).T
    official.index = [to_case_id(s, "tcga") for s in official.index]
    official = official[~official.index.duplicated()]

    shared_cases = derived.index.intersection(official.index)
    shared_arms = [a for a in derived.columns if a in official.columns]
    print(f"  validating {len(shared_cases)} cases x {len(shared_arms)} arms "
          f"against TCGA's official calls")

    rows = []
    for arm in shared_arms:
        truth = official.loc[shared_cases, arm]
        mine = derived.loc[shared_cases, arm]
        for label in ("Gain", "Loss"):
            m = truth == label
            if m.sum() >= 10:
                rows.append({"arm": arm, "call": label, "n": int(m.sum()),
                             "median_derived": round(float(mine[m].median()), 3)})
        m = truth == "Unchanged"
        if m.sum() >= 10:
            rows.append({"arm": arm, "call": "Unchanged", "n": int(m.sum()),
                         "median_derived": round(float(mine[m].median()), 3)})

    agree = pd.DataFrame(rows)
    if agree.empty:
        print("  ! no overlapping calls to score")
        return
    summary = agree.groupby("call")["median_derived"].describe()[["count", "mean", "min", "max"]]
    print(summary.round(3).to_string())
    gain, loss = summary.loc["Gain", "mean"], summary.loc["Loss", "mean"]
    unch = summary.loc["Unchanged", "mean"] if "Unchanged" in summary.index else float("nan")
    verdict = "SEPARATES" if gain > unch > loss else "DOES NOT SEPARATE"
    print(f"  derived median log2 by official call: Gain {gain:.3f} > "
          f"Unchanged {unch:.3f} > Loss {loss:.3f}  -> {verdict}")


def mutation_matrix(tidy: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Binary case x gene matrix over non-silent variants only."""
    coding = tidy[tidy["variant_classification"].isin(CODING)]
    hit = coding[coding["hugo_symbol"].isin(genes)]
    mat = (
        hit.assign(mutated=1)
        .pivot_table(index="case_id", columns="hugo_symbol", values="mutated",
                     aggfunc="max", fill_value=0)
        .reindex(columns=genes, fill_value=0)
    )
    return mat.reindex(sorted(set(tidy["case_id"])), fill_value=0).astype("int8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["cna", "mutations", "all"], default="all")
    ap.add_argument("--representation", choices=["gistic", "log2", "arm", "all"], default="all",
                    help="copy-number feature space; 'arm' is the one a cheap sWGS assay can "
                         "actually produce (see module docstring)")
    ap.add_argument("--validate-arms", action="store_true",
                    help="score the derived arm values against TCGA's official arm calls")
    ap.add_argument("--all-cases", action="store_true",
                    help="keep every case in the study, not just those with WSI features")
    ap.add_argument("--top-mutated", type=int, default=50,
                    help="genes in the binary mutation matrix, by TCGA frequency")
    ap.add_argument("--out", type=Path, default=REPO / ".datasets/cnv")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    keep = {c: (None if args.all_cases else wsi_cases(c)) for c in COHORTS}

    want = {"gistic", "log2", "arm"} if args.representation == "all" else {args.representation}
    # arm-level is derived from log2, so asking for arms implies fetching log2.
    need_log2 = bool(want & {"log2", "arm"})

    if args.what in ("cna", "all"):
        shared = None
        if "gistic" in want:
            print("copy number — GISTIC discrete (gene level)")
            cna = {c: fetch_cna(c, "data_cna.txt", keep[c]) for c in COHORTS}
            shared = sorted(set(cna["tcga"].columns) & set(cna["cptac"].columns))
            print(f"  {len(cna['tcga'].columns)} TCGA genes, {len(cna['cptac'].columns)} CPTAC "
                  f"-> {len(shared)} shared")
            for c, df in cna.items():
                path = args.out / f"{c}_brca_cna_gistic.csv"
                df[shared].to_csv(path)
                print(f"  wrote {path.relative_to(REPO)}  {df.shape[0]} x {len(shared)}")

        if need_log2:
            print("copy number — continuous log2 (gene level)")
            log2 = {c: fetch_cna(c, "data_log2_cna.txt", keep[c]) for c in COHORTS}
            cols = sorted(set(log2["tcga"].columns) & set(log2["cptac"].columns))
            print(f"  {len(log2['tcga'].columns)} TCGA genes, {len(log2['cptac'].columns)} CPTAC "
                  f"-> {len(cols)} shared")
            log2 = {c: df[cols] for c, df in log2.items()}
            if "log2" in want:
                for c, df in log2.items():
                    path = args.out / f"{c}_brca_cna_log2.csv"
                    df.to_csv(path)
                    print(f"  wrote {path.relative_to(REPO)}  {df.shape[0]} x {df.shape[1]}")

            if "arm" in want:
                print("copy number — arm level (derived; the sWGS-reachable representation)")
                gene2arm = gene_arm_map(args.out / "reference")
                for c, df in log2.items():
                    arms = arm_level(df, gene2arm)
                    path = args.out / f"{c}_brca_cna_arm.csv"
                    arms.to_csv(path)
                    print(f"  wrote {path.relative_to(REPO)}  {arms.shape[0]} x {arms.shape[1]}")
                    if c == "tcga" and args.validate_arms:
                        validate_arms(arms, args.out / "reference")

    if args.what in ("mutations", "all"):
        print("somatic mutations")
        tidy = {c: fetch_mutations(c, keep[c]) for c in COHORTS}
        for c, df in tidy.items():
            path = args.out / f"{c}_brca_mutations.csv"
            df.to_csv(path, index=False)
            print(f"  wrote {path.relative_to(REPO)}  {len(df)} calls, "
                  f"{df['case_id'].nunique()} cases")

        # Gene panel chosen on TCGA alone; picking it on the pooled data would leak
        # the external cohort into a modelling decision.
        coding = tidy["tcga"][tidy["tcga"]["variant_classification"].isin(CODING)]
        genes = (coding.drop_duplicates(["case_id", "hugo_symbol"])["hugo_symbol"]
                 .value_counts().head(args.top_mutated).index.tolist())
        for c, df in tidy.items():
            mat = mutation_matrix(df, genes)
            path = args.out / f"{c}_brca_mutation_matrix.csv"
            mat.to_csv(path)
            rate = mat.values.mean()
            print(f"  wrote {path.relative_to(REPO)}  {mat.shape[0]} x {mat.shape[1]}"
                  f"  (mean mutation rate {rate:.3f})")
        print("  panel:", ", ".join(genes[:12]), "…")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
