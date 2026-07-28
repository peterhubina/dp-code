#!/usr/bin/env python3
"""Download CPTAC whole-slide images from TCIA's PathDB over plain HTTPS.

TCIA's recommended route for pathology collections is the Aspera Faspex browser
plugin, which needs a GUI. PathDB also exposes every slide as a direct file URL,
so a headless container can pull the collection without Aspera:

    https://pathdb.cancerimagingarchive.net/collections?_format=json
    https://pathdb.cancerimagingarchive.net/listofimages/<collectionId>?_format=json&page=<n>

The file endpoints honour Range requests, so downloads resume after an interrupt
and a slide is only re-fetched when its local size does not match the server's.

    python tools/download_cptac.py --dry-run
    python tools/download_cptac.py --workers 8
    python tools/download_cptac.py --collection CPTAC-LUAD --output .datasets/cptac-luad
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://pathdb.cancerimagingarchive.net"
CHUNK_SIZE = 1 << 20


def resolve_collection_id(name):
    response = requests.get(f"{BASE_URL}/collections?_format=json", timeout=120)
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


def fetch_manifest(collection_id):
    """Page through the collection and return one record per slide."""
    slides, page = [], 0
    while True:
        response = requests.get(
            f"{BASE_URL}/listofimages/{collection_id}?_format=json&page={page}",
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
                    "subject_id": item["clinicaltrialsubjectid"][0]["value"],
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
        return "skipped", expected
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="CPTAC-BRCA")
    parser.add_argument("--output", type=Path, default=Path(".datasets/cptac-brca"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--limit", type=int, default=0, help="download only the first N slides"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the manifest and report totals without downloading",
    )
    args = parser.parse_args()

    collection_id = resolve_collection_id(args.collection)
    print(f"{args.collection} -> PathDB collection {collection_id}")

    slides = fetch_manifest(collection_id)
    if args.limit:
        slides = slides[: args.limit]
    subjects = {s["subject_id"] for s in slides}
    print(f"{len(slides)} slides across {len(subjects)} subjects")

    wsi_dir = args.output / "wsi"
    wsi_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slides[0]))
        writer.writeheader()
        writer.writerows(slides)
    print(f"manifest -> {manifest_path}")

    session = requests.Session()
    if args.dry_run:
        with ThreadPoolExecutor(args.workers * 2) as pool:
            sizes = list(
                tqdm(
                    pool.map(lambda s: remote_size(s["url"], session), slides),
                    total=len(slides),
                    desc="sizing",
                )
            )
        on_disk = sum(
            (wsi_dir / s["filename"]).stat().st_size
            for s in slides
            if (wsi_dir / s["filename"]).exists()
        )
        print(f"collection size: {sum(sizes) / 1e9:.1f} GB")
        print(f"already on disk:  {on_disk / 1e9:.1f} GB")
        print(f"remaining:        {(sum(sizes) - on_disk) / 1e9:.1f} GB")
        return

    tallies = {"downloaded": 0, "resumed": 0, "skipped": 0}
    transferred = 0
    failures = []

    progress = tqdm(total=len(slides), desc=args.collection, unit="slide")
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(download_slide, s, wsi_dir, session): s for s in slides
        }
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

    print(
        f"downloaded {tallies['downloaded']}, resumed {tallies['resumed']}, "
        f"already complete {tallies['skipped']} — {transferred / 1e9:.1f} GB transferred"
    )
    if failures:
        print(f"\n{len(failures)} slides failed (re-run to retry):", file=sys.stderr)
        for filename, exc in failures[:20]:
            print(f"  {filename}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
