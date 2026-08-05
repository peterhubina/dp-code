"""GeoJSON tumour annotations -> binary tissue masks for the parked HSI-BC cohort.

Part of the dormant HistologyHSI-BC-Recurrence thread, kept because the masks it
produces are the only tumour-region annotations in the project. Nothing in the
CNV + WSI headline chain runs it.

Paths come from `dpcode/conf/paths/default.yaml` (`paths.hsi_bc_root`,
`paths.scratch_root`), overridable per run:

    python tools/extract_masks.py                       # cohort defaults
    python tools/extract_masks.py --wsi_dir ... --geojson_dir ... --mask_dir ...

The previous `__main__` block hardcoded three absolute paths, two of which named
a directory that has never existed in this repository
(`.datasets/PKG - HistologyHSI-BC-Recurrence/`; the real cohort root is
`.datasets/HistologyHSI-BC-Recurrence/`, as `tools/hsi_bc/run_pipeline.sh:20` and
`tools/hsi_bc/prepare_manifest.py:32` both use). The third,
`.scratch/datasets/masks`, named a directory nothing else in the repository
refers to; masks now default under `paths.scratch_root` beside the rest of the
cohort's derived files.
"""

import argparse
import json
import os
import openslide
from PIL import Image, ImageDraw
import numpy as np
from glob import glob
from pathlib import Path

CARCINOMA_CLASSES = {'in situ carcinoma', 'infiltrant carcinoma'}

#: Subdirectories of the cohort root, as published by TCIA and as used by
#: `tools/hsi_bc/`.
IMAGE_SUBDIR = '01_01_Histological_Images'
ANNOTATION_SUBDIR = '01_02_Tissue_Annotations'

def geojson_to_mask(wsi_path, geojson_path, mask_level=6, output_path=None):
    """
    Convert GeoJSON annotations to binary mask at specified pyramid level.
    Only includes 'in situ carcinoma' and 'infiltrant carcinoma' annotations;
    'normal tissue' is excluded.

    Args:
        wsi_path: Path to WSI file (.mrxs, .svs, .ndpi, etc.)
        geojson_path: Path to GeoJSON file with polygons
        mask_level: Downsample level for mask (e.g., 6-8 for CLAM segmentation)
        output_path: Output mask path (default: same basename + '_mask.png')
    """
    # Open WSI
    wsi = openslide.OpenSlide(wsi_path)
    width, height = wsi.level_dimensions[mask_level]

    # Create blank mask
    mask = Image.new('L', (width, height), 0)  # Grayscale, black background
    draw = ImageDraw.Draw(mask)

    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    # Process features/polygons — skip anything that isn't a carcinoma class
    if 'features' in data:
        for feature in data['features']:
            classification = feature.get('properties', {}).get('classification', {}).get('name', '')
            if classification not in CARCINOMA_CLASSES:
                continue

            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]  # Outer ring
            elif feature['geometry']['type'] == 'MultiPolygon':
                coords = feature['geometry']['coordinates'][0][0]  # First polygon outer ring
            else:
                continue

            # Scale coordinates to mask level (assuming GeoJSON coords are in level 0)
            downsample = wsi.level_downsamples[mask_level]
            scaled_coords = [(int(x/downsample), int(y/downsample)) for x, y in coords]

            # Draw filled polygon (255 = white foreground)
            draw.polygon(scaled_coords, fill=255)
    
    if output_path is None:
        output_path = Path(wsi_path).stem + '_mask.png'
    
    mask.save(output_path, 'PNG')
    print(f'Saved mask: {output_path}')
    wsi.close()
    return output_path

def batch_convert(wsidir, geojsondir, maskdir, mask_level=6):
    """
    Batch convert all WSIs and matching GeoJSONs.
    
    Assumes: slide1.svs -> slide1.geojson
    """
    os.makedirs(maskdir, exist_ok=True)
    
    wsi_files = sorted(glob(os.path.join(wsidir, '*.svs')) +
                      glob(os.path.join(wsidir, '*.ndpi')) +
                      glob(os.path.join(wsidir, '*.tif')) +
                      glob(os.path.join(wsidir, '*.mrxs')))
    
    for wsi_path in wsi_files:
        base = Path(wsi_path).stem
        geojson_path = os.path.join(geojsondir, f'{base}.geojson')
        
        if os.path.exists(geojson_path):
            mask_path = os.path.join(maskdir, f'{base}_mask.png')
            if not os.path.exists(mask_path):
                print(f'Processing {base}...')
                geojson_to_mask(wsi_path, geojson_path, mask_level, mask_path)
        else:
            print(f'No GeoJSON found for {base}')

def parse_args(argv=None):
    from dpcode.paths import resolve_paths

    paths = resolve_paths()
    cohort_root = Path(paths['hsi_bc_root'])

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--wsi_dir', default=str(cohort_root / IMAGE_SUBDIR))
    parser.add_argument('--geojson_dir', default=str(cohort_root / ANNOTATION_SUBDIR))
    parser.add_argument(
        '--mask_dir', default=str(Path(paths['scratch_root']) / 'hsi_bc' / 'masks')
    )
    parser.add_argument(
        '--mask_level',
        type=int,
        default=6,
        help='WSI pyramid level the mask is rasterised at (6-8 suits CLAM segmentation).',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    for label, directory in (('--wsi_dir', args.wsi_dir), ('--geojson_dir', args.geojson_dir)):
        if not os.path.isdir(directory):
            raise SystemExit(
                f'{label} does not exist: {directory}\n'
                'The HistologyHSI-BC-Recurrence cohort is parked and its image and '
                'annotation directories are not part of this clone. Set '
                'DP_HSI_BC_ROOT, or pass --wsi_dir/--geojson_dir explicitly.'
            )
    batch_convert(
        wsidir=args.wsi_dir,
        geojsondir=args.geojson_dir,
        maskdir=args.mask_dir,
        mask_level=args.mask_level,
    )
