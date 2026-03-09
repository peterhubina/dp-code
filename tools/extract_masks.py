import json
import os
import openslide
from PIL import Image, ImageDraw
import numpy as np
from glob import glob
from pathlib import Path

CARCINOMA_CLASSES = {'in situ carcinoma', 'infiltrant carcinoma'}

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

# Usage example
if __name__ == '__main__':
    # Single file
    # geojson_to_mask('path/to/slide1.svs', 'path/to/slide1.geojson', mask_level=6)
    
    # Batch process
    batch_convert(
        wsidir='/workspace/dp-code/.datasets/PKG - HistologyHSI-BC-Recurrence/01_01_Histological_Images',
        geojsondir='/workspace/dp-code/.datasets/PKG - HistologyHSI-BC-Recurrence/01_02_Tissue_Annotations',
        maskdir='/workspace/dp-code/.scratch/datasets/masks',
        mask_level=6
    )
