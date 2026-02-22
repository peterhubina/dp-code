import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


LABEL_MAP = {
    "in situ carcinoma": 0,
    "infiltrant carcinoma": 1,
}


class PatchDataset(Dataset):
    """Dataset loading PNG patches from one or more slide directories.

    Each slide directory is expected to contain a ``metadata.csv`` with at least
    columns ``path`` (relative to the slide dir) and ``label``.

    Only samples whose label appears in *classes* are kept.
    """

    def __init__(self, patch_dirs, classes=None, transform=None, label_map=None):
        """
        Args:
            patch_dirs: list of root directories (one per slide), each containing
                        a ``metadata.csv``.
            classes: optional list of label strings to keep (default: all in LABEL_MAP).
            transform: torchvision transform applied to PIL images.
            label_map: dict mapping label string -> int (default: LABEL_MAP).
        """
        self.transform = transform
        self.label_map = label_map or LABEL_MAP
        if classes is None:
            classes = list(self.label_map.keys())

        self.samples = []  # list of (image_path, label_int, slide_id)

        for patch_dir in patch_dirs:
            csv_path = os.path.join(patch_dir, "metadata.csv")
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df = df[df["label"].isin(classes)]
            for _, row in df.iterrows():
                img_path = os.path.join(patch_dir, row["path"])
                label_int = self.label_map[row["label"]]
                slide_id = str(row["slide_id"])
                self.samples.append((img_path, label_int, slide_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, slide_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, slide_id
