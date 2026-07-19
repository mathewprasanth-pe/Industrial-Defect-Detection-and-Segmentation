"""
MVTec AD dataset loader.

Expects the standard MVTec AD directory layout:

data/mvtec_ad/
├── <category>/
│   ├── train/good/                  # normal images only
│   ├── test/good/                   # normal test images
│   ├── test/<defect_type>/          # one folder per defect type
│   └── ground_truth/<defect_type>/  # pixel masks, matched by filename (+_mask suffix)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


@dataclass
class MVTecSample:
    image_path: Path
    mask_path: Path | None   # None for normal images
    label: int                # 0 = normal, 1 = anomalous
    defect_type: str          # "good" or e.g. "scratch"


class MVTecDataset(Dataset):
    """
    split="train" -> only normal ("good") images, used to build the
                      memory bank (PatchCore) or fit Gaussians (PaDiM).
    split="test"  -> normal + all defect types, with masks where available.
    """

    def __init__(self, root: str | Path, category: str, split: str = "train", transform=None):
        if category not in MVTEC_CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Expected one of {MVTEC_CATEGORIES}")
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.samples: list[MVTecSample] = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found for category='{category}', split='{split}' "
                f"under {self.root}. Check that the dataset was extracted correctly."
            )

    def _build_samples(self) -> list[MVTecSample]:
        cat_dir = self.root / self.category
        samples: list[MVTecSample] = []

        if self.split == "train":
            good_dir = cat_dir / "train" / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                samples.append(MVTecSample(img_path, None, 0, "good"))
            return samples

        # split == "test": iterate every subfolder under test/
        test_dir = cat_dir / "test"
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            is_good = defect_type == "good"

            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if not is_good:
                    mask_path = (
                        cat_dir / "ground_truth" / defect_type / f"{img_path.stem}_mask.png"
                    )
                    if not mask_path.exists():
                        mask_path = None  # tolerate missing masks rather than crash

                samples.append(
                    MVTecSample(
                        image_path=img_path,
                        mask_path=mask_path,
                        label=0 if is_good else 1,
                        defect_type=defect_type,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if sample.mask_path is not None:
            mask = Image.open(sample.mask_path).convert("L")
            mask = np.array(mask) > 0  # binary mask
        else:
            # normal images / missing masks -> all-zero mask at original image size
            w, h = Image.open(sample.image_path).size
            mask = np.zeros((h, w), dtype=bool)

        return {
            "image": image,
            "mask": mask,
            "label": sample.label,
            "defect_type": sample.defect_type,
            "path": str(sample.image_path),
        }