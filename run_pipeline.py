"""
Full evaluation pipeline: run PatchCore and/or PaDiM on a full MVTec AD
category and report image-level and pixel-level AUROC.

Usage:
    uv run python run_pipeline.py --category bottle --method patchcore
    uv run python run_pipeline.py --category bottle --method padim
    uv run python run_pipeline.py --category bottle --method both
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image

from src.datasets.mvtec_dataset import MVTecDataset
from src.backbones.vit_extractor import ViTPatchExtractor, get_vit_transform
from src.backbones.cnn_extractor import CNNPatchExtractor, get_cnn_transform
from src.methods.patchcore import PatchCore
from src.methods.padim import PaDiM
from src.eval.metrics import image_auroc, pixel_auroc


def load_all_images(dataset, desc=""):
    """Loads every image (and mask) in a dataset into memory as stacked tensors."""
    images, labels, masks = [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        images.append(sample["image"])
        labels.append(sample["label"])
        masks.append(sample["mask"])
        if (i + 1) % 50 == 0:
            print(f"  {desc}: loaded {i + 1}/{len(dataset)}")
    return torch.stack(images), np.array(labels), np.array(masks)


def resize_masks(masks: np.ndarray, grid_size: int) -> np.ndarray:
    """Resize ground-truth masks down to the heatmap's grid resolution for pixel AUROC."""
    resized = np.zeros((masks.shape[0], grid_size, grid_size), dtype=np.float32)
    for i, m in enumerate(masks):
        img = Image.fromarray((m * 255).astype(np.uint8))
        img = img.resize((grid_size, grid_size), Image.BILINEAR)
        resized[i] = np.array(img) / 255.0
    return resized


def run_patchcore(category: str):
    print(f"\n=== PatchCore | {category} ===")
    transform = get_vit_transform()
    extractor = ViTPatchExtractor()
    grid_size = extractor.feature_map_size(224)

    train_ds = MVTecDataset("data/mvtec_ad", category=category, split="train", transform=transform)
    test_ds = MVTecDataset("data/mvtec_ad", category=category, split="test", transform=transform)

    print(f"Loading {len(train_ds)} training images...")
    train_imgs, _, _ = load_all_images(train_ds, "train")

    print(f"Loading {len(test_ds)} test images...")
    test_imgs, test_labels, test_masks = load_all_images(test_ds, "test")

    t0 = time.time()
    print("Extracting training features...")
    train_feats = extractor.extract_batch(train_imgs, batch_size=16)

    print("Fitting PatchCore memory bank...")
    model = PatchCore(coreset_ratio=0.1)
    model.fit(train_feats)
    print(f"Memory bank size: {model.memory_bank.shape}, fit time: {time.time() - t0:.1f}s")

    print("Extracting test features and scoring...")
    test_feats = extractor.extract_batch(test_imgs, batch_size=16)
    result = model.score(test_feats, grid_size=grid_size)

    img_auroc = image_auroc(result["image_scores"], test_labels)
    resized_masks = resize_masks(test_masks, grid_size)
    px_auroc = pixel_auroc(result["heatmaps"], resized_masks)

    print(f"\nImage-level AUROC: {img_auroc:.4f}")
    print(f"Pixel-level AUROC: {px_auroc:.4f}")
    return img_auroc, px_auroc


def run_padim(category: str):
    print(f"\n=== PaDiM | {category} ===")
    transform = get_cnn_transform()
    extractor = CNNPatchExtractor()
    grid_size = 28

    train_ds = MVTecDataset("data/mvtec_ad", category=category, split="train", transform=transform)
    test_ds = MVTecDataset("data/mvtec_ad", category=category, split="test", transform=transform)

    print(f"Loading {len(train_ds)} training images...")
    train_imgs, _, _ = load_all_images(train_ds, "train")

    print(f"Loading {len(test_ds)} test images...")
    test_imgs, test_labels, test_masks = load_all_images(test_ds, "test")

    t0 = time.time()
    print("Extracting training features...")
    train_feats = extractor.extract_batch(train_imgs, grid_size=grid_size, batch_size=16)

    print("Fitting PaDiM Gaussians...")
    model = PaDiM(n_components=100)
    model.fit(train_feats)
    print(f"Fit time: {time.time() - t0:.1f}s")

    print("Extracting test features and scoring...")
    test_feats = extractor.extract_batch(test_imgs, grid_size=grid_size, batch_size=16)
    result = model.score(test_feats, grid_size=grid_size)

    img_auroc = image_auroc(result["image_scores"], test_labels)
    resized_masks = resize_masks(test_masks, grid_size)
    px_auroc = pixel_auroc(result["heatmaps"], resized_masks)

    print(f"\nImage-level AUROC: {img_auroc:.4f}")
    print(f"Pixel-level AUROC: {px_auroc:.4f}")
    return img_auroc, px_auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--method", type=str, choices=["patchcore", "padim", "both"], default="both")
    args = parser.parse_args()

    if args.method in ("patchcore", "both"):
        run_patchcore(args.category)
    if args.method in ("padim", "both"):
        run_padim(args.category)