"""
Heatmap visualization: overlay per-patch anomaly scores on the original image.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Scale a heatmap to the 0-1 range for consistent color mapping."""
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - h_min) / (h_max - h_min)


def upsample_heatmap(heatmap: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    heatmap: (grid_size, grid_size), e.g. (16, 16) or (28, 28)
    target_size: (width, height) of the original image, e.g. (900, 900)
    Uses PIL's bilinear resize to smoothly upsample the coarse patch grid
    back to full image resolution.
    """
    img = Image.fromarray((heatmap * 255).astype(np.uint8))
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img) / 255.0


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """
    original_image: a PIL Image (RGB), the raw, unnormalized photo
    heatmap: (grid_size, grid_size) raw anomaly scores
    Returns a new PIL Image with the heatmap overlaid in color.
    """
    heatmap_norm = normalize_heatmap(heatmap)
    heatmap_full = upsample_heatmap(heatmap_norm, original_image.size)

    cmap = matplotlib.colormaps[colormap]
    heatmap_colored = cmap(heatmap_full)[:, :, :3]  # drop alpha channel from colormap
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored).convert("RGB")

    blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)
    return blended


def save_comparison_figure(
    original_image: Image.Image,
    mask: np.ndarray,
    heatmap: np.ndarray,
    save_path: str | Path,
    title: str = "",
) -> None:
    """
    Saves a 3-panel figure: original image | ground-truth mask | predicted heatmap overlay.
    This is the standard qualitative figure layout used in anomaly detection papers.
    """
    overlay = overlay_heatmap(original_image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Predicted Anomaly Heatmap")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)