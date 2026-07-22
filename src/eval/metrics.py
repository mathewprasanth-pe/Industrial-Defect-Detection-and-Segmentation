"""
Evaluation metrics for anomaly detection: AUROC (image- and pixel-level).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    scores: (N,) anomaly scores, higher = more anomalous
    labels: (N,) ground truth, 0 = normal, 1 = anomalous
    """
    return roc_auc_score(labels, scores)


def pixel_auroc(heatmaps: np.ndarray, masks: np.ndarray) -> float:
    """
    heatmaps: (N, H, W) per-pixel (or per-patch-grid) anomaly scores
    masks:    (N, H, W) ground truth binary masks, same shape as heatmaps
    Flattens everything into one long vector and computes AUROC across
    every pixel/patch of every image at once.
    """
    return roc_auc_score(masks.flatten().astype(int), heatmaps.flatten())