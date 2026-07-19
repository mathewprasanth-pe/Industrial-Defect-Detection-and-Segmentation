"""
PatchCore: memory-bank + nearest-neighbor anomaly detection.

Pipeline:
1. fit()   - extract patch features from all "good" training images,
             pool them into one big memory bank, then shrink it via
             greedy coreset subsampling (so inference stays fast).
2. score() - for a test image, extract patch features, find each
             patch's nearest neighbor in the memory bank, and use
             that distance as the anomaly score.
"""

import numpy as np
import torch


class PatchCore:
    def __init__(self, coreset_ratio: float = 0.1, device: torch.device | None = None):
        """
        coreset_ratio: fraction of all training patches to keep in the
                       memory bank after coreset subsampling (0.1 = keep 10%).
        """
        self.coreset_ratio = coreset_ratio
        self.device = device or torch.device("cpu")
        self.memory_bank: torch.Tensor | None = None   # (M, D) after fitting

    # ------------------------------------------------------------------
    # Building the memory bank
    # ------------------------------------------------------------------

    def fit(self, train_features: torch.Tensor) -> None:
        """
        train_features: (N, P, D) - N training images, P patches each, D-dim embeddings.
        Flattens to (N*P, D), then subsamples down to a coreset.
        """
        n, p, d = train_features.shape
        all_patches = train_features.reshape(n * p, d)  # (N*P, D)

        coreset_size = max(1, int(all_patches.shape[0] * self.coreset_ratio))
        self.memory_bank = self._greedy_coreset(all_patches, coreset_size)

    def _greedy_coreset(self, patches: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Greedy k-center subsampling: repeatedly pick the patch that is
        FARTHEST from everything already chosen. This keeps the memory
        bank diverse (covering all "kinds" of normal patches) rather
        than redundant, so a smaller bank still represents normal
        patches well.
        """
        patches = patches.to(self.device)
        n = patches.shape[0]

        # Start from a random patch.
        first_idx = torch.randint(0, n, (1,)).item()
        selected_idx = [first_idx]

        # min_dist[i] = distance from patch i to its closest already-selected patch.
        min_dist = torch.cdist(patches, patches[first_idx:first_idx + 1]).squeeze(1)

        for _ in range(target_size - 1):
            next_idx = torch.argmax(min_dist).item()
            selected_idx.append(next_idx)

            new_dist = torch.cdist(patches, patches[next_idx:next_idx + 1]).squeeze(1)
            min_dist = torch.minimum(min_dist, new_dist)

        return patches[selected_idx].cpu()

    # ------------------------------------------------------------------
    # Scoring new images
    # ------------------------------------------------------------------

    def score(self, test_features: torch.Tensor, grid_size: int) -> dict:
        """
        test_features: (B, P, D) - a batch of test images' patch features.
        grid_size: the side length of the patch grid (e.g. 16 for a 16x16 grid),
                   from ViTPatchExtractor.feature_map_size().

        Returns per-image anomaly score and a per-image (grid_size x grid_size)
        heatmap of patch-level anomaly scores.
        """
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before score().")

        b, p, d = test_features.shape
        memory_bank = self.memory_bank.to(self.device)

        image_scores = []
        heatmaps = []

        for i in range(b):
            patches = test_features[i].to(self.device)          # (P, D)
            dists = torch.cdist(patches, memory_bank)             # (P, M)
            nearest_dist, _ = dists.min(dim=1)                    # (P,) - distance to closest normal patch

            heatmap = nearest_dist.reshape(grid_size, grid_size).cpu().numpy()
            image_score = nearest_dist.max().item()               # worst patch = image-level anomaly score

            heatmaps.append(heatmap)
            image_scores.append(image_score)

        return {
            "image_scores": np.array(image_scores),   # (B,)
            "heatmaps": np.stack(heatmaps),            # (B, grid_size, grid_size)
        }