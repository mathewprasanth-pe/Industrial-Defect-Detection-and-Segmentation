"""
PaDiM: per-location Gaussian modeling + Mahalanobis distance anomaly detection.

Pipeline:
1. fit()   - for each spatial location in the patch grid, fit a
             multivariate Gaussian (mean + covariance) over that
             location's features across all "good" training images.
2. score() - for a test image, compute the Mahalanobis distance between
             each patch's features and its location's fitted Gaussian.
"""

import numpy as np
import torch


class PaDiM:
    def __init__(self, n_components: int | None = None, device: torch.device | None = None):
        """
        n_components: optionally reduce feature dimensionality (via random
                      projection) before fitting Gaussians, since PaDiM's
                      original paper found full-dimensional covariance
                      estimation can be unstable/slow with small training sets.
                      None = use full feature dimension.
        """
        self.n_components = n_components
        self.device = device or torch.device("cpu")

        self.mean: torch.Tensor | None = None          # (P, D)
        self.cov_inv: torch.Tensor | None = None        # (P, D, D)
        self.projection: torch.Tensor | None = None      # (D_full, n_components) or None

    # ------------------------------------------------------------------
    # Fitting the per-location Gaussians
    # ------------------------------------------------------------------

    def fit(self, train_features: torch.Tensor) -> None:
        """
        train_features: (N, P, D) - N training images, P patch locations, D-dim features.
        """
        n, p, d = train_features.shape
        train_features = train_features.to(self.device)

        if self.n_components is not None and self.n_components < d:
            self.projection = torch.randn(d, self.n_components, device=self.device)
            self.projection /= self.projection.norm(dim=0, keepdim=True)
            train_features = torch.einsum("npd,dk->npk", train_features, self.projection)
            d = self.n_components

        # Mean feature vector per patch location, across all training images.
        self.mean = train_features.mean(dim=0)  # (P, D)

        # Covariance matrix per patch location.
        cov = torch.zeros(p, d, d, device=self.device)
        eps = 0.01  # regularization, avoids singular (non-invertible) covariance matrices
        identity = torch.eye(d, device=self.device)

        for loc in range(p):
            centered = train_features[:, loc, :] - self.mean[loc]  # (N, D)
            cov[loc] = (centered.T @ centered) / (n - 1) + eps * identity

        self.cov_inv = torch.linalg.inv(cov)  # (P, D, D)

    # ------------------------------------------------------------------
    # Scoring new images
    # ------------------------------------------------------------------

    def score(self, test_features: torch.Tensor, grid_size: int) -> dict:
        """
        test_features: (B, P, D)
        Returns per-image anomaly score and per-image (grid_size x grid_size) heatmap,
        using Mahalanobis distance at each patch location.
        """
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError("Call fit() before score().")

        b, p, d = test_features.shape
        test_features = test_features.to(self.device)

        if self.projection is not None:
            test_features = torch.einsum("npd,dk->npk", test_features, self.projection)

        image_scores = []
        heatmaps = []

        for i in range(b):
            diff = test_features[i] - self.mean  # (P, D)
            # Mahalanobis distance per location: sqrt(diff^T @ cov_inv @ diff)
            m_dist = torch.einsum("pd,pde,pe->p", diff, self.cov_inv, diff)
            m_dist = torch.sqrt(torch.clamp(m_dist, min=0))  # (P,)

            heatmap = m_dist.reshape(grid_size, grid_size).cpu().numpy()
            image_score = m_dist.max().item()

            heatmaps.append(heatmap)
            image_scores.append(image_score)

        return {
            "image_scores": np.array(image_scores),
            "heatmaps": np.stack(heatmaps),
        }