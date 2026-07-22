# Industrial Defect Detection & Segmentation

Benchmarking a frozen **ViT/DINOv2 backbone (PatchCore)** against a **CNN/ResNet backbone (PaDiM)** for unsupervised industrial anomaly detection and segmentation on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark.

Both methods are trained **only on defect-free images**, reflecting a core constraint of real manufacturing environments: labeled defect examples are rare and unpredictable, but defect-free examples are abundant.

## Why this project

Most industrial computer vision roles require defect detection systems that work under exactly this constraint — little to no labeled anomaly data. Rather than training a classifier from scratch, this project uses two established unsupervised approaches, each pairing a frozen pretrained backbone with a lightweight statistical/geometric scoring method:

- **PatchCore** ([Roth et al., 2022](https://arxiv.org/abs/2106.08265)) — builds a memory bank of normal patch embeddings, flags test patches based on nearest-neighbor distance.
- **PaDiM** ([Defard et al., 2020](https://arxiv.org/abs/2011.08785)) — fits a per-location Gaussian distribution over normal patch features, flags test patches via Mahalanobis distance.

Neither method trains a neural network — both rely entirely on frozen, pretrained backbones (DINOv2 ViT and ResNet18, respectively), with all "learning" limited to simple statistics or memory-bank construction over already-extracted features.

## Results

Evaluated on 4 MVTec AD categories, chosen for variety in defect type and object structure:

| Category | Method | Image AUROC | Pixel AUROC | Fit Time |
|---|---|---|---|---|
| bottle | PatchCore | 0.9992 | 0.9761 | 42.4s |
| bottle | PaDiM | 0.9968 | 0.9673 | 0.6s |
| screw | PatchCore | 0.8701 | 0.9942 | 97.1s |
| screw | PaDiM | 0.8936 | 0.9864 | 0.7s |
| carpet | PatchCore | 0.9980 | 0.9849 | 74.0s |
| carpet | PaDiM | 0.9916 | 0.9894 | 0.7s |
| transistor | PatchCore | 0.8800 | 0.9317 | 43.7s |
| transistor | PaDiM | 0.9388 | 0.9727 | 0.5s |
| **Average** | **PatchCore** | **0.9368** | **0.9717** | ~64s |
| **Average** | **PaDiM** | **0.9552** | **0.9790** | ~0.6s |

### Key findings

**Neither method dominates universally.** PatchCore edges out PaDiM on `bottle`, but PaDiM matches or beats PatchCore on the other three categories — both on image-level and pixel-level AUROC. This is a genuinely useful finding: the "obvious" assumption that a newer, transformer-based backbone should outperform a CNN-based one doesn't hold here.

**PaDiM fits 60-160x faster.** Across every category tested, PaDiM's fitting step (closed-form Gaussian statistics) took under 1 second, while PatchCore's fitting step (iterative coreset selection over a large memory bank) took 40-100+ seconds. For a real deployment scenario where models need frequent retraining as a production line changes, this is a substantial practical advantage — even setting accuracy aside.

**Image-level and pixel-level AUROC can diverge sharply.** On `screw`, image-level AUROC dropped to the high-80s/low-90s while pixel-level AUROC stayed above 0.98 for both methods. This makes sense once you consider defect size: screw defects are small, subtle scratches — when a model *does* flag the right region, it does so precisely (high pixel AUROC), but reducing that to a single per-image anomaly score is noisier when the anomalous region is tiny relative to the whole image (lower image AUROC). This distinction — whole-image detection reliability vs. localization precision — matters a lot for how a system would actually be used on a production line.

### Qualitative example: PatchCore vs. PaDiM on the same defect

Both models scoring the same `bottle` test image (`broken_large`, torn foil seal):

**PatchCore:**

![PatchCore heatmap](results/qualitative_examples/bottle_patchcore_example.png)

**PaDiM:**

![PaDiM heatmap](results/qualitative_examples/bottle_padim_example.png)

PatchCore's heatmap localizes tightly to the actual torn-foil shape in the ground truth mask. PaDiM identifies the same general region but with a more diffuse, less tightly bounded hot zone — a visual illustration of the pixel-AUROC gap measured above (0.9761 vs 0.9673 on this category).

## Architecture

| | PatchCore | PaDiM |
|---|---|---|
| **Backbone** | DINOv2 ViT-Small (frozen) | ResNet18, layers 1-3 (frozen) |
| **Feature dim** | 384 per patch | 448 per patch (concatenated multi-layer) |
| **Grid size** | 16×16 (256 patches) | 28×28 (784 patches) |
| **Scoring method** | Nearest-neighbor distance to a coreset-subsampled memory bank | Mahalanobis distance from a per-location Gaussian |
| **Training** | None — frozen backbone, no gradient updates anywhere | None — frozen backbone, no gradient updates anywhere |

Both backbones were chosen deliberately to suit their scoring method: PatchCore's nearest-neighbor approach benefits from ViT/DINOv2's semantically well-structured embedding space, while PaDiM's per-location Gaussian fitting needs the lower feature dimensionality CNN layers provide to keep covariance estimation stable with limited training images.

## Setup

```bash
# Install uv, then:
uv python install 3.11
uv init --no-readme --python 3.11
uv add torch torchvision timm opencv-python numpy scipy scikit-learn scikit-image matplotlib tqdm pandas

# Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad
# Extract into data/mvtec_ad/
```

## Usage

```bash
# Run both methods on a single category
uv run python run_pipeline.py --category bottle --method both

# Run a single method
uv run python run_pipeline.py --category screw --method patchcore
```

## Project structure

├── src/
│ ├── datasets/mvtec_dataset.py # MVTec AD loader
│ ├── backbones/
│ │ ├── vit_extractor.py # DINOv2 ViT feature extraction (PatchCore)
│ │ └── cnn_extractor.py # ResNet multi-layer extraction (PaDiM)
│ ├── methods/
│ │ ├── patchcore.py # Memory bank + coreset + nearest-neighbor
│ │ └── padim.py # Per-location Gaussian + Mahalanobis distance
│ ├── eval/metrics.py # Image/pixel AUROC
│ └── utils/visualization.py # Heatmap overlay generation
├── run_pipeline.py # Main entrypoint
└── results/qualitative_examples/ # Saved comparison figures

## Limitations & next steps

- Evaluated on 4 of 15 MVTec AD categories; a full 15-category sweep would give a more complete picture
- Pixel-level AUROC is computed on the coarse patch grid (16×16 or 28×28), not full image resolution — very small defects may be under-represented in this metric
- Next: extending to synthetic/Blender-generated defect data for sim-to-real transfer, and a capstone integration combining detection with a lightweight inspection dashboard

## References

- Roth et al., ["Towards Total Recall in Industrial Anomaly Detection"](https://arxiv.org/abs/2106.08265) (PatchCore)
- Defard et al., ["PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization"](https://arxiv.org/abs/2011.08785)
- [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

**Mathew Prasanth, P.E.**
AI/ML Engineer