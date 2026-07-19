"""
ViT/DINO feature extractor for PatchCore.

PatchCore needs patch-level embeddings (not a single pooled vector),
so we pull the token embeddings from a chosen transformer block —
these correspond to spatial patches across the image.
"""

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

# DINOv2 ViT-Small — a good default: strong features, manageable size for a laptop GPU.
DEFAULT_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_vit_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ViTPatchExtractor:
    """
    Wraps a frozen pretrained ViT and exposes per-patch feature maps
    suitable for PatchCore's memory bank.

    Output shape per image: (num_patches, embed_dim)
    e.g. for a 224x224 image with patch_size=14 -> 16x16 = 256 patches.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: torch.device | None = None):
        self.device = device or get_device()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.model.eval()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False  # frozen — no fine-tuning

        self.patch_size = self.model.patch_embed.patch_size[0]
        self.embed_dim = self.model.embed_dim

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W) already normalized via get_vit_transform
        returns: (B, num_patches, embed_dim) — CLS token excluded
        """
        images = images.to(self.device)
        tokens = self.model.forward_features(images)  # (B, N, D), N = 1 (CLS) + num_patches

        # Some timm ViT variants prepend a CLS token; drop it to keep only patch tokens.
        if tokens.shape[1] > self._expected_num_patches(images):
            tokens = tokens[:, 1:, :]

        return tokens.cpu()

    def _expected_num_patches(self, images: torch.Tensor) -> int:
        h, w = images.shape[-2], images.shape[-1]
        return (h // self.patch_size) * (w // self.patch_size)

    def feature_map_size(self, image_size: int) -> int:
        """Number of patches per side, e.g. 224/14 = 16 -> a 16x16 grid."""
        return image_size // self.patch_size

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        """Convenience wrapper for extracting over a large stack of images in chunks."""
        all_feats = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i:i + batch_size]
            all_feats.append(self.extract(chunk))
        return torch.cat(all_feats, dim=0)