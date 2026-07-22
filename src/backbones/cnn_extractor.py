"""
CNN (ResNet) feature extractor for PaDiM.

PaDiM fits a per-location Gaussian using features pooled from multiple
CNN layers (early + mid layers capture texture; later layers capture
more semantic structure). We hook into a pretrained, frozen ResNet
and grab intermediate layer activations directly.
"""

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

DEFAULT_MODEL_NAME = "resnet18"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_cnn_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CNNPatchExtractor:
    """
    Wraps a frozen pretrained ResNet and exposes pooled multi-layer
    patch-grid features suitable for PaDiM's per-location Gaussian fitting.

    Output shape per image: (grid_size * grid_size, D)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME,
                 layers: tuple = ("layer1", "layer2", "layer3"),
                 device: torch.device | None = None):
        self.device = device or get_device()
        self.layers = layers

        # features_only=True + out_indices lets timm return intermediate
        # feature maps directly, instead of the final classification output.
        self.model = timm.create_model(
            model_name, pretrained=True, features_only=True,
            out_indices=self._layer_indices(model_name, layers),
        )
        self.model.eval()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

    def _layer_indices(self, model_name: str, layers: tuple) -> tuple:
        # ResNet's timm feature stages are indexed 0-4 (stem, layer1..layer4).
        name_to_index = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
        return tuple(name_to_index[l] for l in layers)

    @torch.no_grad()
    def extract(self, images: torch.Tensor, grid_size: int = 28) -> torch.Tensor:
        """
        images: (B, 3, H, W)
        Returns: (B, grid_size*grid_size, D) - concatenated multi-layer
                 features, resized to a common grid_size x grid_size map.
        """
        images = images.to(self.device)
        feature_maps = self.model(images)  # list of (B, C_i, H_i, W_i), one per layer

        resized = []
        for fmap in feature_maps:
            fmap = F.interpolate(fmap, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
            resized.append(fmap)

        # Concatenate along the channel dimension: (B, C1+C2+C3, grid, grid)
        combined = torch.cat(resized, dim=1)

        b, d, h, w = combined.shape
        # Flatten spatial grid into a patch sequence, matching ViT's (B, P, D) shape.
        combined = combined.permute(0, 2, 3, 1).reshape(b, h * w, d)

        return combined.cpu()

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor, grid_size: int = 28, batch_size: int = 16) -> torch.Tensor:
        all_feats = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i:i + batch_size]
            all_feats.append(self.extract(chunk, grid_size=grid_size))
        return torch.cat(all_feats, dim=0)