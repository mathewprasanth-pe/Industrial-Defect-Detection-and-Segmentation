import torch
from PIL import Image
from src.datasets.mvtec_dataset import MVTecDataset
from src.backbones.vit_extractor import ViTPatchExtractor, get_vit_transform
from src.methods.patchcore import PatchCore
from src.utils.visualization import save_comparison_figure

transform = get_vit_transform()
extractor = ViTPatchExtractor()
grid_size = extractor.feature_map_size(224)

train_ds = MVTecDataset('data/mvtec_ad', category='bottle', split='train', transform=transform)
train_imgs = torch.stack([train_ds[i]['image'] for i in range(209)])
train_feats = extractor.extract_batch(train_imgs, batch_size=16)

model = PatchCore(coreset_ratio=0.1)
model.fit(train_feats)

test_ds = MVTecDataset('data/mvtec_ad', category='bottle', split='test', transform=None)
defective_idx = next(i for i in range(len(test_ds)) if test_ds[i]['label'] == 1 and test_ds[i]['mask'].sum() > 0)
raw_sample = test_ds[defective_idx]

test_ds_transformed = MVTecDataset('data/mvtec_ad', category='bottle', split='test', transform=transform)
transformed_sample = test_ds_transformed[defective_idx]
image_tensor = transformed_sample['image'].unsqueeze(0)

test_feats = extractor.extract(image_tensor)
result = model.score(test_feats, grid_size=grid_size)

title = "PatchCore | bottle | " + raw_sample["defect_type"] + " | score=" + str(round(result["image_scores"][0], 1))

save_comparison_figure(
    original_image=raw_sample['image'],
    mask=raw_sample['mask'],
    heatmap=result['heatmaps'][0],
    save_path='results/qualitative_examples/bottle_patchcore_example.png',
    title=title,
)
print('saved!')