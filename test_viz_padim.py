import torch
from PIL import Image
from src.datasets.mvtec_dataset import MVTecDataset
from src.backbones.cnn_extractor import CNNPatchExtractor, get_cnn_transform
from src.methods.padim import PaDiM
from src.utils.visualization import save_comparison_figure

transform = get_cnn_transform()
extractor = CNNPatchExtractor()
grid_size = 28

train_ds = MVTecDataset('data/mvtec_ad', category='bottle', split='train', transform=transform)
train_imgs = torch.stack([train_ds[i]['image'] for i in range(209)])
train_feats = extractor.extract_batch(train_imgs, grid_size=grid_size, batch_size=16)

model = PaDiM(n_components=100)
model.fit(train_feats)

test_ds = MVTecDataset('data/mvtec_ad', category='bottle', split='test', transform=None)
defective_idx = next(i for i in range(len(test_ds)) if test_ds[i]['label'] == 1 and test_ds[i]['mask'].sum() > 0)
raw_sample = test_ds[defective_idx]

test_ds_transformed = MVTecDataset('data/mvtec_ad', category='bottle', split='test', transform=transform)
transformed_sample = test_ds_transformed[defective_idx]
image_tensor = transformed_sample['image'].unsqueeze(0)

test_feats = extractor.extract(image_tensor, grid_size=grid_size)
result = model.score(test_feats, grid_size=grid_size)

title = "PaDiM | bottle | " + raw_sample["defect_type"] + " | score=" + str(round(result["image_scores"][0], 1))

save_comparison_figure(
    original_image=raw_sample['image'],
    mask=raw_sample['mask'],
    
    heatmap=result['heatmaps'][0],
    save_path='results/qualitative_examples/bottle_padim_example.png',
    title=title,
)
print('saved!')