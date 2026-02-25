import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add experiments folder to path for imports
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, experiments_dir)

from Involuted_SIMAN_UNET import UNetInvSimAM

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ========================
# Dataset class
# ========================

class ISICSegmentationDataset:
    VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, base_dir, split="train", transform=None, seed=42):
        self.samples = []
        self.transform = transform

        all_samples = []
        for folder in ["train", "val"]:
            img_dir = os.path.join(base_dir, folder, "images")
            mask_dir = os.path.join(base_dir, folder, "masks")
            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                continue

            images = sorted([
                f for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in self.VALID_IMG_EXTENSIONS
                and "_superpixels" not in f
            ])
            masks = sorted([
                f for f in os.listdir(mask_dir)
                if os.path.splitext(f)[1].lower() in self.VALID_IMG_EXTENSIONS
                and "_superpixels" not in f
            ])

            for i, m in zip(images, masks):
                all_samples.append((os.path.join(img_dir, i), os.path.join(mask_dir, m)))

        import random
        random.Random(seed).shuffle(all_samples)

        split_idx = round(0.7 * len(all_samples))
        if split == "train":
            self.samples = all_samples[:split_idx]
        elif split == "test":
            self.samples = all_samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        img = np.array(img)
        mask = np.array(mask)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        return img, mask


# ========================
# Setup
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Test transform
transform_test = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Dataset paths - check common locations
possible_paths_2017 = [
    "/home/almaan/datasets/ISIC2017",
    "/home/aminu_yusuf/msgunet/datasets/ISIC2017",
    "/home/almaan/ISIC2017"
]

possible_paths_2018 = [
    "/home/almaan/datasets/ISIC2018",
    "/home/aminu_yusuf/msgunet/datasets/ISIC2018",
    "/home/almaan/ISIC2018"
]

base_dir_2017 = None
base_dir_2018 = None

for path in possible_paths_2017:
    if os.path.exists(path):
        base_dir_2017 = path
        break

for path in possible_paths_2018:
    if os.path.exists(path):
        base_dir_2018 = path
        break

if not base_dir_2017:
    print("❌ ISIC2017 dataset not found!")
    sys.exit(1)

if not base_dir_2018:
    print("❌ ISIC2018 dataset not found!")
    sys.exit(1)

print(f"✅ ISIC2017 dataset found at: {base_dir_2017}")
print(f"✅ ISIC2018 dataset found at: {base_dir_2018}")

# ========================
# Load datasets
# ========================

print("Loading test datasets...")
test_dataset_2017 = ISICSegmentationDataset(
    base_dir=base_dir_2017, split="test", transform=transform_test, seed=SEED
)
test_dataset_2018 = ISICSegmentationDataset(
    base_dir=base_dir_2018, split="test", transform=transform_test, seed=SEED
)

test_loader_2017 = DataLoader(test_dataset_2017, batch_size=8, shuffle=False)
test_loader_2018 = DataLoader(test_dataset_2018, batch_size=8, shuffle=False)

print(f"ISIC2017 test set size: {len(test_dataset_2017)}")
print(f"ISIC2018 test set size: {len(test_dataset_2018)}")

# ========================
# Load models
# ========================

print("\nLoading models...")
project_root = experiments_dir

# Involuted SIMAN UNet
print("Loading Involuted SIMAN UNet...")
repghost_model = UNetInvSimAM(in_channels=3, num_classes=1).to(device)
repghost_2017_path = os.path.join(project_root, "weights/best_involuted_siman_unet_isic2017.pth")

# Load weights if available
models_loaded = True
for model_name, model, path in [
    ("Involuted SIMAN UNet ISIC2017", repghost_model, repghost_2017_path),
]:
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"✅ Loaded {model_name}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            models_loaded = False
    else:
        print(f"⚠️  Weights not found for {model_name}: {path}")

if not models_loaded:
    print("\n⚠️  Warning: Some model weights were not found. Using pretrained models with random weights.")

# ========================
# Generate predictions and visualizations
# ========================

def visualize_predictions(model, loader, dataset_name, save_prefix):
    """Generate prediction visualizations with ground truth"""
    model.eval()
    
    imgs_batch, masks_batch = next(iter(loader))
    imgs_batch, masks_batch = imgs_batch.to(device), masks_batch.to(device)
    
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs_batch))
    
    n_samples = min(6, imgs_batch.shape[0])
    
    # Create figure with 4 columns: Input, GT Mask, Prediction, Overlay
    fig = plt.figure(figsize=(14, n_samples * 3))
    
    for idx in range(n_samples):
        img = imgs_batch[idx].cpu().numpy()
        mask_gt = masks_batch[idx, 0].cpu().numpy()
        pred = preds[idx, 0].cpu().numpy()
        
        # Denormalize image for display
        img_np = np.transpose(img, (1, 2, 0))
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # Binarize prediction
        pred_binary = pred > 0.5
        
        # 1. Input image
        plt.subplot(n_samples, 4, idx * 4 + 1)
        plt.imshow(img_np)
        plt.title(f"Input {idx+1}", fontsize=10, fontweight='bold')
        plt.axis('off')
        
        # 2. Ground truth mask
        plt.subplot(n_samples, 4, idx * 4 + 2)
        plt.imshow(mask_gt, cmap='gray')
        plt.title("Ground Truth", fontsize=10, fontweight='bold')
        plt.axis('off')
        
        # 3. Involuted SIMAN UNet prediction
        plt.subplot(n_samples, 4, idx * 4 + 3)
        plt.imshow(pred, cmap='viridis')
        plt.title("Prediction (Raw)", fontsize=10, fontweight='bold')
        plt.axis('off')
        
        # 4. Binary prediction with ground truth overlay
        plt.subplot(n_samples, 4, idx * 4 + 4)
        overlay = np.zeros((*pred_binary.shape, 3))
        # Red for prediction, Blue for ground truth, Magenta for both
        overlay[pred_binary, 0] = 1  # Red for prediction
        overlay[mask_gt > 0.5, 2] = 1  # Blue for GT
        overlay[(pred_binary) & (mask_gt > 0.5), 0] = 1  # Keep red when overlapping
        overlay[(pred_binary) & (mask_gt > 0.5), 2] = 1  # Add blue when overlapping = Magenta
        plt.imshow(overlay)
        plt.title("Overlay\n(Red=Pred, Blue=GT, Magenta=Both)", fontsize=9, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle(f'Involuted SIMAN UNet Predictions vs Ground Truth - {dataset_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = f"{save_prefix}_involuted_siman_unet_predictions_vs_gt.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.show()


# ========================
# Generate visualizations
# ========================

print("\n" + "="*70)
print("GENERATING SAMPLE PREDICTION VISUALIZATIONS")
print("="*70)

print("\n📊 ISIC2017 Predictions vs Ground Truth...")
try:
    visualize_predictions(repghost_model, test_loader_2017, 
                         "ISIC2017", "involuted_siman_unet_isic2017")
except Exception as e:
    print(f"Error generating ISIC2017 visualizations: {e}")
    import traceback
    traceback.print_exc()

print("\n📊 ISIC2018 Predictions vs Ground Truth...")
try:
    visualize_predictions(repghost_model, test_loader_2018, 
                         "ISIC2018", "involuted_siman_unet_isic2018")
except Exception as e:
    print(f"Error generating ISIC2018 visualizations: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ All prediction visualizations generated successfully!")
print("="*70)
