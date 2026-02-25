import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path to import UNetInvSimAM from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UNetInvSimAM
from Involuted_SIMAN_UNET import UNetInvSimAM

from PIL import Image
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ISICSegmentationDataset(Dataset):
    VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # normal images/masks

    def __init__(self, base_dir, split="train", transform=None, seed=42):
        """
        Args:
            base_dir (str): Path to the ISIC dataset (ISIC2017 or ISIC2018).
            split (str): One of ["train", "test"] for the final 70:30 split.
            transform: Albumentations transforms (handles both image and mask).
            seed (int): Random seed for shuffling.
        """
        self.samples = []
        self.transform = transform

        # Gather images/masks from both train and val folders
        all_samples = []
        for folder in ["train", "val"]:
            img_dir = os.path.join(base_dir, folder, "images")
            mask_dir = os.path.join(base_dir, folder, "masks")
            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                continue

            # Keep only normal images (exclude *_superpixels.png)
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

        # Shuffle once to avoid bias before splitting
        random.Random(seed).shuffle(all_samples)

        # 70:30 split
        split_idx = round(0.7 * len(all_samples))
        if split == "train":
            self.samples = all_samples[:split_idx]
        elif split == "test":
            self.samples = all_samples[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Convert PIL to numpy arrays for albumentations
        img = np.array(img)
        mask = np.array(mask)

        # Apply albumentations transforms (handles image-mask sync automatically)
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # Add channel dimension to mask for BCEWithLogitsLoss
        # Albumentations ToTensorV2 outputs mask as [H, W], but we need [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        # Normalize mask to [0, 1] range if needed
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        return img, mask


def calculate_metrics(predictions, ground_truth, threshold=0.5):
    """
    Calculate segmentation metrics: mIoU, DSC, Specificity, and Sensitivity
    
    Args:
        predictions: torch.Tensor, predicted masks (after sigmoid)
        ground_truth: torch.Tensor, ground truth masks
        threshold: float, threshold for binarizing predictions
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Binarize predictions
    pred_binary = (predictions > threshold).float()
    gt_binary = ground_truth.float()
    
    # Flatten tensors for easier calculation
    pred_flat = pred_binary.view(-1)
    gt_flat = gt_binary.view(-1)
    
    # Calculate confusion matrix components
    tp = torch.sum(pred_flat * gt_flat).item()  # True Positives
    fp = torch.sum(pred_flat * (1 - gt_flat)).item()  # False Positives
    fn = torch.sum((1 - pred_flat) * gt_flat).item()  # False Negatives
    tn = torch.sum((1 - pred_flat) * (1 - gt_flat)).item()  # True Negatives
    
    # Calculate metrics
    # Intersection over Union (IoU)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Dice Similarity Coefficient (DSC)
    dsc = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # Sensitivity (Recall/True Positive Rate)
    sensitivity = tp / (tp + fn + 1e-8)
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'mIoU': iou * 100,  # Convert to percentage
        'DSC': dsc * 100,   # Convert to percentage
        'Sensitivity': sensitivity * 100,  # Convert to percentage
        'Specificity': specificity * 100   # Convert to percentage
    }


def evaluate_model_metrics(model, test_loader, device):
    """
    Evaluate model on test dataset and calculate average metrics
    
    Args:
        model: trained model
        test_loader: DataLoader for test dataset
        device: torch device
    
    Returns:
        dict: Average metrics across all test samples
    """
    model.eval()
    
    all_ious = []
    all_dscs = []
    all_sensitivities = []
    all_specificities = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating on test set")
        
        for imgs, masks in progress_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Get predictions
            outputs = model(imgs)
            predictions = torch.sigmoid(outputs)
            
            # Calculate metrics for each sample in the batch
            for i in range(imgs.shape[0]):
                metrics = calculate_metrics(predictions[i:i+1], masks[i:i+1])
                all_ious.append(metrics['mIoU'])
                all_dscs.append(metrics['DSC'])
                all_sensitivities.append(metrics['Sensitivity'])
                all_specificities.append(metrics['Specificity'])
    
    # Calculate average metrics
    avg_metrics = {
        'mIoU': np.mean(all_ious),
        'DSC': np.mean(all_dscs),
        'Sensitivity': np.mean(all_sensitivities),
        'Specificity': np.mean(all_specificities)
    }
    
    return avg_metrics


base_dir_2018 = "/home/almaan/datasets/ISIC2018"

# Training transforms with augmentation (using albumentations)
transform_train = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5), 
    A.Rotate(limit=15, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Test transforms (no augmentation - deterministic only)
transform_test = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

train_dataset_2018 = ISICSegmentationDataset(
    base_dir=base_dir_2018, split="train", transform=transform_train, seed=SEED
)
test_dataset_2018 = ISICSegmentationDataset(
    base_dir=base_dir_2018, split="test", transform=transform_test, seed=SEED
)

train_loader_2018 = DataLoader(train_dataset_2018, batch_size=8, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
test_loader_2018 = DataLoader(test_dataset_2018, batch_size=8, shuffle=False, worker_init_fn=seed_worker)

print("Train size:", len(train_dataset_2018))
print("Test size:", len(test_dataset_2018))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize model
model = UNetInvSimAM(in_channels=3, num_classes=1, base_c=64, bilinear=True).to(device)

# Load pretrained weights (path relative to project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pretrained_weights_path = os.path.join(project_root, "weights/best_involuted_siman_unet_isic2018.pth")
if os.path.exists(pretrained_weights_path):
    print(f"Loading pretrained weights from: {pretrained_weights_path}")
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    print("✅ Pretrained weights loaded successfully!")
else:
    print(f"❌ Warning: Pretrained weights not found at {pretrained_weights_path}")
    print("Please ensure the weights file exists or update the path.")

# Set model to evaluation mode for inference
model.eval()
print("Model set to evaluation mode.")


# Run inference on test dataset and calculate metrics
print("Running inference on test dataset...")
print("=" * 50)

# Calculate metrics on test dataset
test_metrics = evaluate_model_metrics(model, test_loader_2018, device)

# Display results
print("ISIC2018 Test Dataset Results:")
print("=" * 50)
print(f"mIoU (Mean Intersection over Union): {test_metrics['mIoU']:.2f}%")
print(f"DSC (Dice Similarity Coefficient): {test_metrics['DSC']:.2f}%")
print(f"Sensitivity (Recall): {test_metrics['Sensitivity']:.2f}%")
print(f"Specificity: {test_metrics['Specificity']:.2f}%")
print("=" * 50)

# ISIC2018 Inference and Metrics Evaluation
# 
# This script loads a pretrained RepGhost U-Net model and evaluates it on the ISIC2018 test dataset to calculate:
# - **mIoU (Mean Intersection over Union)**: Measures overlap between predicted and ground truth masks
# - **DSC (Dice Similarity Coefficient)**: Measures similarity between predicted and ground truth masks  
# - **Sensitivity (Recall)**: True positive rate - ability to correctly identify positive pixels
# - **Specificity**: True negative rate - ability to correctly identify negative pixels
# 
# All metrics are reported as percentages with two decimal places.


# Visualize sample predictions with individual metrics
model.eval()
imgs, masks = next(iter(test_loader_2018))
imgs, masks = imgs.to(device), masks.to(device)

with torch.no_grad():
    outputs = model(imgs)
    preds = torch.sigmoid(outputs)

n_samples = min(6, imgs.shape[0])
plt.figure(figsize=(15, n_samples * 3))

for idx in range(n_samples):
    # Calculate metrics for this specific sample
    sample_metrics = calculate_metrics(preds[idx:idx+1], masks[idx:idx+1])
    
    # Original image
    plt.subplot(n_samples, 3, idx * 3 + 1)
    plt.title(f"Image {idx+1}")
    # Denormalize image for display
    img_denorm = imgs[idx].cpu()
    img_denorm = img_denorm * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img_denorm = torch.clamp(img_denorm, 0, 1)
    plt.imshow(np.transpose(img_denorm.numpy(), (1,2,0)))
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(n_samples, 3, idx * 3 + 2)
    plt.title("Ground Truth")
    plt.imshow(masks[idx,0].cpu().numpy(), cmap="gray")
    plt.axis('off')
    
    # Prediction with metrics
    plt.subplot(n_samples, 3, idx * 3 + 3)
    plt.title(f"Prediction\nIoU: {sample_metrics['mIoU']:.1f}% | DSC: {sample_metrics['DSC']:.1f}%")
    plt.imshow(preds[idx,0].cpu().numpy() > 0.5, cmap="gray")
    plt.axis('off')

plt.tight_layout()
# Save plots to project root plots directory
plots_dir = os.path.join(project_root, "plots")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'involuted_siman_unet_predictions_with_metrics_isic2018.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"mIoU (Mean Intersection over Union): {test_metrics['mIoU']:.2f}%")
print(f"DSC (Dice Similarity Coefficient): {test_metrics['DSC']:.2f}%")
print(f"Sensitivity (Recall): {test_metrics['Sensitivity']:.2f}%")
print(f"Specificity: {test_metrics['Specificity']:.2f}%")
print("="*60)


