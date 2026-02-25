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

# Add parent directory to path to import ESEUNet from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add scripts directory to path for loss_functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models and loss functions
from Involuted_SIMAN_UNET import UNetInvSimAM
from loss_functions import CombinedDeepSupervisionLoss

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
            transform: Albumentations transform for both images and masks.
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


base_dir_2018 = "/home/almaan/datasets/ISIC2018"

# Training transforms with augmentation (matching MSGU-Net)
transform_train = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5), 
    A.Rotate(limit=15, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Test transforms (no augmentation)
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

# Initialize UNetInvSimAM (Involuted SIMAN UNet)
model = UNetInvSimAM(
    in_channels=3,
    num_classes=1,
    base_c=64,
    bilinear=True
).to(device)

# Use same loss as MSGU-Net
criterion = nn.BCEWithLogitsLoss()   # binary segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5
)


def train_epoch(loader, model, criterion, optimizer, device, epoch=None, n_epochs=None):
    model.train()
    running_loss = 0.0
    
    # Wrap loader with tqdm
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
    
    for imgs, masks in progress_bar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  # live update of batch loss
    
    return running_loss / len(loader)


def eval_epoch(loader, model, criterion, device, epoch=None, n_epochs=None):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False)
        
        for imgs, masks in progress_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    
    return running_loss / len(loader)


if __name__ == "__main__":
    # ISIC2018 Training + Testing

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    n_epochs = 100
    train_losses = []
    test_losses = []

    best_loss = float("inf")
    best_model_path = os.path.join(weights_dir, "best_involuted_siman_unet_isic2018.pth")

    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader_2018, model, criterion, optimizer, device, epoch, n_epochs)
        train_losses.append(train_loss)
        
        test_loss = eval_epoch(test_loader_2018, model, criterion, device, epoch, n_epochs)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model at epoch {epoch+1} with Test Loss: {test_loss:.4f}")

    # Optionally save final model too
    final_model_path = os.path.join(weights_dir, "involuted_siman_unet_isic2018.pth")
    torch.save(model.state_dict(), final_model_path)
    print("💾 Training complete, final model saved.")

    model.eval()
    imgs, masks = next(iter(test_loader_2018))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(imgs)  # Returns single output
        preds = torch.sigmoid(outputs)

    n_samples = min(6, imgs.shape[0])
    plt.figure(figsize=(12, n_samples * 3))
    for idx in range(n_samples):
        # Base image
        plt.subplot(n_samples, 3, idx * 3 + 1)
        plt.title(f"Image {idx+1}")
        plt.imshow(np.transpose(imgs[idx].cpu().numpy(), (1,2,0)))
        plt.axis('off')
        # Ground truth mask
        plt.subplot(n_samples, 3, idx * 3 + 2)
        plt.title("Mask")
        plt.imshow(masks[idx,0].cpu().numpy(), cmap="gray")
        plt.axis('off')
        # Prediction
        plt.subplot(n_samples, 3, idx * 3 + 3)
        plt.title("Involuted SIMAN UNet Prediction")
        plt.imshow(preds[idx,0].cpu().numpy() > 0.5, cmap="gray")
        plt.axis('off')

    plt.tight_layout()
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'involuted_siman_unet_predictions_grid_isic2018.png'))
    plt.show()

    # After training cell (after training loop and model saving)
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Involuted SIMAN UNet - Loss Curve (ISIC2018)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'involuted_siman_unet_loss_curve_isic2018.png'))
    plt.show()




