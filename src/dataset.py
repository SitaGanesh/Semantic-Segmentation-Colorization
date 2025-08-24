
# src/dataset.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import color

class ColorSegDataset(Dataset):
    """
    Dataset class for semantic colorization combining segmentation and colorization tasks.
    Expects RGB images and corresponding binary masks.
    """

    def __init__(self, img_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = target_size

        # Get all image files
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.img_files = []
        self.mask_files = []

        if os.path.exists(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if any(fname.lower().endswith(ext) for ext in img_extensions):
                    img_path = os.path.join(img_dir, fname)
                    # Try different mask naming conventions
                    mask_name = fname.rsplit('.', 1)[0] + '.png'  # Change extension to .png
                    mask_path = os.path.join(mask_dir, mask_name)

                    if os.path.exists(mask_path):
                        self.img_files.append(img_path)
                        self.mask_files.append(mask_path)
                    else:
                        print(f"Warning: Mask not found for {fname}")
        else:
            print(f"Warning: Image directory {img_dir} does not exist")

        if len(self.img_files) == 0:
            print("Warning: No valid image-mask pairs found. Creating dummy data for testing.")
            self._create_dummy_data()

        # Default transform
        self.transform = transform or T.Compose([
            T.ToTensor(),
        ])

        print(f"Dataset initialized with {len(self.img_files)} image-mask pairs")

    def _create_dummy_data(self):
        """Create dummy data for testing purposes"""
        # Create dummy directories if they don't exist
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

        # Create a few dummy RGB images and corresponding masks
        for i in range(5):
            # Create dummy RGB image (colorful)
            img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

            # Add some structure to the image
            cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Red rectangle
            cv2.circle(img, (200, 200), 30, (0, 255, 0), -1)  # Green circle

            # Create corresponding binary mask (foreground/background)
            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)  # Rectangle as foreground
            cv2.circle(mask, (200, 200), 30, 255, -1)  # Circle as foreground

            # Save dummy data
            img_path = os.path.join(self.img_dir, f"dummy_{i}.png")
            mask_path = os.path.join(self.mask_dir, f"dummy_{i}.png")

            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)

            self.img_files.append(img_path)
            self.mask_files.append(mask_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            # Load RGB image
            img_path = self.img_files[idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)

            # Load mask
            mask_path = self.mask_files[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size)

            # Normalize mask to 0 and 1
            mask = (mask > 128).astype(np.uint8)

            # Convert RGB to LAB color space
            img_lab = color.rgb2lab(img / 255.0)  # rgb2lab expects [0,1] range

            # Extract L, a, b channels
            L = img_lab[:, :, 0:1]  # L channel [0, 100]
            ab = img_lab[:, :, 1:3]  # ab channels [-128, 127]

            # Normalize for neural network
            L = L / 100.0  # Normalize L to [0, 1]
            ab = (ab + 128) / 255.0  # Normalize ab to [0, 1]

            # Convert to tensors
            L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).float()  # (1, H, W)
            ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).float()  # (2, H, W)
            mask_tensor = torch.from_numpy(mask).long()  # (H, W)

            # Apply transform to RGB image for visualization
            img_rgb_tensor = self.transform(img / 255.0)  # (3, H, W)

            return {
                "image": img_rgb_tensor,
                "L_channel": L_tensor,
                "ab_channels": ab_tensor,
                "mask": mask_tensor,
                "img_path": img_path
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample in case of error
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Return a dummy sample in case of loading error"""
        H, W = self.target_size
        return {
            "image": torch.randn(3, H, W),
            "L_channel": torch.randn(1, H, W),
            "ab_channels": torch.randn(2, H, W),
            "mask": torch.zeros(H, W, dtype=torch.long),
            "img_path": "dummy"
        }

def get_dataloader(img_dir, mask_dir, batch_size=8, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the ColorSegDataset
    """
    dataset = ColorSegDataset(img_dir, mask_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

# Test the dataset
if __name__ == "__main__":
    # Test dataset with dummy data
    print("Testing dataset...")

    # Create test directories
    test_img_dir = "data/processed/images"
    test_mask_dir = "data/processed/masks"

    # Create dataset
    dataset = ColorSegDataset(test_img_dir, test_mask_dir)
    print(f"Dataset length: {len(dataset)}")

    # Test a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"L channel shape: {sample['L_channel'].shape}")
        print(f"AB channels shape: {sample['ab_channels'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")

        # Test dataloader
        dataloader = get_dataloader(test_img_dir, test_mask_dir, batch_size=2)
        for batch in dataloader:
            print(f"Batch image shape: {batch['image'].shape}")
            print(f"Batch L channel shape: {batch['L_channel'].shape}")
            print(f"Batch AB channels shape: {batch['ab_channels'].shape}")
            print(f"Batch mask shape: {batch['mask'].shape}")
            break

        print("Dataset test completed successfully!")
    else:
        print("No samples found in dataset")
