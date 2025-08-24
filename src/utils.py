# src/utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import cv2
import os
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint with all necessary information

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss,
        'model_class': model.__class__.__name__
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path, device='cpu'):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (can be None)
        path: Path to checkpoint file
        device: Device to load the model on

    Returns:
        epoch: Loaded epoch number
        loss: Loaded loss value
    """
    if not os.path.exists(path):
        print(f"Checkpoint file not found: {path}")
        return 0, float('inf')

    try:
        checkpoint = torch.load(path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        print(f"Checkpoint loaded from {path}, epoch {epoch}, loss {loss:.4f}")
        return epoch, loss

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, float('inf')

def lab_to_rgb(lab_images):
    """
    Convert LAB images to RGB using scikit-image

    Args:
        lab_images: numpy array of shape (H, W, 3) or (B, H, W, 3) in LAB color space
                   L: [0, 100], a,b: [-127, 127]

    Returns:
        rgb_images: numpy array of RGB images [0, 1]
    """
    if len(lab_images.shape) == 3:
        # Single image (H, W, 3)
        lab_normalized = lab_images.copy()
        lab_normalized[:, :, 0] = lab_normalized[:, :, 0] / 100.0  # L: [0,100] -> [0,1]
        lab_normalized[:, :, 1:] = (lab_normalized[:, :, 1:] + 127) / 254.0  # ab: [-127,127] -> [0,1]

        rgb = color.lab2rgb(lab_normalized)
        return np.clip(rgb, 0, 1)

    elif len(lab_images.shape) == 4:
        # Batch of images (B, H, W, 3)
        rgb_batch = []
        for i in range(lab_images.shape[0]):
            lab_single = lab_images[i]
            lab_normalized = lab_single.copy()
            lab_normalized[:, :, 0] = lab_normalized[:, :, 0] / 100.0
            lab_normalized[:, :, 1:] = (lab_normalized[:, :, 1:] + 127) / 254.0

            rgb = color.lab2rgb(lab_normalized)
            rgb_batch.append(np.clip(rgb, 0, 1))

        return np.stack(rgb_batch, axis=0)
    else:
        raise ValueError("lab_images should have shape (H, W, 3) or (B, H, W, 3)")

def rgb_to_lab(rgb_images):
    """
    Convert RGB images to LAB using scikit-image

    Args:
        rgb_images: numpy array of shape (H, W, 3) or (B, H, W, 3) in RGB color space [0, 1]

    Returns:
        lab_images: numpy array of LAB images, L: [0, 100], a,b: [-127, 127]
    """
    if len(rgb_images.shape) == 3:
        # Single image
        lab = color.rgb2lab(rgb_images)
        # Ensure proper range
        lab[:, :, 1:] = np.clip(lab[:, :, 1:], -127, 127)
        return lab

    elif len(rgb_images.shape) == 4:
        # Batch of images
        lab_batch = []
        for i in range(rgb_images.shape[0]):
            lab = color.rgb2lab(rgb_images[i])
            lab[:, :, 1:] = np.clip(lab[:, :, 1:], -127, 127)
            lab_batch.append(lab)

        return np.stack(lab_batch, axis=0)
    else:
        raise ValueError("rgb_images should have shape (H, W, 3) or (B, H, W, 3)")

def tensor_to_numpy_image(tensor):
    """
    Convert torch tensor to numpy image

    Args:
        tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)

    Returns:
        numpy array of shape (H, W, C) or (B, H, W, C)
    """
    if tensor.dim() == 4:
        # Batch of images (B, C, H, W) -> (B, H, W, C)
        return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    elif tensor.dim() == 3:
        # Single image (C, H, W) -> (H, W, C)
        return tensor.detach().cpu().permute(1, 2, 0).numpy()
    else:
        raise ValueError("Tensor should have 3 or 4 dimensions")

def numpy_to_tensor_image(array):
    """
    Convert numpy image to torch tensor

    Args:
        array: numpy array of shape (H, W, C) or (B, H, W, C)

    Returns:
        torch.Tensor of shape (C, H, W) or (B, C, H, W)
    """
    if len(array.shape) == 4:
        # Batch of images (B, H, W, C) -> (B, C, H, W)
        return torch.from_numpy(array).permute(0, 3, 1, 2).float()
    elif len(array.shape) == 3:
        # Single image (H, W, C) -> (C, H, W)
        return torch.from_numpy(array).permute(2, 0, 1).float()
    else:
        raise ValueError("Array should have 3 or 4 dimensions")

def show_results(images, seg_logits, color_ab, masks, save_path=None, max_images=4):
    """
    Visualize segmentation and colorization results

    Args:
        images: Input RGB images tensor (B, 3, H, W)
        seg_logits: Segmentation logits (B, n_classes, H, W)
        color_ab: Predicted ab channels (B, 2, H, W)
        masks: Ground truth masks (B, H, W)
        save_path: Path to save the visualization
        max_images: Maximum number of images to show
    """
    batch_size = min(images.size(0), max_images)

    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        # Convert to numpy
        img_np = tensor_to_numpy_image(images[i])
        mask_gt = masks[i].cpu().numpy()

        # Get predicted segmentation
        seg_pred = torch.argmax(seg_logits[i], dim=0).cpu().numpy()

        # Convert to LAB and create colorized image
        try:
            # Convert RGB to LAB
            img_lab = rgb_to_lab(img_np)
            L_channel = img_lab[:, :, 0:1]

            # Use predicted ab channels
            ab_pred = tensor_to_numpy_image(color_ab[i])  # (H, W, 2)

            # Denormalize ab channels (assuming they were normalized to [-1, 1])
            ab_pred = ab_pred * 127  # Scale back to [-127, 127]

            # Combine L and predicted ab
            lab_colorized = np.concatenate([L_channel, ab_pred], axis=2)

            # Convert back to RGB
            rgb_colorized = lab_to_rgb(lab_colorized)

        except Exception as e:
            print(f"Error in colorization conversion for image {i}: {e}")
            rgb_colorized = img_np  # Fallback to original image

        # Plot results
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_gt, cmap='gray')
        axes[i, 1].set_title('GT Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(seg_pred, cmap='viridis')
        axes[i, 2].set_title('Pred. Seg')
        axes[i, 2].axis('off')

        # Show ab channels
        ab_vis = (ab_pred - ab_pred.min()) / (ab_pred.max() - ab_pred.min() + 1e-8)
        axes[i, 3].imshow(ab_vis)
        axes[i, 3].set_title('Pred. AB')
        axes[i, 3].axis('off')

        axes[i, 4].imshow(rgb_colorized)
        axes[i, 4].set_title('Colorized')
        axes[i, 4].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to {save_path}")

    plt.show()

def create_project_structure():
    """
    Create the proper directory structure for the project
    """
    directories = [
        "data/raw",
        "data/processed/images",
        "data/processed/masks",
        "data/external",
        "outputs/checkpoints",
        "outputs/results",
        "outputs/logs",
        "src",
        "tests",
        "notebooks"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create .gitkeep files to ensure directories are tracked
    for directory in directories:
        gitkeep_path = Path(directory) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()

def calculate_metrics(seg_pred, seg_gt, num_classes=2):
    """
    Calculate segmentation metrics

    Args:
        seg_pred: Predicted segmentation (numpy array)
        seg_gt: Ground truth segmentation (numpy array)
        num_classes: Number of classes

    Returns:
        dict: Dictionary containing various metrics
    """
    # Flatten arrays
    pred_flat = seg_pred.flatten()
    gt_flat = seg_gt.flatten()

    # Calculate accuracy
    accuracy = np.mean(pred_flat == gt_flat)

    # Calculate IoU for each class
    ious = []
    for class_id in range(num_classes):
        pred_class = (pred_flat == class_id)
        gt_class = (gt_flat == class_id)

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        if union == 0:
            iou = 0.0  # Handle case where class is not present
        else:
            iou = intersection / union

        ious.append(iou)

    mean_iou = np.mean(ious)

    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'class_ious': ious
    }

def colorization_metrics(rgb_pred, rgb_gt):
    """
    Calculate colorization metrics

    Args:
        rgb_pred: Predicted RGB images (numpy array)
        rgb_gt: Ground truth RGB images (numpy array)

    Returns:
        dict: Dictionary containing colorization metrics
    """
    # Mean Squared Error
    mse = np.mean((rgb_pred - rgb_gt) ** 2)

    # Peak Signal-to-Noise Ratio
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))

    # Structural Similarity (simplified version)
    # This is a simplified SSIM calculation
    mu_pred = np.mean(rgb_pred)
    mu_gt = np.mean(rgb_gt)
    sigma_pred = np.var(rgb_pred)
    sigma_gt = np.var(rgb_gt)
    sigma_pred_gt = np.mean((rgb_pred - mu_pred) * (rgb_gt - mu_gt))

    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_pred * mu_gt + c1) * (2 * sigma_pred_gt + c2)) / \
           ((mu_pred**2 + mu_gt**2 + c1) * (sigma_pred + sigma_gt + c2))

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }

# Test the utilities
if __name__ == "__main__":
    print("Testing utilities...")

    # Test directory creation
    create_project_structure()

    # Test image conversion functions
    # Create a dummy RGB image
    rgb_img = np.random.rand(64, 64, 3)

    # Test RGB to LAB conversion
    lab_img = rgb_to_lab(rgb_img)
    print(f"RGB shape: {rgb_img.shape}, LAB shape: {lab_img.shape}")
    print(f"LAB ranges - L: [{lab_img[:,:,0].min():.2f}, {lab_img[:,:,0].max():.2f}], "
          f"a: [{lab_img[:,:,1].min():.2f}, {lab_img[:,:,1].max():.2f}], "
          f"b: [{lab_img[:,:,2].min():.2f}, {lab_img[:,:,2].max():.2f}]")

    # Test LAB to RGB conversion
    rgb_recovered = lab_to_rgb(lab_img)
    print(f"RGB recovered shape: {rgb_recovered.shape}")

    # Test tensor conversions
    tensor_img = numpy_to_tensor_image(rgb_img)
    numpy_recovered = tensor_to_numpy_image(tensor_img)
    print(f"Tensor shape: {tensor_img.shape}, Numpy recovered shape: {numpy_recovered.shape}")

    # Test metrics
    seg_pred = np.random.randint(0, 2, (64, 64))
    seg_gt = np.random.randint(0, 2, (64, 64))
    metrics = calculate_metrics(seg_pred, seg_gt)
    print(f"Segmentation metrics: {metrics}")

    colorization_metrics_result = colorization_metrics(rgb_img, rgb_recovered)
    print(f"Colorization metrics: {colorization_metrics_result}")

    print("Utilities test completed successfully!")