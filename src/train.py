# src/train.py
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

# Import our modules
from src_dataset import get_dataloader
from src_model import SegColorNet
from src_utils import save_checkpoint, load_checkpoint, calculate_metrics, colorization_metrics, create_project_structure

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationColorationLoss(torch.nn.Module):
    """
    Combined loss for segmentation and colorization tasks
    """

    def __init__(self, seg_weight=1.0, color_weight=1.0, ignore_index=-1):
        super().__init__()
        self.seg_weight = seg_weight
        self.color_weight = color_weight
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.color_criterion = torch.nn.MSELoss()

    def forward(self, seg_logits, color_pred, seg_target, color_target, mask=None):
        """
        Calculate combined loss

        Args:
            seg_logits: Predicted segmentation logits (B, n_classes, H, W)
            color_pred: Predicted ab channels (B, 2, H, W) in range [-1, 1]
            seg_target: Ground truth segmentation (B, H, W)
            color_target: Ground truth ab channels (B, 2, H, W) in range [0, 1]
            mask: Optional mask to focus colorization loss (B, H, W)
        """
        # Segmentation loss
        seg_loss = self.seg_criterion(seg_logits, seg_target)

        # Colorization loss
        # Convert color_target from [0, 1] to [-1, 1] to match prediction range
        color_target_scaled = color_target * 2.0 - 1.0

        if mask is not None:
            # Apply mask to focus on foreground pixels
            mask_expanded = mask.unsqueeze(1).float()  # (B, 1, H, W)
            color_loss = self.color_criterion(
                color_pred * mask_expanded, 
                color_target_scaled * mask_expanded
            )
        else:
            color_loss = self.color_criterion(color_pred, color_target_scaled)

        total_loss = self.seg_weight * seg_loss + self.color_weight * color_loss

        return total_loss, seg_loss, color_loss

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch
    """
    model.train()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_color_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        try:
            # Move data to device
            images = batch['image'].to(device)  # RGB images for display
            L_channel = batch['L_channel'].to(device)  # L channel input
            ab_target = batch['ab_channels'].to(device)  # ab channel target
            seg_target = batch['mask'].to(device)  # segmentation target

            # Forward pass - use L channel as input
            optimizer.zero_grad()
            seg_logits, color_pred = model(L_channel)

            # Calculate loss
            total_loss_batch, seg_loss_batch, color_loss_batch = criterion(
                seg_logits, color_pred, seg_target, ab_target, 
                mask=seg_target.float()  # Use segmentation mask to focus colorization
            )

            # Backward pass
            total_loss_batch.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update running averages
            total_loss += total_loss_batch.item()
            total_seg_loss += seg_loss_batch.item()
            total_color_loss += color_loss_batch.item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Seg': f'{seg_loss_batch.item():.4f}',
                'Color': f'{color_loss_batch.item():.4f}'
            })

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / num_batches
    avg_seg_loss = total_seg_loss / num_batches
    avg_color_loss = total_color_loss / num_batches

    return avg_loss, avg_seg_loss, avg_color_loss

def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_color_loss = 0.0
    num_batches = len(dataloader)

    all_seg_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation')):
            try:
                # Move data to device
                images = batch['image'].to(device)
                L_channel = batch['L_channel'].to(device)
                ab_target = batch['ab_channels'].to(device)
                seg_target = batch['mask'].to(device)

                # Forward pass
                seg_logits, color_pred = model(L_channel)

                # Calculate loss
                total_loss_batch, seg_loss_batch, color_loss_batch = criterion(
                    seg_logits, color_pred, seg_target, ab_target,
                    mask=seg_target.float()
                )

                total_loss += total_loss_batch.item()
                total_seg_loss += seg_loss_batch.item()
                total_color_loss += color_loss_batch.item()

                # Calculate metrics
                seg_pred = torch.argmax(seg_logits, dim=1).cpu().numpy()
                seg_gt = seg_target.cpu().numpy()

                for i in range(seg_pred.shape[0]):
                    metrics = calculate_metrics(seg_pred[i], seg_gt[i])
                    all_seg_metrics.append(metrics)

            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / num_batches
    avg_seg_loss = total_seg_loss / num_batches
    avg_color_loss = total_color_loss / num_batches

    # Calculate average metrics
    if all_seg_metrics:
        avg_accuracy = np.mean([m['accuracy'] for m in all_seg_metrics])
        avg_miou = np.mean([m['mean_iou'] for m in all_seg_metrics])
    else:
        avg_accuracy = 0.0
        avg_miou = 0.0

    return avg_loss, avg_seg_loss, avg_color_loss, avg_accuracy, avg_miou

def main():
    parser = argparse.ArgumentParser(description='Train Semantic Colorization Model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')

    args = parser.parse_args()

    # Create project structure
    create_project_structure()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f'Using device: {device}')

    # Setup data loaders
    train_img_dir = os.path.join(args.data_dir, 'images')
    train_mask_dir = os.path.join(args.data_dir, 'masks')

    try:
        train_loader = get_dataloader(
            train_img_dir, train_mask_dir, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers
        )

        # Use same data for validation (in practice, you'd have separate val data)
        val_loader = get_dataloader(
            train_img_dir, train_mask_dir,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        logger.info(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    except Exception as e:
        logger.error(f'Error setting up data loaders: {e}')
        return

    # Setup model
    model = SegColorNet(n_classes=2, pretrained=True).to(device)

    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Setup loss function
    criterion = SegmentationColorationLoss(seg_weight=1.0, color_weight=0.5)

    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
        logger.info(f'Resuming training from epoch {start_epoch}')

    # Training loop
    train_losses = []
    val_losses = []

    logger.info(f'Starting training for {args.epochs} epochs')

    for epoch in range(start_epoch, args.epochs):
        try:
            # Train
            train_loss, train_seg_loss, train_color_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )

            # Validate
            val_loss, val_seg_loss, val_color_loss, val_acc, val_miou = validate_epoch(
                model, val_loader, criterion, device
            )

            # Update scheduler
            scheduler.step(val_loss)

            # Log results
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                       f'Val Acc: {val_acc:.4f}, Val mIoU: {val_miou:.4f}')

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_loss, best_checkpoint_path)
                logger.info(f'New best model saved with val loss: {val_loss:.4f}')

            # Save recent checkpoint (for resuming)
            recent_checkpoint_path = os.path.join(args.checkpoint_dir, 'recent_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, recent_checkpoint_path)

        except Exception as e:
            logger.error(f'Error in epoch {epoch}: {e}')
            continue

    logger.info('Training completed!')

    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs - 1, val_loss, final_checkpoint_path)

    logger.info(f'Final model saved to {final_checkpoint_path}')

if __name__ == '__main__':
    main()