# src/evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging
from tqdm import tqdm

# Import our modules
from src_dataset import get_dataloader
from src_model import SegColorNet
from src_utils import load_checkpoint, show_results, calculate_metrics, colorization_metrics, create_project_structure

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, device, save_dir=None, max_visualizations=10):
    """
    Evaluate the model on a dataset

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run evaluation on
        save_dir: Directory to save visualizations
        max_visualizations: Maximum number of result visualizations to save

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    all_seg_metrics = []
    all_color_metrics = []
    visualized_count = 0

    logger.info(f'Evaluating on {len(dataloader)} batches...')

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluation')):
            try:
                # Move data to device
                images = batch['image'].to(device)  # RGB images
                L_channel = batch['L_channel'].to(device)  # L channel input
                ab_target = batch['ab_channels'].to(device)  # Ground truth ab channels
                seg_target = batch['mask'].to(device)  # Ground truth segmentation

                # Forward pass
                seg_logits, color_pred = model(L_channel)

                # Convert to numpy for metrics calculation
                seg_pred_np = torch.argmax(seg_logits, dim=1).cpu().numpy()
                seg_target_np = seg_target.cpu().numpy()

                # Calculate segmentation metrics
                for i in range(seg_pred_np.shape[0]):
                    seg_metrics = calculate_metrics(seg_pred_np[i], seg_target_np[i])
                    all_seg_metrics.append(seg_metrics)

                # Calculate colorization metrics
                # Convert predictions and targets to RGB for comparison
                batch_size = images.size(0)
                for i in range(batch_size):
                    try:
                        # Get L channel and predicted ab channels
                        L_single = L_channel[i:i+1]  # Keep batch dimension
                        ab_pred_single = color_pred[i:i+1]

                        # Convert to RGB using model's prediction method
                        rgb_pred = model.predict_colors(L_single)
                        rgb_pred_np = rgb_pred[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)

                        # Get ground truth RGB
                        rgb_gt_np = images[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)

                        # Calculate colorization metrics
                        color_metrics = colorization_metrics(rgb_pred_np, rgb_gt_np)
                        all_color_metrics.append(color_metrics)

                    except Exception as e:
                        logger.warning(f'Error calculating colorization metrics for sample {i}: {e}')
                        continue

                # Save visualizations for first few batches
                if save_dir and visualized_count < max_visualizations:
                    save_path = os.path.join(save_dir, f'results_batch_{batch_idx}.png')

                    try:
                        show_results(
                            images, seg_logits, color_pred, seg_target,
                            save_path=save_path,
                            max_images=min(4, images.size(0))
                        )
                        visualized_count += 1

                    except Exception as e:
                        logger.warning(f'Error saving visualization for batch {batch_idx}: {e}')

            except Exception as e:
                logger.error(f'Error processing batch {batch_idx}: {e}')
                continue

    # Calculate average metrics
    results = {}

    if all_seg_metrics:
        results['segmentation'] = {
            'accuracy': np.mean([m['accuracy'] for m in all_seg_metrics]),
            'mean_iou': np.mean([m['mean_iou'] for m in all_seg_metrics]),
            'accuracy_std': np.std([m['accuracy'] for m in all_seg_metrics]),
            'mean_iou_std': np.std([m['mean_iou'] for m in all_seg_metrics])
        }

    if all_color_metrics:
        results['colorization'] = {
            'mse': np.mean([m['mse'] for m in all_color_metrics]),
            'psnr': np.mean([m['psnr'] for m in all_color_metrics]),
            'ssim': np.mean([m['ssim'] for m in all_color_metrics]),
            'mse_std': np.std([m['mse'] for m in all_color_metrics]),
            'psnr_std': np.std([m['psnr'] for m in all_color_metrics]),
            'ssim_std': np.std([m['ssim'] for m in all_color_metrics])
        }

    return results

def single_image_inference(model, image_path, device, save_path=None):
    """
    Perform inference on a single image

    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run inference on
        save_path: Path to save result

    Returns:
        Dictionary containing results
    """
    import cv2
    from skimage import color

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    # Convert to LAB
    img_lab = color.rgb2lab(img / 255.0)
    L = img_lab[:, :, 0:1] / 100.0  # Normalize L channel

    # Convert to tensor
    L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        seg_logits, color_ab = model(L_tensor)

        # Get segmentation prediction
        seg_pred = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()

        # Get colorized image
        rgb_colorized = model.predict_colors(L_tensor)
        rgb_colorized_np = rgb_colorized[0].cpu().numpy().transpose(1, 2, 0)

    # Visualize results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img / 255.0)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(L.squeeze(), cmap='gray')
    axes[1].set_title('L Channel (Input)')
    axes[1].axis('off')

    axes[2].imshow(seg_pred, cmap='viridis')
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    axes[3].imshow(rgb_colorized_np)
    axes[3].set_title('Colorized Result')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'Result saved to {save_path}')

    plt.show()

    return {
        'original': img / 255.0,
        'L_channel': L.squeeze(),
        'segmentation': seg_pred,
        'colorized': rgb_colorized_np
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate Semantic Colorization Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/results', help='Output directory for results')
    parser.add_argument('--single-image', type=str, default='', help='Path to single image for inference')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--max-vis', type=int, default=10, help='Maximum number of visualization batches')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f'Using device: {device}')

    # Load model
    model = SegColorNet(n_classes=2, pretrained=False).to(device)

    if not os.path.exists(args.checkpoint):
        logger.error(f'Checkpoint file not found: {args.checkpoint}')
        return

    epoch, loss = load_checkpoint(model, None, args.checkpoint, device)
    logger.info(f'Loaded model from epoch {epoch} with loss {loss:.4f}')

    # Single image inference
    if args.single_image:
        if not os.path.exists(args.single_image):
            logger.error(f'Image file not found: {args.single_image}')
            return

        logger.info(f'Performing inference on single image: {args.single_image}')

        save_path = os.path.join(args.output_dir, 'single_image_result.png')
        try:
            result = single_image_inference(model, args.single_image, device, save_path)
            logger.info('Single image inference completed')
        except Exception as e:
            logger.error(f'Error in single image inference: {e}')

        return

    # Dataset evaluation
    img_dir = os.path.join(args.data_dir, 'images')
    mask_dir = os.path.join(args.data_dir, 'masks')

    try:
        dataloader = get_dataloader(
            img_dir, mask_dir,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        logger.info(f'Evaluating on {len(dataloader)} batches')

        # Evaluate model
        results = evaluate_model(
            model, dataloader, device, 
            save_dir=args.output_dir,
            max_visualizations=args.max_vis
        )

        # Print results
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)

        if 'segmentation' in results:
            seg_results = results['segmentation']
            logger.info("SEGMENTATION METRICS:")
            logger.info(f"  Accuracy: {seg_results['accuracy']:.4f} ± {seg_results['accuracy_std']:.4f}")
            logger.info(f"  Mean IoU: {seg_results['mean_iou']:.4f} ± {seg_results['mean_iou_std']:.4f}")

        if 'colorization' in results:
            color_results = results['colorization']
            logger.info("\nCOLORIZATION METRICS:")
            logger.info(f"  MSE: {color_results['mse']:.6f} ± {color_results['mse_std']:.6f}")
            logger.info(f"  PSNR: {color_results['psnr']:.2f} ± {color_results['psnr_std']:.2f}")
            logger.info(f"  SSIM: {color_results['ssim']:.4f} ± {color_results['ssim_std']:.4f}")

        logger.info("="*50)

        # Save results to file
        results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*50 + "\n")

            if 'segmentation' in results:
                seg_results = results['segmentation']
                f.write("SEGMENTATION METRICS:\n")
                f.write(f"  Accuracy: {seg_results['accuracy']:.4f} ± {seg_results['accuracy_std']:.4f}\n")
                f.write(f"  Mean IoU: {seg_results['mean_iou']:.4f} ± {seg_results['mean_iou_std']:.4f}\n")

            if 'colorization' in results:
                color_results = results['colorization']
                f.write("\nCOLORIZATION METRICS:\n")
                f.write(f"  MSE: {color_results['mse']:.6f} ± {color_results['mse_std']:.6f}\n")
                f.write(f"  PSNR: {color_results['psnr']:.2f} ± {color_results['psnr_std']:.2f}\n")
                f.write(f"  SSIM: {color_results['ssim']:.4f} ± {color_results['ssim_std']:.4f}\n")

        logger.info(f'Results saved to {results_file}')

    except Exception as e:
        logger.error(f'Error during evaluation: {e}')

if __name__ == '__main__':
    main()
    