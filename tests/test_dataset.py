# tests/test_dataset.py
import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from PIL import Image
import cv2

# Import from src
import sys
sys.path.append(os.path.abspath('.'))

from src_dataset import ColorSegDataset, get_dataloader

class TestColorSegDataset:
    """Test cases for ColorSegDataset"""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        temp_dir = tempfile.mkdtemp()
        img_dir = os.path.join(temp_dir, 'images')
        mask_dir = os.path.join(temp_dir, 'masks')

        os.makedirs(img_dir)
        os.makedirs(mask_dir)

        yield img_dir, mask_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def create_dummy_data(self, img_dir, mask_dir, num_samples=3):
        """Create dummy test data"""
        for i in range(num_samples):
            # Create dummy RGB image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img_path = os.path.join(img_dir, f'test_{i}.png')
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Create dummy mask
            mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
            mask_path = os.path.join(mask_dir, f'test_{i}.png')
            cv2.imwrite(mask_path, mask)

    def test_dataset_initialization_empty(self, temp_dirs):
        """Test dataset initialization with empty directories"""
        img_dir, mask_dir = temp_dirs
        dataset = ColorSegDataset(img_dir, mask_dir)

        # Should create dummy data automatically
        assert len(dataset) > 0

    def test_dataset_initialization_with_data(self, temp_dirs):
        """Test dataset initialization with real data"""
        img_dir, mask_dir = temp_dirs
        self.create_dummy_data(img_dir, mask_dir, num_samples=5)

        dataset = ColorSegDataset(img_dir, mask_dir)
        assert len(dataset) == 5

    def test_dataset_getitem(self, temp_dirs):
        """Test dataset __getitem__ method"""
        img_dir, mask_dir = temp_dirs
        self.create_dummy_data(img_dir, mask_dir, num_samples=2)

        dataset = ColorSegDataset(img_dir, mask_dir)
        sample = dataset[0]

        # Check sample structure
        expected_keys = ['image', 'L_channel', 'ab_channels', 'mask', 'img_path']
        assert all(key in sample for key in expected_keys)

        # Check tensor shapes
        assert sample['image'].shape == (3, 256, 256)  # RGB image
        assert sample['L_channel'].shape == (1, 256, 256)  # L channel
        assert sample['ab_channels'].shape == (2, 256, 256)  # AB channels
        assert sample['mask'].shape == (256, 256)  # Mask

        # Check data types
        assert sample['image'].dtype == torch.float32
        assert sample['L_channel'].dtype == torch.float32
        assert sample['ab_channels'].dtype == torch.float32
        assert sample['mask'].dtype == torch.long

    def test_dataset_length(self, temp_dirs):
        """Test dataset length"""
        img_dir, mask_dir = temp_dirs
        num_samples = 7
        self.create_dummy_data(img_dir, mask_dir, num_samples)

        dataset = ColorSegDataset(img_dir, mask_dir)
        assert len(dataset) == num_samples

    def test_dataloader_creation(self, temp_dirs):
        """Test dataloader creation"""
        img_dir, mask_dir = temp_dirs
        self.create_dummy_data(img_dir, mask_dir, num_samples=4)

        dataloader = get_dataloader(img_dir, mask_dir, batch_size=2, shuffle=True)

        assert len(dataloader) == 2  # 4 samples / batch_size 2

        # Test one batch
        batch = next(iter(dataloader))

        # Check batch structure
        expected_keys = ['image', 'L_channel', 'ab_channels', 'mask', 'img_path']
        assert all(key in batch for key in expected_keys)

        # Check batch shapes
        assert batch['image'].shape == (2, 3, 256, 256)  # (batch_size, channels, H, W)
        assert batch['L_channel'].shape == (2, 1, 256, 256)
        assert batch['ab_channels'].shape == (2, 2, 256, 256)
        assert batch['mask'].shape == (2, 256, 256)

    def test_color_space_conversion(self, temp_dirs):
        """Test color space conversion in dataset"""
        img_dir, mask_dir = temp_dirs

        # Create a specific test image
        test_img = np.ones((64, 64, 3), dtype=np.uint8) * 128  # Gray image
        test_img[:32, :32] = [255, 0, 0]  # Red square
        test_img[:32, 32:] = [0, 255, 0]  # Green square
        test_img[32:, :32] = [0, 0, 255]  # Blue square

        img_path = os.path.join(img_dir, 'test.png')
        mask_path = os.path.join(mask_dir, 'test.png')

        cv2.imwrite(img_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, np.ones((64, 64), dtype=np.uint8) * 255)

        dataset = ColorSegDataset(img_dir, mask_dir, target_size=(64, 64))
        sample = dataset[0]

        # Check that L channel has reasonable values [0, 1]
        L_channel = sample['L_channel']
        assert torch.all(L_channel >= 0) and torch.all(L_channel <= 1)

        # Check that AB channels have reasonable values [0, 1]
        ab_channels = sample['ab_channels']
        assert torch.all(ab_channels >= 0) and torch.all(ab_channels <= 1)

# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])