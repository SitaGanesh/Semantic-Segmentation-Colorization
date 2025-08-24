# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SegColorNet(nn.Module):
    """
    A neural network that performs both semantic segmentation and image colorization.

    Architecture:
    - Uses ResNet34 as backbone encoder
    - Has two heads: segmentation head and colorization head
    - Takes L channel as input, predicts ab channels and segmentation mask
    """

    def __init__(self, n_classes=2, pretrained=True):
        super(SegColorNet, self).__init__()

        self.n_classes = n_classes

        # Load pretrained ResNet34 and remove final layers
        resnet = models.resnet34(pretrained=pretrained)

        # Remove the final fully connected layer and average pooling
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Get feature dimensions (ResNet34 outputs 512 channels)
        self.encoder_channels = 512

        # Decoder for upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

        # Colorization head - predicts ab channels
        self.color_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh()  # ab channels are typically in range [-1, 1] after normalization
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of decoder and head layers"""
        for m in [self.decoder, self.seg_head, self.color_head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, C, H, W)
               For colorization: C=1 (L channel)
               For normal RGB: C=3 (will use only first channel)

        Returns:
            seg_logits: Segmentation logits of shape (B, n_classes, H, W)
            color_ab: Predicted ab channels of shape (B, 2, H, W)
        """
        # Handle different input channels
        if x.size(1) == 3:
            # If RGB input, convert to grayscale (use as L channel approximation)
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        elif x.size(1) == 1:
            # L channel input, use as is
            pass
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.size(1)}")

        # Repeat L channel to match ResNet input (3 channels)
        x_input = x.repeat(1, 3, 1, 1)

        # Extract features using encoder
        features = self.encoder(x_input)

        # Decode features
        decoded = self.decoder(features)

        # Ensure decoded features match input size
        if decoded.size(-2) != x.size(-2) or decoded.size(-1) != x.size(-1):
            decoded = F.interpolate(decoded, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Segmentation head
        seg_logits = self.seg_head(decoded)

        # Colorization head
        color_ab = self.color_head(decoded)

        return seg_logits, color_ab

    def predict_colors(self, L_channel):
        """
        Predict colorization for L channel input

        Args:
            L_channel: L channel tensor of shape (B, 1, H, W) in range [0, 1]

        Returns:
            rgb_images: Predicted RGB images of shape (B, 3, H, W)
        """
        self.eval()
        with torch.no_grad():
            _, ab_pred = self.forward(L_channel)

            # Convert predictions back to LAB color space
            L = L_channel * 100.0  # Scale back to [0, 100]
            ab = ab_pred * 127.5  # Scale from [-1, 1] to [-127.5, 127.5]

            # Combine L and ab channels
            lab_images = torch.cat([L, ab], dim=1)  # (B, 3, H, W)

            # Convert LAB to RGB (simplified conversion for demonstration)
            rgb_images = self._lab_to_rgb_torch(lab_images)

        return rgb_images

    def _lab_to_rgb_torch(self, lab):
        """
        Simplified LAB to RGB conversion using PyTorch tensors
        Note: This is an approximation. For accurate conversion, use skimage.color
        """
        # Extract L, a, b channels
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

        # Normalize L to [0, 1]
        L_norm = L / 100.0

        # Simplified conversion (not colorimetrically accurate)
        R = torch.clamp(L_norm + 0.002 * a + 0.003 * b, 0, 1)
        G = torch.clamp(L_norm - 0.001 * a - 0.001 * b, 0, 1) 
        B = torch.clamp(L_norm - 0.003 * a + 0.002 * b, 0, 1)

        rgb = torch.cat([R, G, B], dim=1)
        return rgb


class UNetSegColorNet(nn.Module):
    """
    Alternative U-Net based architecture for semantic colorization
    """

    def __init__(self, n_classes=2):
        super(UNetSegColorNet, self).__init__()

        self.n_classes = n_classes

        # Encoder
        self.enc1 = self._make_encoder_block(1, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)

        # Decoder
        self.dec4 = self._make_decoder_block(512, 256)
        self.dec3 = self._make_decoder_block(256 + 256, 128)  # Skip connection
        self.dec2 = self._make_decoder_block(128 + 128, 64)   # Skip connection
        self.dec1 = self._make_decoder_block(64 + 64, 32)     # Skip connection

        # Output heads
        self.seg_head = nn.Conv2d(32, n_classes, kernel_size=1)
        self.color_head = nn.Conv2d(32, 2, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Handle input channels (same as SegColorNet)
        if x.size(1) == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder with skip connections
        d4 = F.interpolate(self.dec4(e4), scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.dec3(torch.cat([d4, e3], dim=1)), scale_factor=2, mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.dec2(torch.cat([d3, e2], dim=1)), scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # Output heads
        seg_logits = self.seg_head(d1)
        color_ab = torch.tanh(self.color_head(d1))

        return seg_logits, color_ab


# Test the model
if __name__ == "__main__":
    print("Testing SegColorNet...")

    # Create model
    model = SegColorNet(n_classes=2, pretrained=False)  # Set to False for testing
    model.eval()

    # Test with L channel input
    batch_size, height, width = 2, 256, 256
    L_input = torch.randn(batch_size, 1, height, width)

    with torch.no_grad():
        seg_logits, color_ab = model(L_input)

    print(f"Input shape: {L_input.shape}")
    print(f"Segmentation output shape: {seg_logits.shape}")
    print(f"Colorization output shape: {color_ab.shape}")

    # Test prediction function
    rgb_pred = model.predict_colors(L_input)
    print(f"Predicted RGB shape: {rgb_pred.shape}")

    # Test with RGB input
    rgb_input = torch.randn(batch_size, 3, height, width)
    with torch.no_grad():
        seg_logits2, color_ab2 = model(rgb_input)

    print(f"RGB input shape: {rgb_input.shape}")
    print(f"Segmentation output shape (RGB input): {seg_logits2.shape}")
    print(f"Colorization output shape (RGB input): {color_ab2.shape}")

    print("\nTesting UNetSegColorNet...")
    unet_model = UNetSegColorNet(n_classes=2)
    unet_model.eval()

    with torch.no_grad():
        seg_logits3, color_ab3 = unet_model(L_input)

    print(f"U-Net Segmentation output shape: {seg_logits3.shape}")
    print(f"U-Net Colorization output shape: {color_ab3.shape}")

    print("Model tests completed successfully!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nSegColorNet - Total parameters: {total_params:,}")
    print(f"SegColorNet - Trainable parameters: {trainable_params:,}")

    unet_total_params = sum(p.numel() for p in unet_model.parameters())
    unet_trainable_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
    print(f"UNetSegColorNet - Total parameters: {unet_total_params:,}")
    print(f"UNetSegColorNet - Trainable parameters: {unet_trainable_params:,}")