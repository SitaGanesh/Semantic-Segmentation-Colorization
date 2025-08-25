# Semantic Colorization with Deep Learning

A comprehensive deep learning project that combines **semantic segmentation** and **image colorization** in a single neural network. This system can take grayscale images and simultaneously:
1. **Segment objects** (identify foreground vs background)
2. **Colorize the image** (add realistic colors)

![Project Banner](https://img.shields.io/badge/Deep%20Learning-Semantic%20Colorization-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

## 🎯 What is Semantic Colorization?

**Semantic Colorization** is an advanced computer vision technique that goes beyond simple colorization. Instead of just adding colors to grayscale images, it:

- **Understands the scene** by segmenting different objects
- **Applies contextually appropriate colors** based on object understanding
- **Maintains spatial consistency** across similar objects
- **Learns from paired examples** of color images and their segmentation masks

### Why is this Different from Regular Colorization?

| Regular Colorization | Semantic Colorization |
|---------------------|----------------------|
| Only predicts colors | Predicts colors + object boundaries |
| No understanding of objects | Understands what objects are where |
| May color inconsistently | Colors objects uniformly |
| Single-task learning | Multi-task learning (better features) |

## 🏗️ Technical Architecture

### Core Models Used

#### 1. **ResNet34 Backbone** 
```
Purpose: Feature extraction from input images
Why ResNet34?
- ✅ Pre-trained on ImageNet (transfer learning)
- ✅ Skip connections prevent vanishing gradients
- ✅ Optimal balance between depth and computational efficiency
- ✅ 34 layers provide rich feature representations
- ✅ Well-established architecture for computer vision
```

#### 2. **Encoder-Decoder Architecture**
```
Encoder (ResNet34): Extracts hierarchical features
    Input Image (256×256×1) → Features (8×8×512)
    
Decoder (Custom): Upsamples features back to image size
    Features (8×8×512) → Output (256×256×C)
    
Dual Heads: Two separate output branches
    - Segmentation Head: Outputs class probabilities
    - Colorization Head: Outputs color channels
```

#### 3. **Alternative U-Net Architecture**
```
Purpose: Alternative architecture with skip connections
Why U-Net?
- ✅ Skip connections preserve spatial details
- ✅ Symmetrical encoder-decoder design
- ✅ Excellent for pixel-level predictions
- ✅ Combines low-level and high-level features
```

### Color Space: LAB vs RGB

The project uses **LAB color space** instead of RGB:

```
LAB Color Space Benefits:
- L Channel: Lightness (0-100) - what we input to the model
- A Channel: Green-Red axis (-127 to +127)
- B Channel: Blue-Yellow axis (-127 to +127)

Why LAB?
✅ Separates brightness from color information
✅ More perceptually uniform than RGB
✅ Easier for neural networks to learn color relationships
✅ Industry standard for colorization tasks
✅ Better gradient flow during training
```

### Loss Function Design

**Combined Multi-Task Loss:**
```python
Total Loss = λ₁ × Segmentation Loss + λ₂ × Colorization Loss

Segmentation Loss: CrossEntropyLoss
- Measures how well the model segments objects
- Pixel-wise classification accuracy

Colorization Loss: MSE Loss  
- Measures color prediction accuracy
- Applied in LAB color space
- Focuses on foreground pixels (masked loss)
```

## 📁 Project Structure Deep Dive

```
Semantic-Segmentation-Colorization/
├── 📁 data/
│   ├── raw/                  # 🎨 Your colorful training images
│   ├── masks/                # 🎯 Ground-truth segmentation masks  
│   └── processed/            # 🔧 Preprocessed data (auto-generated)
│
├── 📁 src/                   # 🧠 Core source code
│   ├── dataset.py           # 📊 Data loading & LAB conversion
│   ├── model.py             # 🏗️ Neural network architectures
│   ├── train.py             # 🚂 Training loop & optimization
│   ├── evaluate.py          # 📈 Model evaluation & metrics
│   ├── gui.py               # 🖥️ User-friendly GUI application
│   └── utils.py             # 🔧 Helper functions & utilities
│
├── 📁 notebooks/            # 🔬 Experimental & analysis
│   └── exploration.ipynb    # 🧪 Interactive testing & visualization
│
├── 📁 outputs/              # 💾 Generated results
│   ├── checkpoints/         # 🏁 Trained model weights
│   └── results/             # 🎨 Colorized output images
│
├── 📁 tests/                # ✅ Unit tests
├── requirements.txt         # 📦 Python dependencies
├── README.md               # 📖 This comprehensive guide
└── .gitignore              # 🚫 Files to ignore in version control
```

## 🚀 Complete Setup Guide (Beginner-Friendly)

### Step 1: System Requirements

**Minimum Requirements:**
- **Python:** 3.8 or higher
- **RAM:** 8GB (16GB recommended)
- **Storage:** 5GB free space
- **GPU:** Optional but recommended (NVIDIA with CUDA support)

**Check your Python version:**
```bash
python --version
# Should show Python 3.8.x or higher
```

### Step 2: Clone the Repository
```bash
# Open terminal/command prompt and run:
git clone https://github.com/SitaGanesh/Semantic-Segmentation-Colorization.git
cd Semantic-Segmentation-Colorization

# Verify you're in the right directory:
ls
# You should see: data/ src/ notebooks/ requirements.txt README.md
```

### Step 3: Create Virtual Environment

**Why Virtual Environment?**
- Isolates project dependencies from system Python
- Prevents version conflicts between projects
- Makes project portable and reproducible

```bash
# Create virtual environment
python -m venv colorizer_env

# Activate virtual environment
# On Windows:
colorizer_env\Scripts\activate

# On macOS/Linux:
source colorizer_env/bin/activate

# You should see (colorizer_env) in your terminal prompt
```

### Step 4: Install Dependencies

```bash
# Make sure virtual environment is activated
# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt

# This will install:
# - PyTorch (deep learning framework)
# - torchvision (computer vision utilities)
# - OpenCV (image processing)
# - scikit-image (color space conversion)
# - matplotlib (visualization)
# - Pillow (image handling)
# - tkinter (GUI framework)
# - numpy (numerical computing)
# - tqdm (progress bars)
# - jupyter (notebook environment)
```

### Step 5: Prepare Your Data

**Option A: Use Your Own Images**
```bash
# Place your colorful images in:
data/raw/

# Create corresponding masks in:
data/masks/

# Mask format: Binary images (black=background, white=foreground)
```

**Option B: Let the System Create Dummy Data**
```bash
# The system will automatically create dummy data if no images are found
# This is perfect for testing the setup
```

### Step 6: Test the Installation

```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to: notebooks/exploration.ipynb
# Run all cells to test the installation
```

## 🎮 How to Use the Project

### Method 1: GUI Application (Easiest)

```bash
# Activate virtual environment (if not already active)
source colorizer_env/bin/activate  # macOS/Linux
# OR
colorizer_env\Scripts\activate     # Windows

# Launch the GUI
python src/gui.py
```

**GUI Features:**
- 📁 **Load Image:** Browse and select any image
- 🎨 **Colorize:** AI-powered colorization with segmentation
- 👁️ **Preview:** See original, segmentation, and colorized results
- 💾 **Save:** Export colorized images
- 🔄 **Reset:** Clear and start over

### Method 2: Train Your Own Model

```bash
# Basic training (uses dummy data if no custom data provided)
python src/train.py --epochs 20 --batch-size 8

# Advanced training with custom parameters
python src/train.py \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.0002 \
    --data-dir data/processed \
    --checkpoint-dir outputs/checkpoints
```

**Training Parameters Explained:**
- `--epochs`: How many times to see the entire dataset
- `--batch-size`: How many images to process simultaneously
- `--lr`: Learning rate (how fast the model learns)
- `--data-dir`: Where your training data is located
- `--checkpoint-dir`: Where to save trained models

### Method 3: Evaluate a Trained Model

```bash
# Evaluate on test dataset
python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pth

# Process a single image
python src/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --single-image path/to/your/image.jpg \
    --output-dir outputs/results
```

### Method 4: Interactive Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/exploration.ipynb
# Follow the interactive cells to:
# - Test dataset loading
# - Visualize color space conversions  
# - Test model architecture
# - Run inference on sample images
# - Analyze results
```

## 🧠 Understanding the Deep Learning Pipeline

### 1. Data Processing Pipeline

```
Input RGB Image (256×256×3)
           ↓
Convert to LAB Color Space
           ↓
Extract L Channel (Grayscale) → Model Input
Extract AB Channels → Ground Truth Target
           ↓
Resize & Normalize
           ↓
Convert to PyTorch Tensors
```

### 2. Model Forward Pass

```
L Channel Input (B×1×H×W)
           ↓
Repeat to 3 Channels (ResNet expects RGB)
           ↓
ResNet34 Encoder → Features (B×512×H/32×W/32)
           ↓
Custom Decoder → Upsampled Features (B×64×H×W)
           ↓
      Split into Two Heads
           ↓                    ↓
Segmentation Head          Colorization Head
    (B×2×H×W)                (B×2×H×W)
Class Probabilities         AB Channels
```

### 3. Loss Calculation & Training

```
Predictions: seg_logits, color_ab
Ground Truth: seg_masks, ab_channels
           ↓
Segmentation Loss = CrossEntropy(seg_logits, seg_masks)
Colorization Loss = MSE(color_ab, ab_channels)
           ↓
Total Loss = λ₁ × seg_loss + λ₂ × color_loss
           ↓
Backward Pass → Update Weights
```

## 📊 Evaluation Metrics Explained

### Segmentation Metrics
- **Accuracy:** Percentage of correctly classified pixels
- **Mean IoU:** Average Intersection over Union across classes
- **Class IoU:** IoU for each individual class

### Colorization Metrics  
- **MSE:** Mean Squared Error in RGB space
- **PSNR:** Peak Signal-to-Noise Ratio (higher = better)
- **SSIM:** Structural Similarity Index (0-1, higher = better)

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Run from project root directory, not from notebooks/
cd semantic-colorizer  # Make sure you're in project root
python src/gui.py
```

#### 2. CUDA/GPU Issues
```bash
# Error: CUDA out of memory
# Solution: Reduce batch size
python src/train.py --batch-size 4  # Instead of 8

# Error: CUDA not available
# Solution: The code automatically falls back to CPU
# Check with: python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Data Loading Issues
```bash
# Error: No valid image-mask pairs found
# Solution: Either add your data or let system create dummy data
# System will automatically create dummy data for testing
```

#### 4. Memory Issues
```bash
# Error: Out of memory during training
# Solutions:
python src/train.py --batch-size 2 --num-workers 0
# Or train on CPU: --device cpu
```

#### 5. GUI Issues
```bash
# Error: GUI doesn't start
# Solution: Check tkinter installation
python -c "import tkinter; print('Tkinter available')"

# If not available:
# Ubuntu/Debian: sudo apt-get install python3-tk
# Windows/macOS: Usually comes with Python
```

## 🎯 Model Performance & Expectations

### What to Expect

**With Dummy Data (Testing):**
- ✅ System runs without errors
- ✅ GUI functions properly  
- ✅ Basic colorization (not realistic)
- ✅ Segmentation shows learned patterns

**With Real Data (10,000+ images):**
- ✅ Realistic colorization
- ✅ Accurate object segmentation
- ✅ Consistent color application
- ✅ Generalizes to new images

**Training Time Estimates:**
- **CPU Only:** ~2-4 hours per epoch
- **GPU (GTX 1060):** ~15-30 minutes per epoch  
- **GPU (RTX 3080):** ~5-10 minutes per epoch

## 🔬 Advanced Usage

### Custom Dataset Preparation

```python
# Your image structure should be:
data/
├── raw/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── masks/
    ├── image001.png  # Binary mask for image001.jpg
    ├── image002.png  # Binary mask for image002.jpg
    └── ...

# Mask Requirements:
# - Same filename as corresponding image (different extension OK)
# - Binary image: 0 = background, 255 = foreground
# - Same dimensions as source image (or will be automatically resized)
```

### Model Customization

```python
# In src/model.py, you can modify:

# 1. Change backbone architecture
resnet = models.resnet50(pretrained=True)  # Use ResNet50 instead

# 2. Adjust number of classes
model = SegColorNet(n_classes=3)  # For 3-class segmentation

# 3. Modify loss weights
criterion = SegmentationColorationLoss(seg_weight=2.0, color_weight=1.0)
```

### Hyperparameter Tuning

```bash
# Learning rate schedules
python src/train.py --lr 0.001 --epochs 10    # Fast learning
python src/train.py --lr 0.0001 --epochs 50   # Careful learning

# Batch size effects
python src/train.py --batch-size 32   # Better gradients (needs more memory)
python src/train.py --batch-size 4    # Less memory, noisier gradients

# Loss balancing
# Edit src/train.py: SegmentationColorationLoss(seg_weight=1.0, color_weight=0.5)
```

## 📈 Model Architecture Details

### ResNet34 Backbone Specifications
```
Total Parameters: ~21.8M
Input: 224×224×3 (adapted for 256×256×1 in our case)
Architecture:
- Conv1: 7×7 conv, 64 filters
- Layer1: 3 residual blocks, 64 filters
- Layer2: 4 residual blocks, 128 filters  
- Layer3: 6 residual blocks, 256 filters
- Layer4: 3 residual blocks, 512 filters
Output Features: 8×8×512 (for 256×256 input)
```

### Custom Decoder Specifications
```
Purpose: Upsample features back to input resolution
Architecture:
- ConvTranspose2d: 512→256, stride=2 (16×16×256)
- ConvTranspose2d: 256→128, stride=2 (32×32×128)  
- ConvTranspose2d: 128→64,  stride=2 (64×64×64)
- Bilinear interpolation to final size (256×256×64)

Dual Heads:
1. Segmentation: 64→32→n_classes
2. Colorization: 64→32→2 (AB channels)
```
## Images Walkthrough

![15Image1](https://github.com/user-attachments/assets/f0772334-f59b-4c0a-bdbe-7e6f037628f7)

![16Image2](https://github.com/user-attachments/assets/00a76e7e-df8e-4792-9cde-8fb2ef91447b)

![17Image3](https://github.com/user-attachments/assets/c9c202c8-b9ea-4097-a7ef-9386bf7284bf)

![18Image4](https://github.com/user-attachments/assets/283d5537-2411-4fe7-bcb5-eea394142dca)

![19Image5](https://github.com/user-attachments/assets/e1473be5-1782-481c-82b8-a669d9899b3a)
![20Image6](https://github.com/user-attachments/assets/5767836b-a6e8-472f-847f-02144b872964)
![21Images7](https://github.com/user-attachments/assets/e4fe86d0-d6bc-466b-a000-035e06af2da0)


## 🎨 Understanding Color Science

### Why LAB Color Space?

**RGB Limitations:**
- All three channels contain brightness information
- Not perceptually uniform
- Difficult to separate color from brightness
- Complex for neural networks to learn

**LAB Advantages:**
- L channel: Pure brightness information (0-100)
- A channel: Green-Red axis (-127 to +127)  
- B channel: Blue-Yellow axis (-127 to +127)
- Perceptually uniform color differences
- Easier neural network learning

### Color Space Conversion Pipeline

```python
# During Training:
RGB Image → LAB → Split L, AB → Train on L, predict AB

# During Inference:  
Grayscale/L → Model → Predicted AB → Combine L+AB → LAB → RGB
```

## 🚀 Performance Optimization Tips

### For Training:
```bash
# Use mixed precision (if supported)
python src/train.py --device cuda --batch-size 16

# Increase number of workers
python src/train.py --num-workers 4

# Use larger batch sizes if memory allows
python src/train.py --batch-size 32
```

### For Inference:
```python
# In src/gui.py or src/evaluate.py:
# Model runs in eval mode automatically
# Uses torch.no_grad() for memory efficiency
# Processes single images efficiently
```

### Memory Management:
```python
# The code includes several memory optimizations:
# - Gradient clipping prevents exploding gradients
# - Proper tensor device management
# - Automatic fallback to CPU if GPU unavailable
# - Efficient color space conversions
```

## 🤝 Contributing to the Project

### Code Structure Guidelines
- **dataset.py:** Handle data loading, preprocessing, augmentation
- **model.py:** Define neural network architectures
- **train.py:** Training loop, optimization, checkpointing
- **evaluate.py:** Model evaluation, metrics calculation
- **utils.py:** Helper functions, color space conversions
- **gui.py:** User interface, real-time inference

### Adding New Features
1. **New Model Architecture:** Add to `src/model.py`
2. **New Loss Function:** Modify `src/train.py`
3. **New Metrics:** Add to `src/utils.py`
4. **GUI Improvements:** Enhance `src/gui.py`

---


**Happy Colorizing! 🎨**

---
