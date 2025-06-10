# Medical Image Super-Resolution GAN

This project implements a Generative Adversarial Network (GAN) for super-resolving low-resolution medical images (e.g., MRI scans) to high-resolution images, preserving diagnostic details. The generator is a custom U-Net, and the discriminator is a custom CNN. The pipeline supports custom loss functions (perceptual, L1, adversarial) and is designed for easy experimentation and visualization.

## Features
- **Custom U-Net Generator** for high-quality upsampling
- **CNN Discriminator** for adversarial training
- **Custom Losses:** Perceptual, L1, and adversarial
- **Synthetic Dataset** for quick testing (real datasets can be used with minor changes)
- **Visualization** of low-res, super-res, and ground truth images

## Project Structure
```
models/           # Generator (U-Net) and Discriminator (CNN)
losses/           # Custom loss functions
data/             # Data loading and transforms
results/          # Output images and visualizations
train.py          # Training script
visualize.py      # Visualization script
create_synthetic_dataset.py # Script to generate synthetic data
requirements.txt  # Python dependencies
```

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Create dataset:**
   - To use synthetic data (default):
     ```bash
     python create_synthetic_dataset.py
     ```
   - For real medical images, place PNGs in `data/IXI_Dataset/png/`.

## Training
Run the training script:
```bash
python train.py
```
Sample outputs will be saved in `results/` after each epoch.

## Visualization
To compare low-res, super-resolved, and ground truth images:
```bash
python visualize.py
```
Visualizations are saved in `results/visualizations/`.

## Model Architecture
- **Generator:** U-Net with skip connections, downsampling/upsampling blocks, and final upsampling to match HR size.
- **Discriminator:** PatchGAN-style CNN for distinguishing real/fake HR images.
- **Losses:**
  - *Adversarial Loss:* BCEWithLogitsLoss
  - *L1 Loss:* Pixel-wise L1
  - *Perceptual Loss:* VGG16 feature-based (for 3-channel images; grayscale images are repeated to 3 channels)

## Customization
- To use your own dataset, place PNG/JPG images in a folder and update `data_dir` in `train.py` and `visualize.py`.
- For NIfTI/DICOM, use `convert_nii_to_png.py` or adapt the data loader.

## References
- [IXI Dataset](https://brain-development.org/ixi-dataset/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [SRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

---
