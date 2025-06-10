import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_synthetic_brain_images(dest_dir="data/IXI_Dataset/png", num_images=20):
    """
    Create synthetic brain-like images for testing the GAN pipeline.
    These are not real medical images but will allow you to test the complete pipeline.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Creating {num_images} synthetic brain-like images...")
    
    for i in range(num_images):
        # Create a synthetic brain-like image
        img_size = 256
        
        # Create a brain-like shape using ellipses and noise
        x, y = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))
        
        # Main brain shape (ellipse)
        brain_mask = ((x/0.8)**2 + (y/0.9)**2) < 1
        
        # Add some internal structure
        internal_structure = np.sin(x * 5) * np.cos(y * 5) * 0.3
        ventricles = ((x/0.3)**2 + (y/0.4)**2) < 1
        
        # Combine structures
        brain_img = np.zeros((img_size, img_size))
        brain_img[brain_mask] = 0.7 + internal_structure[brain_mask] * 0.3
        brain_img[ventricles] = 0.2
        
        # Add noise for realism
        noise = np.random.normal(0, 0.05, (img_size, img_size))
        brain_img += noise
        
        # Normalize to 0-255
        brain_img = np.clip(brain_img, 0, 1)
        brain_img = (brain_img * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(brain_img, mode='L')
        img.save(os.path.join(dest_dir, f"synthetic_brain_{i+1:03d}.png"))
    
    print(f"Created {num_images} synthetic brain images in {dest_dir}")

def download_sample_brain_images(dest_dir="data/IXI_Dataset/png"):
    """
    Try to download some sample brain images from a public source.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    # Sample brain MRI images from public datasets (these are smaller samples)
    sample_urls = [
        # These are example URLs - in practice you'd need to find actual public datasets
        # "https://example.com/brain1.jpg",
        # "https://example.com/brain2.jpg",
    ]
    
    # Since reliable public URLs are hard to find, we'll create synthetic data
    print("Creating synthetic brain data since public datasets require registration...")
    create_synthetic_brain_images(dest_dir)

if __name__ == "__main__":
    download_sample_brain_images()
