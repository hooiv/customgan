# Placeholder for visualizations and comparison utilities

import os
import matplotlib.pyplot as plt
import torch
from models.generator import UNetGenerator
from data.dataset import MedicalImageDataset
from data.transforms import get_transforms

def visualize_samples(data_dir, checkpoint_path=None, num_samples=5, out_dir="results/visualizations"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MedicalImageDataset(data_dir, transform=get_transforms())
    generator = UNetGenerator().to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    for i in range(num_samples):
        lr_img, hr_img = dataset[i]
        lr_img = lr_img.unsqueeze(0).to(device)
        with torch.no_grad():
            sr_img = generator(lr_img)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(lr_img.squeeze().cpu().numpy(), cmap="gray")
        axs[0].set_title("Low-Res Input")
        axs[1].imshow(sr_img.squeeze().cpu().numpy(), cmap="gray")
        axs[1].set_title("Super-Resolved")
        axs[2].imshow(hr_img.squeeze().cpu().numpy(), cmap="gray")
        axs[2].set_title("Ground Truth")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"visualization_{i+1}.png"))
        plt.close()

if __name__ == "__main__":
    # Example usage: update data_dir and checkpoint_path as needed
    visualize_samples(data_dir="data/IXI_Dataset/png", checkpoint_path=None)
