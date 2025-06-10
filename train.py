from models.generator import UNetGenerator
from models.discriminator import Discriminator
from losses.perceptual_loss import PerceptualLoss
from losses.l1_loss import L1Loss
from losses.adversarial_loss import AdversarialLoss
from data.dataset import MedicalImageDataset
from data.transforms import get_transforms
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def save_sample(lr, sr, hr, epoch, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr.squeeze().cpu().numpy(), cmap="gray")
    axs[0].set_title("Low-Res Input")
    axs[1].imshow(sr.squeeze().detach().cpu().numpy(), cmap="gray")
    axs[1].set_title("Super-Resolved")
    axs[2].imshow(hr.squeeze().cpu().numpy(), cmap="gray")
    axs[2].set_title("Ground Truth")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_epoch_{epoch}.png"))
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Paths
    data_dir = "data/IXI_Dataset/png"  # Use synthetic PNG dataset
    batch_size = 4
    num_epochs = 100
    lr = 2e-4
    # Data
    dataset = MedicalImageDataset(data_dir, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Models
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)
    # Losses
    perceptual_loss = PerceptualLoss().to(device)
    l1_loss = L1Loss().to(device)
    adv_loss = AdversarialLoss().to(device)
    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    for epoch in range(num_epochs):
        loop = tqdm(loader, leave=False)
        for lr_img, hr_img in loop:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            # Train Discriminator
            fake_hr = generator(lr_img)
            real_pred = discriminator(hr_img)
            fake_pred = discriminator(fake_hr.detach())
            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)
            d_loss_real = adv_loss(real_pred, real_labels)
            d_loss_fake = adv_loss(fake_pred, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            # Train Generator
            fake_pred = discriminator(fake_hr)
            real_labels_gen = torch.ones_like(fake_pred)
            g_adv = adv_loss(fake_pred, real_labels_gen)
            g_l1 = l1_loss(fake_hr, hr_img)
            g_perc = perceptual_loss(fake_hr.repeat(1,3,1,1), hr_img.repeat(1,3,1,1))
            g_loss = g_adv + 0.01 * g_l1 + 0.006 * g_perc
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
        # Save sample
        save_sample(lr_img[0], fake_hr[0], hr_img[0], epoch+1)

if __name__ == "__main__":
    main()
