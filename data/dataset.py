import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, lr_size=(64, 64), hr_size=(256, 256)):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.lr_transform = T.Compose([
            T.Resize(lr_size),
            T.ToTensor(),
        ])
        self.hr_transform = T.Compose([
            T.Resize(hr_size),
            T.ToTensor(),
        ])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('L')
        lr_img = self.lr_transform(img)
        hr_img = self.hr_transform(img)
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        return lr_img, hr_img
