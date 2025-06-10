import os
import nibabel as nib
import numpy as np
from PIL import Image

def nii_to_png(nii_path, out_dir, axis=2, limit_slices=10):
    os.makedirs(out_dir, exist_ok=True)
    img = nib.load(nii_path)
    data = img.get_fdata()
    # Normalize to 0-255
    data = (255 * (data - np.min(data)) / (np.ptp(data))).astype(np.uint8)
    # Save up to limit_slices slices along the specified axis
    num_slices = min(data.shape[axis], limit_slices)
    for i in range(num_slices):
        if axis == 0:
            slice_ = data[i, :, :]
        elif axis == 1:
            slice_ = data[:, i, :]
        else:
            slice_ = data[:, :, i]
        im = Image.fromarray(slice_)
        im.save(os.path.join(out_dir, f"{os.path.basename(nii_path).replace('.nii.gz','')}_slice_{i}.png"))
    print(f"Saved {num_slices} PNG slices from {nii_path} to {out_dir}")

def convert_all_nii_in_folder(nii_dir="data/IXI_Dataset", out_dir="data/IXI_Dataset/png", axis=2, limit_slices=10):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(nii_dir):
        if fname.endswith(".nii.gz"):
            nii_to_png(os.path.join(nii_dir, fname), out_dir, axis=axis, limit_slices=limit_slices)

if __name__ == "__main__":
    convert_all_nii_in_folder()
