import torchvision.transforms as T

def get_transforms():
    return T.Compose([
        T.Normalize(mean=[0.5], std=[0.5])
    ])
