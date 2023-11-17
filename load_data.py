from torchvision import datasets, transforms
import os
from PIL import Image
from torch.utils.data import Dataset

# Define your data transformation pipeline
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Convert images to PyTorch tensors (0-1 range)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image data
])


def load_monet():
    data_dir = os.path.join(os.getcwd(), 'monet_jpg')
    custom_dataset = CustomImageDataset(root_dir=data_dir, transform=data_transforms)
    return custom_dataset


def load_photo():
    data_dir = os.path.join(os.getcwd(), 'photo_jpg')
    custom_dataset = CustomImageDataset(root_dir=data_dir, transform=data_transforms)
    return custom_dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
