import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Path to the dataset
dataset_dir = '/content/your_dataset_directory'

# Check the structure
for root, dirs, files in os.walk(dataset_dir):
    print(f"Found {len(files)} files in {root}")

# Load the dataset
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
