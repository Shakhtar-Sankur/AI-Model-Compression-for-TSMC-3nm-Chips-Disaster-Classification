import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import requests
import zipfile
from pathlib import Path

class DisasterDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, split: str = 'train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.classes = ['fire', 'flood', 'earthquake', 'hurricane']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / self.split / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    samples.append((str(img_path), self.class_to_idx[class_name]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class SyntheticDisasterDataset(Dataset):
    def __init__(self, num_samples: int = 1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.classes = ['fire', 'flood', 'earthquake', 'hurricane']
        self.data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> List[Tuple[np.ndarray, int]]:
        data = []
        samples_per_class = self.num_samples // 4
        np.random.seed(42)
        
        for class_idx, class_name in enumerate(self.classes):
            for _ in range(samples_per_class):
                if class_name == 'fire':
                    img = np.random.rand(224, 224, 3)
                    img[:, :, 0] = np.clip(img[:, :, 0] + 0.5, 0, 1)
                    img[:, :, 1] = np.clip(img[:, :, 1] + 0.3, 0, 1)
                    img[:, :, 2] = img[:, :, 2] * 0.2
                elif class_name == 'flood':
                    img = np.random.rand(224, 224, 3)
                    img[:, :, 0] = img[:, :, 0] * 0.3
                    img[:, :, 1] = img[:, :, 1] * 0.4
                    img[:, :, 2] = np.clip(img[:, :, 2] + 0.4, 0, 1)
                elif class_name == 'earthquake':
                    img = np.random.rand(224, 224, 3) * 0.6
                    img[:, :, :] = np.clip(img[:, :, :] + 0.2, 0, 0.8)
                else:
                    img = np.random.rand(224, 224, 3)
                    center = (112, 112)
                    y, x = np.ogrid[:224, :224]
                    mask = ((x - center[0])**2 + (y - center[1])**2) < 80**2
                    img[mask] = img[mask] * 0.8 + 0.2
                img += np.random.normal(0, 0.05, img.shape)
                img = np.clip(img, 0, 1)
                data.append((img, class_idx))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_array, label = self.data[idx]
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        if self.transform:
            image = self.transform(image)
        return image, label

class DatasetManager:
    def __init__(self, data_dir: str = "data/disasters"):
        self.data_dir = Path(data_dir)
        self.classes = ['fire', 'flood', 'earthquake', 'hurricane']
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def create_synthetic_dataset(self, num_samples: int = 1000) -> Tuple[Dataset, Dataset]:
        full_dataset = SyntheticDisasterDataset(
            num_samples=num_samples, 
            transform=self.train_transform
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_dataset_copy = SyntheticDisasterDataset(
            num_samples=num_samples, 
            transform=self.val_transform
        )
        val_indices = val_dataset.indices
        val_dataset_final = torch.utils.data.Subset(val_dataset_copy, val_indices)
        return train_dataset, val_dataset_final
    
    def create_dataloaders(self, batch_size: int = 32, num_workers: int = 2) -> Dict[str, DataLoader]:
        train_dataset, val_dataset = self.create_synthetic_dataset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return {
            'train': train_loader,
            'val': val_loader
        }
    
    def load_real_dataset(self, data_path: str) -> Tuple[Dataset, Dataset]:
        data_path = Path(data_path)
        if not data_path.exists():
            return self.create_synthetic_dataset()
        try:
            train_dataset = ImageFolder(
                root=data_path / 'train',
                transform=self.train_transform
            )
            val_dataset = ImageFolder(
                root=data_path / 'val',
                transform=self.val_transform
            )
            return train_dataset, val_dataset
        except Exception as e:
            return self.create_synthetic_dataset()
    
    def visualize_samples(self, dataset: Dataset, num_samples: int = 8):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle('Disaster Classification Samples', fontsize=16)
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        for i, idx in enumerate(indices):
            row = i // 4
            col = i % 4
            image, label = dataset[idx]
            if isinstance(image, torch.Tensor):
                if image.min() < 0:
                    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    image = torch.clamp(image, 0, 1)
                image = image.permute(1, 2, 0).numpy()
            axes[row, col].imshow(image)
            axes[row, col].set_title(f'{self.classes[label]}')
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.savefig('disaster_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def get_class_distribution(self, dataset: Dataset) -> Dict[str, int]:
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in dataset:
            class_counts[self.classes[label]] += 1
        return class_counts
    
    def create_inference_dataset(self, image_paths: List[str]) -> Dataset:
        class InferenceDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                image = Image.open(self.paths[idx]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, self.paths[idx]
        return InferenceDataset(image_paths, self.inference_transform)

if __name__ == "__main__":
    dataset_manager = DatasetManager()
    dataloaders = dataset_manager.create_dataloaders(batch_size=16)
    train_loader = dataloaders['train']
    batch = next(iter(train_loader))
    images, labels = batch
    train_dataset, _ = dataset_manager.create_synthetic_dataset()
    distribution = dataset_manager.get_class_distribution(train_dataset)