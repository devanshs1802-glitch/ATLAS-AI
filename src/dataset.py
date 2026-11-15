import torch
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

def get_dataloaders(data_dir='./data', batch_size=64, val_split=0.15, test_split=0.15, seed=42):
    """
    Downloads the EuroSAT RGB dataset to the local data folder, 
    applies ResNet preprocessing transforms, splits deterministically, 
    and returns optimized PyTorch DataLoaders[cite: 13, 14].
    """
    
    # Preprocessing pipeline: upscale 64x64 tiles to 224x224 and normalize for ResNet[cite: 13, 14]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Automatically downloads locally into ./data if not already present[cite: 13, 14]
    print("Initializing EuroSAT dataset...")
    dataset = EuroSAT(root=data_dir, download=True, transform=transform)
    
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Fixed seed for consistent, reproducible data splits across the team
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )
    
    print(f"Dataset successfully split:")
    print(f"  Training samples:   {len(train_set)}")
    print(f"  Validation samples: {len(val_set)}")
    print(f"  Test samples:       {len(test_set)}")
    print(f"  Total classes:      {len(dataset.classes)}")
    
    # Optimized DataLoaders configured for local RTX 4070 execution
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Classes available: {classes}")
    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch label tensor shape: {labels.shape}")