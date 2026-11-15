import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_from_disk, load_dataset

# BigEarthNet v2.0 (reBEN) 19 Official Land-Cover Classes
BEN_V2_CLASSES = [
    "Urban fabric", "Industrial or commercial units", "Arable land", 
    "Permanent crops", "Pastures", "Complex cultivation patterns", 
    "Land principally occupied by agriculture, with significant areas of natural vegetation", 
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", 
    "Mixed forest", "Natural grassland and sparsely vegetated areas", 
    "Moors, heathland and sclerophyllous vegetation", "Transitional woodland, shrub", 
    "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands", 
    "Inland waters", "Marine waters"
]

class BigEarthNetV2Dataset(Dataset):
    def __init__(self, split="train", data_dir="./data", transform=None):
        """
        Loads BigEarthNet v2.0 from the local folder structure.
        """
        print(f"Loading BigEarthNet v2.0 ({split}) from local directory {data_dir}...")
        
        # Pointing directly to the local cache/dataset folder structure shown in your workspace
        data_path = os.path.join(data_dir, "timm__bigearthnet-v2-rgb")
        
        try:
            # Load dataset from local disk path
            self.hf_dataset = load_dataset("timm/bigearthnet-v2-rgb", data_dir=data_path, split=split)
        except Exception:
            # Fallback to loading via standard dataset builder pointing to local files
            self.hf_dataset = load_dataset("timm/bigearthnet-v2-rgb", split=split, cache_dir=data_dir)
            
        self.transform = transform
        self.num_classes = len(BEN_V2_CLASSES)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]  # PIL Image
        
        if self.transform:
            image = self.transform(image)
            
        # Convert multi-hot label indices into a binary vector (19 classes)
        label_indices = item["labels"]
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(label_indices) > 0:
            target[label_indices] = 1.0
            
        return image, target

def get_dataloaders(batch_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_set = BigEarthNetV2Dataset(split="train", data_dir=data_dir, transform=transform)
    val_set = BigEarthNetV2Dataset(split="validation", data_dir=data_dir, transform=transform)
    test_set = BigEarthNetV2Dataset(split="test", data_dir=data_dir, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, BEN_V2_CLASSES