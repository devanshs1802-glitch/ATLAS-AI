import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from dataset import get_dataloaders
from sklearn.metrics import average_precision_score
import numpy as np

def train_model(num_epochs=12, batch_size=64, learning_rate=0.001):
    """
    Trains fine-tuned ResNet-18 on local BigEarthNet v2.0 dataset using 
    PyTorch Automatic Mixed Precision and multi-label BCEWithLogitsLoss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    # Load dataloaders from local files
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=batch_size)
    
    print("Initializing ResNet-18 architecture with transfer learning weights...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Modify final fully connected layer for 19 BigEarthNet multi-label classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model = model.to(device)

    # Multi-label binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Automatic Mixed Precision scalar for RTX Tensor Cores
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs('models', exist_ok=True)
    best_model_path = os.path.join('models', 'resnet18_bigearthnet_v2.pth')
    
    best_val_loss = float('inf')
    start_time = time.time()

    print(f"\nStarting training loop for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scaled backward pass and step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # --- Validation & mAP Evaluation Phase ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                # Apply sigmoid to extract probabilities for mAP scoring
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate Mean Average Precision (mAP) across all 19 land-cover classes
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        try:
            val_map = average_precision_score(all_targets, all_preds, average='macro')
        except ValueError:
            val_map = 0.0

        # Adjust learning rate on validation loss plateau
        scheduler.step(epoch_val_loss)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"Val mAP: {val_map*100:.2f}%"
        )

        # Save best performing model weights locally
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Saved new best model checkpoint to {best_model_path} (mAP: {val_map*100:.2f}%)")

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration/60:.2f} minutes.")

if __name__ == "__main__":
    train_model(num_epochs=12, batch_size=64, learning_rate=0.001)
