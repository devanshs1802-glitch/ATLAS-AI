import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from dataset import get_dataloaders
def train_model(num_epochs=15, batch_size=64, learning_rate=0.001):
    """
    Trains a fine-tuned ResNet-18 model on the EuroSAT dataset using 
    automatic mixed precision, optimized for NVIDIA RTX 4070 Mobile GPUs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    # Load optimized data loaders from dataset.py
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=batch_size)
    
    # Initialize ResNet-18 with pretrained ImageNet weights
    print("Initializing ResNet-18 architecture with transfer learning weights...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Modify final fully connected layer for 10 EuroSAT land-use classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model = model.to(device)

    # Loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Automatic Mixed Precision scalar for RTX Tensor Cores
    scaler = torch.amp.GradScaler('cuda')

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    best_model_path = os.path.join('models', 'resnet18_atlas.pth')
    
    best_val_acc = 0.0
    start_time = time.time()

    print(f"\nStarting training loop for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scaled backward pass and optimization step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Step learning rate scheduler based on validation loss plateaus
        scheduler.step(epoch_val_loss)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
        )

        # Save best performing model weights locally
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Saved new best model checkpoint to {best_model_path} (Val Acc: {best_val_acc*100:.2f}%)")

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration/60:.2f} minutes.")

    # Final evaluation on the held-out test set
    print("\nEvaluating model on held-out test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    test_acc = correct_test / total_test
    print(f"Final Held-out Test Set Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    train_model(num_epochs=15, batch_size=64, learning_rate=0.001)