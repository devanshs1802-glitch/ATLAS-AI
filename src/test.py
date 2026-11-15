import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def evaluate_model():
    """
    Evaluates the trained ATLAS ResNet-18 model on the held-out test set,
    generating comprehensive classification metrics and confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataloader from dataset.py
    _, _, test_loader, classes = get_dataloaders()

    # Initialize model architecture
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    
    # Path to saved model weights
    model_path = os.path.join('models', 'resnet18_atlas.pth')
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}. Please run train.py first.")
        return

    # Load trained weights onto device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on the test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute overall accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Held-Out Test Accuracy: {acc * 100:.2f}%\n")

    # Detailed per-class precision, recall, and f1-score
    print("Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix representation
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    evaluate_model()