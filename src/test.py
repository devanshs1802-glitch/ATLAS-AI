import os
import torch
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, classification_report, hamming_loss
from torchvision.models import resnet18
from dataset import get_dataloaders, BEN_V2_CLASSES

def evaluate_model():
    """
    Evaluates the trained ResNet-18 multi-label model on the held-out test set
    and reports core metrics for urban sprawl and environmental telemetry.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load test dataloader
    _, _, test_loader, classes = get_dataloaders()

    # Initialize architecture and load best saved weights
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    
    model_path = os.path.join('models', 'resnet18_bigearthnet_v2.pth')
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}. Please complete training first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("\nRunning inference on held-out test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Apply sigmoid to convert logits to probabilities [0, 1]
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Binarize predictions using a standard 0.5 threshold for discrete metrics
    binary_preds = (all_preds >= 0.5).astype(int)

    # --- Compute Evaluation Metrics ---
    macro_map = average_precision_score(all_targets, all_preds, average='macro')
    micro_map = average_precision_score(all_targets, all_preds, average='micro')
    
    macro_f1 = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, binary_preds, average='micro', zero_division=0)
    h_loss = hamming_loss(all_targets, binary_preds)

    print("\n" + "="*50)
    print(" 📊 ATLAS-AI BIGEARTHNET V2.0 EVALUATION REPORT")
    print("="*50)
    print(f"• Macro Mean Average Precision (mAP): {macro_map*100:.2f}%")
    print(f"• Micro Mean Average Precision (mAP): {micro_map*100:.2f}%")
    print(f"• Macro F1-Score:                     {macro_f1*100:.2f}%")
    print(f"• Micro F1-Score:                     {micro_f1*100:.2f}%")
    print(f"• Hamming Loss (Error Rate):          {h_loss*100:.2f}%")
    print("="*50)

    # Class-by-class breakdown report
    print("\nDetailed Class-by-Class Performance:")
    report = classification_report(
        all_targets, 
        binary_preds, 
        target_names=classes, 
        zero_division=0
    )
    print(report)

if __name__ == "__main__":
    evaluate_model()