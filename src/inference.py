import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define EuroSAT classes matching training order
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

def load_inference_model():
    """
    Initializes the ResNet-18 architecture and loads the saved ATLAS weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18_architecture(len(CLASSES))
    
    model_path = os.path.join('models', 'resnet18_atlas.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Please run train.py first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def resnet18_architecture(num_classes):
    """
    Reconstructs the ResNet-18 backbone with the custom 10-class linear head.
    """
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def get_image_transform():
    """
    Standard preprocessing transform matching training specifications.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_image(image_path):
    """
    Performs inference on a single image path, returning top class probabilities 
    and a calculated ATLAS Urban Sprawl Risk Score.
    """
    model, device = load_inference_model()
    transform = get_image_transform()
    
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}")
        return None, None

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    prob_dict = {CLASSES[i]: probabilities[i].item() for i in range(len(CLASSES))}
    sorted_preds = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate ATLAS Urban Sprawl Risk Score based on built-up classifications
    urban_classes = {'Residential', 'Industrial', 'Highway'}
    sprawl_score = sum(prob_dict[c] for c in urban_classes) * 100
    
    return sorted_preds, sprawl_score

if __name__ == "__main__":
    print("ATLAS Inference Engine Initialized.")
    
    sample_files = [
        ('Ramhan Island Baseline (2016)', 'sample_tiles/ramhan_2016.png'),
        ('Ramhan Island Expansion (2026)', 'sample_tiles/ramhan_2026.png')
    ]
    
    for name, path in sample_files:
        if os.path.exists(path):
            print(f"\nAnalyzing: {name} ({path})")
            preds, sprawl_score = predict_image(path)
            if preds:
                print(f"ATLAS Urban Sprawl Risk Score: {sprawl_score:.2f}%")
                print("Top 3 Class Predictions:")
                for class_name, prob in preds[:3]:
                    print(f"  {class_name}: {prob * 100:.2f}%")
        else:
            print(f"\nSample file not found: {path}. Skipping preset test.")