import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from dataset import BEN_V2_CLASSES

URBAN_CLASSES = {"Urban fabric", "Industrial or commercial units"}
WATER_CLASSES = {"Marine waters", "Inland waters", "Coastal wetlands"}

def load_model(model_path="models/resnet18_bigearthnet_v2.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(BEN_V2_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def predict_tile(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy() * 100.0

    prob_dict = {BEN_V2_CLASSES[i]: float(probs[i]) for i in range(len(BEN_V2_CLASSES))}
    urban_score = min(100.0, sum(prob_dict[c] for c in URBAN_CLASSES) / len(URBAN_CLASSES))
    water_score = min(100.0, sum(prob_dict[c] for c in WATER_CLASSES) / len(WATER_CLASSES))
    top_classes = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "urban_sprawl_index": urban_score,
        "hydrology_index": water_score,
        "top_predictions": top_classes
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS Single-Tile Multi-Label Inference")
    parser.add_argument("--image", type=str, default="sample_tiles/ramhan_2026.png", help="Path to Sentinel-2 tile")
    parser.add_argument("--weights", type=str, default="models/resnet18_bigearthnet_v2.pth", help="Path to model weights")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found at {args.weights}")

    model, device = load_model(args.weights)
    results = predict_tile(args.image, model, device)

    print(f"\n🛰️ ATLAS Telemetry Report: {args.image}")
    print(f"• Urban Sprawl Index: {results['urban_sprawl_index']:.2f}%")
    print(f"• Hydrology Index:    {results['hydrology_index']:.2f}%")
    print("• Top Detected Land-Cover Classes:")
    for cls_name, conf in results["top_predictions"]:
        print(f"  - {cls_name}: {conf:.2f}%")
