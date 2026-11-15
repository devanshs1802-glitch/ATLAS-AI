import os
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

# Page Configuration
st.set_page_config(
    page_title="ATLAS | Environmental & Sprawl Monitoring",
    page_icon="🌍",
    layout="wide"
)

# Define Classes
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    
    model_path = os.path.join('models', 'resnet18_atlas.pth')
    if not os.path.exists(model_path):
        return None, device
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def process_image(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    prob_dict = {CLASSES[i]: probabilities[i].item() for i in range(len(CLASSES))}
    sorted_preds = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    urban_classes = {'Residential', 'Industrial', 'Highway'}
    sprawl_score = sum(prob_dict[c] for c in urban_classes) * 100
    
    return sorted_preds, sprawl_score

# UI Header
st.title("🌍 ATLAS: Automated Tracking of Land, Agriculture & Sprawl")
st.markdown("### High-Performance Satellite Environmental & Urban Encroachment Monitoring")

model, device = load_model()

if model is None:
    st.error("Model weights not found at `models/resnet18_atlas.pth`. Please run `python src/train.py` first!")
else:
    # Sidebar Controls
    st.sidebar.header("Configuration & Data Source")
    mode = st.sidebar.radio("Select Mode", ["Preset UAE Test Case (Ramhan Island)", "Upload Custom Tile"])

    target_image = None
    image_name = ""

    if mode == "Preset UAE Test Case (Ramhan Island)":
        preset_choice = st.sidebar.selectbox(
            "Choose Temporal Tile",
            ["Ramhan Island Baseline (2016)", "Ramhan Island Expansion (2026)"]
        )
        if "2016" in preset_choice:
            path = "sample_tiles/ramhan_2016.png"
            image_name = "Ramhan Island 2016 (Sabkha Baseline)"
        else:
            path = "sample_tiles/ramhan_2026.png"
            image_name = "Ramhan Island 2026 (Urban Megaproject)"
            
        if os.path.exists(path):
            target_image = Image.open(path).convert("RGB")
        else:
            st.sidebar.error(f"Preset file not found at {path}.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Sentinel-2 Tile (.png/.jpg)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            target_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

    # Main Analysis Section
    if target_image is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"Satellite View: {image_name}")
            st.image(target_image, use_container_width=True, caption="Sentinel-2 L2A Tile")

        with col2:
            st.subheader("ATLAS Intelligence & Risk Analytics")
            with st.spinner("Running ResNet-18 Inference..."):
                preds, sprawl_score = process_image(target_image, model, device)

            # Display Risk Metric Card
            st.metric(label="Urban Sprawl Risk Score", value=f"{sprawl_score:.2f}%")
            st.progress(int(sprawl_score))

            st.markdown("#### Top Class Predictions")
            for class_name, prob in preds[:5]:
                st.text(f"{class_name:<20} | {prob * 100:.2f}%")
                st.progress(float(prob))

        # Comparative Insight Callout
        st.markdown("---")
        st.info(
            "**Hackathon Note:** This analysis utilizes transfer learning on EuroSAT features "
            "to evaluate structural changes across arid regional zones in Abu Dhabi."
        )
    else:
        st.info("Please select a preset UAE sample tile from the sidebar or upload your own image to begin analysis.")
        