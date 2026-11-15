import os
import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ATLAS | Geospatial Intelligence & Sprawl Risk",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for an Executive Dark Geospatial Dashboard Look
st.markdown("""
<style>
    /* Main background & typography */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    
    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Custom KPI Metric Cards */
    .kpi-card {
        background-color: #131B2E;
        border: 1px solid #26324B;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
    }
    .badge-high {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .badge-mod {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .badge-low {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }

    /* Section Containers */
    .panel-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F5F9;
        border-bottom: 1px solid #26324B;
        padding-bottom: 8px;
        margin-bottom: 14px;
    }
    
    /* Hide default Streamlit footer/menu clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL & INFERENCE ENGINE
# -----------------------------------------------------------------------------
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

def analyze_tile(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    prob_dict = {CLASSES[i]: probabilities[i].item() * 100 for i in range(len(CLASSES))}
    sorted_preds = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    urban_classes = {'Residential', 'Industrial', 'Highway'}
    sprawl_score = sum(prob_dict[c] for c in urban_classes)
    
    water_classes = {'SeaLake', 'River'}
    water_coverage = sum(prob_dict[c] for c in water_classes)
    
    veg_classes = {'Forest', 'HerbaceousVegetation', 'AnnualCrop', 'PermanentCrop', 'Pasture'}
    veg_coverage = sum(prob_dict[c] for c in veg_classes)
    
    return prob_dict, sorted_preds, sprawl_score, water_coverage, veg_coverage

# -----------------------------------------------------------------------------
# 3. HERO HEADER & SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
model, device = load_model()

st.markdown("""
<div class="hero-box">
    <p class="hero-title">🛰️ ATLAS // Geospatial Change & Risk Intelligence</p>
    <p class="hero-subtitle">Automated Tracking of Land, Agriculture & Sprawl • ResNet-18 Sentinel-2 L2A Inference Engine</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model weights missing at `models/resnet18_atlas.pth`. Please verify the weights path.")
    st.stop()

with st.sidebar:
    st.markdown("### 🎛️ Mission Control")
    st.caption(f"Compute Backend: `{str(device).upper()}` | Test Acc: `97.63%`")
    st.markdown("---")
    
    source_mode = st.radio(
        "Select Telemetry Source",
        ["UAE Case Study: Ramhan Island", "Upload Custom Historical & Modern Tiles"]
    )
    
    img_before, img_after = None, None
    label_before, label_after = "Historical Baseline", "Modern State"

    if source_mode == "Upload Custom Historical & Modern Tiles":
        st.markdown("#### 1. Baseline Imagery (Before)")
        file_before = st.file_uploader("Upload Historical Tile (PNG/JPG)", type=["png", "jpg", "jpeg"], key="before")
        label_before = st.text_input("Baseline Label / Year", value="2016 Baseline")
        
        st.markdown("#### 2. Target Imagery (After)")
        file_after = st.file_uploader("Upload Modern Tile (PNG/JPG)", type=["png", "jpg", "jpeg"], key="after")
        label_after = st.text_input("Modern Label / Year", value="2026 Current")
        
        if file_before:
            img_before = Image.open(file_before).convert("RGB")
        if file_after:
            img_after = Image.open(file_after).convert("RGB")
    else:
        st.info("**Active Region:** Ramhan Island, Abu Dhabi (24.51° N, 54.63° E)\n\nComparing natural coastal sabkha & early earthworks (2016) against completed waterfront megaproject (2026).")
        path_2016 = "sample_tiles/ramhan_2016.png"
        path_2026 = "sample_tiles/ramhan_2026.png"
        label_before = "2016 Sabkha Baseline"
        label_after = "2026 Urban Megaproject"
        if os.path.exists(path_2016) and os.path.exists(path_2026):
            img_before = Image.open(path_2016).convert("RGB")
            img_after = Image.open(path_2026).convert("RGB")
        else:
            st.error("Sample tiles not found in `sample_tiles/`.")

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD ANALYTICS
# -----------------------------------------------------------------------------
if img_before is not None and img_after is not None:
    dict_b, preds_b, sprawl_b, water_b, veg_b = analyze_tile(img_before, model, device)
    dict_a, preds_a, sprawl_a, water_a, veg_a = analyze_tile(img_after, model, device)
    
    sprawl_delta = sprawl_a - sprawl_b
    water_delta = water_a - water_b
    
    # Sprawl Badge Logic
    if sprawl_a >= 75:
        sprawl_badge = f'<span class="badge-high">CRITICAL SPRAWL ({sprawl_delta:+.1f}%)</span>'
    elif sprawl_a >= 40:
        sprawl_badge = f'<span class="badge-mod">MODERATE EXPANSION ({sprawl_delta:+.1f}%)</span>'
    else:
        sprawl_badge = f'<span class="badge-low">LOW URBAN FOOTPRINT ({sprawl_delta:+.1f}%)</span>'

    # Water & Coastal Encroachment Risk Logic
    if water_delta < -5.0 and sprawl_delta > 5.0:
        water_risk_score = min(100.0, abs(water_delta) * 1.5 + (sprawl_a * 0.3))
        water_badge = '<span class="badge-high">HIGH COASTAL RECLAMATION</span>'
    elif water_delta < -2.0:
        water_risk_score = min(100.0, abs(water_delta) * 2.0)
        water_badge = '<span class="badge-mod">SURFACE WATER DEPLETION</span>'
    elif sprawl_a > 80.0 and water_a > 5.0:
        water_risk_score = min(100.0, (sprawl_a * 0.5) + (water_a * 0.5))
        water_badge = '<span class="badge-mod">ELEVATED WATERFRONT LOAD</span>'
    else:
        water_risk_score = max(4.2, abs(water_delta))
        water_badge = '<span class="badge-low">STABLE HYDROLOGICAL BOUNDARY</span>'

    # --- ROW 1: EXECUTIVE KPI CARDS ---
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Urban Sprawl Risk Score</div>
            <div class="kpi-value">{sprawl_a:.2f}%</div>
            {sprawl_badge}
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Water & Coastal Risk</div>
            <div class="kpi-value">{water_risk_score:.1f}%</div>
            {water_badge}
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Surface Water Coverage</div>
            <div class="kpi-value">{water_a:.2f}%</div>
            <span class="badge-low">Δ {water_delta:+.2f}% vs Baseline</span>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Dominant Modern Class</div>
            <div class="kpi-value">{preds_a[0][0]}</div>
            <span class="badge-low">{preds_a[0][1]:.2f}% Confidence</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 2: VISUAL SATELLITE COMPARISON ---
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown(f'<div class="panel-header">🕰️ {label_before}</div>', unsafe_allow_html=True)
        st.image(img_before, use_container_width=True)
        st.caption(f"Built-Up Index: **{sprawl_b:.2f}%** • Water Index: **{water_b:.2f}%** • Vegetation/Agri: **{veg_b:.2f}%**")
        
    with col_right:
        st.markdown(f'<div class="panel-header">🛰️ {label_after}</div>', unsafe_allow_html=True)
        st.image(img_after, use_container_width=True)
        st.caption(f"Built-Up Index: **{sprawl_a:.2f}%** • Water Index: **{water_a:.2f}%** • Vegetation/Agri: **{veg_a:.2f}%**")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 3: DETAILED LAND-USE SHIFT ANALYTICS ---
    tab1, tab2 = st.tabs(["📊 Land-Use Class Shift Chart", "📋 Full Telemetry Matrix"])
    
    with tab1:
        st.markdown("#### Temporal Probability Distribution Shift (%)")
        chart_df = pd.DataFrame({
            "Land-Use Class": CLASSES,
            label_before: [dict_b[c] for c in CLASSES],
            label_after: [dict_a[c] for c in CLASSES]
        }).set_index("Land-Use Class")
        st.bar_chart(chart_df, height=340, color=["#64748B", "#38BDF8"])

    with tab2:
        st.markdown("#### Class-by-Class Quantitative Breakdown")
        table_data = []
        for c in CLASSES:
            b_val = dict_b[c]
            a_val = dict_a[c]
            diff = a_val - b_val
            table_data.append({
                "Land-Use Category": c,
                f"{label_before} (%)": round(b_val, 2),
                f"{label_after} (%)": round(a_val, 2),
                "Net Shift (%)": f"{diff:+.2f}%"
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

elif img_before is not None or img_after is not None:
    st.warning("⏳ Waiting for second tile: Please upload **both** a Before and After satellite image in the sidebar to run temporal comparison.")
else:
    st.info("👈 Select **Upload Custom Historical & Modern Tiles** in the sidebar to analyze your own satellite imagery.")
