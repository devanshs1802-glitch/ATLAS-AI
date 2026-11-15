import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

# Optional: Interactive swipe comparison (pip install streamlit-image-comparison)
try:
    from streamlit_image_comparison import image_comparison
    HAS_IMG_COMP = True
except ImportError:
    HAS_IMG_COMP = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ATLAS | Geospatial Intelligence & Sprawl Risk",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for an Executive Dark Geospatial Command Center Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* Main background & typography */
    .stApp {
        background-color: #050811;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(14, 165, 233, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(99, 102, 241, 0.05) 0%, transparent 40%);
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #080C17;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        background-color: #0F172A;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #1E293B;
    }

    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 4px solid #38BDF8;
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Pulsing Live Indicator */
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }
    .status-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #34D399;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.06em;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #34D399;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    /* Intelligence Briefing Box */
    .intel-brief {
        background: linear-gradient(135deg, #0F172A 0%, #111C33 100%);
        border: 1px solid #1E293B;
        border-top: 2px solid #38BDF8;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 22px;
        font-size: 0.9rem;
        color: #CBD5E1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .intel-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #38BDF8;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
        font-weight: 700;
    }

    /* Custom KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(180deg, #0F172A 0%, #0A101F 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
        height: 100%;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.08);
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.2;
    }
    .badge-high, .badge-mod, .badge-low, .badge-info {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-mod  { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-low  { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-info { background-color: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); }

    /* Viewport Frame */
    .viewport-card {
        background-color: #0A101F;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .panel-header {
        font-size: 1rem;
        font-weight: 700;
        color: #F1F5F9;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 10px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .panel-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #64748B;
        background: #111C33;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #1E293B;
    }

    /* Macro Index Bars */
    .index-row {
        margin-top: 14px;
        font-size: 0.8rem;
    }
    .index-label-row {
        display: flex;
        justify-content: space-between;
        color: #CBD5E1;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
    }
    .progress-track {
        width: 100%;
        height: 6px;
        background-color: #1E293B;
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 8px;
    }
    .progress-fill-urban { height: 100%; background: linear-gradient(90deg, #F59E0B, #EF4444); }
    .progress-fill-water { height: 100%; background: linear-gradient(90deg, #0284C7, #38BDF8); }
    .progress-fill-veg   { height: 100%; background: linear-gradient(90deg, #059669, #34D399); }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0A101F;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }

    /* Hide Streamlit Branding */
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

def render_macro_bars(sprawl, water, veg):
    return f"""
    <div class="index-row">
        <div class="index-label-row"><span>BUILT-UP / URBAN</span><span>{sprawl:.2f}%</span></div>
        <div class="progress-track"><div class="progress-fill-urban" style="width: {min(sprawl, 100)}%;"></div></div>
        <div class="index-label-row"><span>HYDROLOGY / WATER</span><span>{water:.2f}%</span></div>
        <div class="progress-track"><div class="progress-fill-water" style="width: {min(water, 100)}%;"></div></div>
        <div class="index-label-row"><span>VEGETATION / AGRI</span><span>{veg:.2f}%</span></div>
        <div class="progress-track"><div class="progress-fill-veg" style="width: {min(veg, 100)}%;"></div></div>
    </div>
    """

# -----------------------------------------------------------------------------
# 3. HERO HEADER & SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
model, device = load_model()

st.markdown("""
<div class="hero-box">
    <div>
        <p class="hero-title">🛰️ ATLAS // Geospatial Intelligence & Sprawl Risk</p>
        <p class="hero-subtitle">Automated Tracking of Land, Agriculture & Sprawl • ResNet-18 Sentinel-2 L2A Engine</p>
    </div>
    <div class="status-pill"><div class="status-dot"></div>TELEMETRY ONLINE</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model weights missing at `models/resnet18_atlas.pth`. Please verify the weights path.")
    st.stop()

with st.sidebar:
    st.markdown("### 🎛️ Mission Control")
    st.markdown(
        f"<div style='font-family: JetBrains Mono, monospace; font-size: 0.76rem; color: #94A3B8; margin-bottom: 12px;'>"
        f"BACKEND: <span style='color:#38BDF8;'>{str(device).upper()}</span> &nbsp;|&nbsp; ACC: <span style='color:#34D399;'>97.63%</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    source_mode = st.radio(
        "Select Telemetry Source",
        ["UAE Case Study: Ramhan Island", "Upload Custom Historical & Modern Tiles"]
    )
    
    st.markdown("---")
    
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
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-left:3px solid #38BDF8; padding:12px; border-radius:8px; font-size:0.84rem; color:#CBD5E1;">
            <strong style="color:#F8FAFC;">📍 Active AOI: Ramhan Island</strong><br>
            <span style="font-family:'JetBrains Mono', monospace; font-size:0.74rem; color:#38BDF8;">24.51° N, 54.63° E • Abu Dhabi</span><br><br>
            Comparing natural coastal sabkha & early earthworks (2016) against completed waterfront megaproject (2026).
        </div>
        """, unsafe_allow_html=True)
        path_2016 = "sample_tiles/ramhan_2016.png"
        path_2026 = "sample_tiles/ramhan_2026.png"
        label_before = "2016 Sabkha Baseline"
        label_after = "2026 Urban Megaproject"
        if os.path.exists(path_2016) and os.path.exists(path_2026):
            img_before = Image.open(path_2016).convert("RGB")
            img_after = Image.open(path_2026).convert("RGB")
        else:
            st.error("Sample tiles not found in `sample_tiles/`.")

    if HAS_IMG_COMP and img_before and img_after:
        st.markdown("---")
        view_mode = st.radio("Viewport Display Mode", ["Side-by-Side Split", "Interactive Swipe Slider"])
    else:
        view_mode = "Side-by-Side Split"

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
        sprawl_badge = f'<span class="badge-high">CRITICAL SPRAWL (Δ {sprawl_delta:+.1f}%)</span>'
    elif sprawl_a >= 40:
        sprawl_badge = f'<span class="badge-mod">MODERATE EXPANSION (Δ {sprawl_delta:+.1f}%)</span>'
    else:
        sprawl_badge = f'<span class="badge-low">LOW URBAN FOOTPRINT (Δ {sprawl_delta:+.1f}%)</span>'

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

    # --- AI EXECUTIVE BRIEFING BANNER ---
    primary_shift = preds_a[0][0]
    briefing_text = (
        f"Neural inference engine evaluated baseline <b>{label_before}</b> against modern target <b>{label_after}</b>. "
        f"Detected a net urban expansion of <b>{sprawl_delta:+.2f}%</b>, shifting dominant classification to <b>{primary_shift}</b> "
        f"({preds_a[0][1]:.1f}% confidence). Hydrological variance stands at <b>{water_delta:+.2f}%</b>."
    )
    st.markdown(f"""
    <div class="intel-brief">
        <div class="intel-title">⚡ ATLAS AI Spatial Intelligence Briefing</div>
        <div>{briefing_text}</div>
    </div>
    """, unsafe_allow_html=True)

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
        water_delta_badge = "badge-mod" if water_delta < -2.0 else "badge-info"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Surface Water Coverage</div>
            <div class="kpi-value">{water_a:.2f}%</div>
            <span class="{water_delta_badge}">Δ {water_delta:+.2f}% vs Baseline</span>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Dominant Modern Class</div>
            <div class="kpi-value" style="font-size: 1.45rem;">{preds_a[0][0]}</div>
            <span class="badge-low">{preds_a[0][1]:.2f}% Confidence</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 2: VISUAL SATELLITE COMPARISON ---
    if view_mode == "Interactive Swipe Slider" and HAS_IMG_COMP:
        st.markdown(f'<div class="panel-header">🛰️ Interactive Temporal Swipe <span class="panel-tag">{label_before} ⇄ {label_after}</span></div>', unsafe_allow_html=True)
        image_comparison(
            img1=img_before,
            img2=img_after,
            label1=label_before,
            label2=label_after,
            width=1100,
            starting_position=50,
            show_labels=True,
            make_responsive=True
        )
        c_b, c_a = st.columns(2, gap="large")
        with c_b:
            st.markdown(f"**{label_before} Macro Indices**" + render_macro_bars(sprawl_b, water_b, veg_b), unsafe_allow_html=True)
        with c_a:
            st.markdown(f"**{label_after} Macro Indices**" + render_macro_bars(sprawl_a, water_a, veg_a), unsafe_allow_html=True)
    else:
        col_left, col_right = st.columns(2, gap="large")
        
        with col_left:
            st.markdown(f"""
            <div class="viewport-card">
                <div class="panel-header">
                    <span>🕰️ {label_before}</span>
                    <span class="panel-tag">T0 BASELINE</span>
                </div>
            """, unsafe_allow_html=True)
            st.image(img_before, use_container_width=True)
            st.markdown(render_macro_bars(sprawl_b, water_b, veg_b) + "</div>", unsafe_allow_html=True)
            
        with col_right:
            st.markdown(f"""
            <div class="viewport-card">
                <div class="panel-header">
                    <span>🛰️ {label_after}</span>
                    <span class="panel-tag">T1 TARGET</span>
                </div>
            """, unsafe_allow_html=True)
            st.image(img_after, use_container_width=True)
            st.markdown(render_macro_bars(sprawl_a, water_a, veg_a) + "</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 3: DETAILED LAND-USE SHIFT ANALYTICS ---
    tab1, tab2 = st.tabs(["📊 Land-Use Class Shift Chart", "📋 Full Telemetry Matrix"])
    
    with tab1:
        st.markdown("#### Temporal Probability Distribution Shift (%)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=CLASSES,
            y=[dict_b[c] for c in CLASSES],
            name=label_before,
            marker_color="#64748B",
            hovertemplate="%{x}<br>" + label_before + ": <b>%{y:.2f}%</b><extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=CLASSES,
            y=[dict_a[c] for c in CLASSES],
            name=label_after,
            marker_color="#38BDF8",
            hovertemplate="%{x}<br>" + label_after + ": <b>%{y:.2f}%</b><extra></extra>"
        ))
        fig.update_layout(
            barmode="group",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 16, 31, 0.4)",
            font=dict(color="#CBD5E1", family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, tickangle=-15),
            yaxis=dict(title="Class Probability (%)", gridcolor="#1E293B", zerolinecolor="#334155")
        )
        st.plotly_chart(fig, use_container_width=True)

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
                "Net Shift (%)": round(diff, 2)
            })
        df_table = pd.DataFrame(table_data)

        def highlight_shift(val):
            if val > 1.0:
                return "color: #38BDF8; font-weight: 700;"
            elif val < -1.0:
                return "color: #F87171; font-weight: 700;"
            return "color: #94A3B8;"

        styled_df = df_table.style.map(highlight_shift, subset=["Net Shift (%)"])
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                f"{label_before} (%)": st.column_config.ProgressColumn(
                    f"{label_before} (%)", format="%.2f%%", min_value=0, max_value=100
                ),
                f"{label_after} (%)": st.column_config.ProgressColumn(
                    f"{label_after} (%)", format="%.2f%%", min_value=0, max_value=100
                ),
                "Net Shift (%)": st.column_config.NumberColumn("Net Shift (%)", format="%+.2f%%")
            }
        )

elif img_before is not None or img_after is not None:
    st.warning("⏳ Waiting for second tile: Please upload **both** a Before and After satellite image in the sidebar to run temporal comparison.")
else:
    st.info("👈 Select **Upload Custom Historical & Modern Tiles** in the sidebar to analyze your own satellite imagery.")
