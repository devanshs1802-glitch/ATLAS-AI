```markdown
# ATLAS: Automated Tracking of Land, Agriculture & Sprawl

A production-grade geospatial AI monitoring pipeline built using PyTorch, Torchvision (**ResNet-18**), and accelerated locally via NVIDIA CUDA. Designed for high-accuracy land-use classification, temporal change detection, and urban sprawl risk scoring on satellite imagery.

---

##  Model Performance & Evaluation Metrics

Evaluated on the EuroSAT 10-class benchmark (27,000 labeled Sentinel-2 tiles):
- **Validation Accuracy:** `98.00%`
- **Held-Out Test Accuracy:** `97.63%`
- **Residential Class Precision:** `1.00` (Flawless classification confidence on built-up infrastructure)

---

##  Regional Case Study: Ramhan Island, Abu Dhabi
- **2016 Baseline (`ramhan_2016.png`):** Captures natural coastal sabkha and early construction earthworks (~92.74% Urban Sprawl Risk Score driven by industrial features).
- **2026 Expansion (`ramhan_2026.png`):** Captures the completed luxury waterfront canal and residential megaproject (99.99% Urban Sprawl Risk Score, 98.38% Residential confidence).

---

##  Project Structure

```text
Atlas-AI/
├── models/
│   └── resnet18_atlas.pth   # Trained model weights
├── sample_tiles/
│   ├── ramhan_2016.png      # Baseline validation tile
│   └── ramhan_2026.png      # Expansion validation tile
├── src/
│   ├── dataset.py           # Optimized data loader pipeline
│   ├── train.py             # Mixed-precision training script
│   ├── test.py              # Evaluation metrics & confusion matrix
│   └── inference.py         # Single-tile prediction & risk scoring
├── app.py                   # Streamlit geospatial command-center interface
├── requirements.txt         # Locked python package dependencies
├── .gitignore               # Git exclusion rules for large checkpoints
└── README.md                # Project documentation & metrics summary

```

---

## How to Run Locally

1. **Install Dependencies**
```bash
pip install -r requirements.txt

```


2. **Run Inference on Sample Tiles**
```bash
python src/inference.py

```


3. **Launch the Streamlit Command-Center Dashboard**
```bash
streamlit run app.py

```



---

## Authors & Team

* **Ryan Dinto:** Model Architecture, Training & Backend Execution
* **Aaryan Kalra:** Dataset Curation & Data Splitting Lead
* **Devansh Sharma:** Application Integration & Streamlit Demo Interface Lead

```

```
