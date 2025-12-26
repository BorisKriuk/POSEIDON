# 🔱 POSEIDON

**Physics-Optimized Seismic Energy Inference and Detection Operating Network**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Poseidon-orange)](https://huggingface.co/datasets/BorisKriuk/Poseidon)

A physics-informed Energy-Based Model (EBM) for multi-task seismic event prediction, trained on 2.8 million earthquakes spanning 35 years (1990-2020).

## 🌊 Overview

POSEIDON combines deep learning with established seismological principles (Gutenberg-Richter law, Omori-Utsu decay) to predict:

- **Aftershock sequences** following major earthquakes
- **Tsunami generation potential** from seismic events  
- **Foreshock identification** preceding larger events

## 📊 Performance

| Task | F1 Score | AUC-ROC | Precision | Recall |
|------|----------|---------|-----------|--------|
| **Aftershock** | 0.762 | 0.799 | 0.675 | 0.873 |
| **Tsunami** | 0.407 | 0.971 | 0.983 | 0.255 |
| **Foreshock** | 0.556 | 0.865 | 0.513 | 0.608 |
| **Average** | **0.615** | **0.878** | 0.724 | 0.579 |

## 🧠 Architecture

```
Input Features (26D)
        ↓
┌─────────────────────────────────────┐
│     Multiscale Feature Encoder      │
│  ┌─────────┬─────────┬─────────┐   │
│  │ Scale 1 │ Scale 2 │ Scale 3 │   │
│  │  64ch   │  128ch  │  256ch  │   │
│  └────┬────┴────┬────┴────┬────┘   │
│       └─────────┼─────────┘        │
│                 ↓                   │
│         Feature Fusion (448D)       │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      Physics-Constrained EBM        │
│  ┌─────────────────────────────┐   │
│  │  Gutenberg-Richter: b=0.752 │   │
│  │  Omori-Utsu: p=0.835, c=0.19│   │
│  └─────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│     Multi-Task Prediction Heads     │
│  ┌──────────┬──────────┬─────────┐ │
│  │Aftershock│ Tsunami  │Foreshock│ │
│  │  Head    │  Head    │  Head   │ │
│  └──────────┴──────────┴─────────┘ │
└─────────────────────────────────────┘
```

## 📁 Dataset

This model is trained on the **Poseidon Global Earthquake Dataset**:

Hugging Face 🤗 **[BorisKriuk/Poseidon](https://huggingface.co/datasets/BorisKriuk/Poseidon)**

| Metric | Value |
|--------|-------|
| Total Events | 2,877,769 |
| Time Span | 1990-01-01 to 2024-12-31 |
| Magnitude Range | 0.0 - 9.1 |
| Geographic Coverage | Global |
| Spatial Resolution | 1° × 1° grid |

### Dataset Features

- **Core Properties**: latitude, longitude, depth, magnitude, time
- **Energy Features**: energy_joules, log_energy (Gutenberg-Richter derived)
- **Quality Metrics**: rms, gap, nst, dmin
- **Labels**: tsunami flag, event significance

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/BorisKriuk/POSEIDON.git
cd POSEIDON
pip install -r requirements.txt
```

### Training

```python
from model import PIEBMClassifier
from data_fetcher import fetch_earthquake_data

# Load data
df = fetch_earthquake_data(start_year=1990, end_year=2024)

# Initialize model
model = PIEBMClassifier(
    input_dim=26,
    hidden_dims=[512, 256, 128],
    num_tasks=3
)

# Train with two-stage approach
# Stage 1: Physics parameter learning
# Stage 2: Full model fine-tuning
```

### Inference

```python
import torch

# Load trained model
model = PIEBMClassifier.load_from_checkpoint('poseidon_weights.pt')
model.eval()

# Predict on new earthquake data
with torch.no_grad():
    predictions = model(earthquake_features)
    aftershock_prob = predictions['aftershock']
    tsunami_prob = predictions['tsunami']
    foreshock_prob = predictions['foreshock']
```

## 🔬 Physics Integration

POSEIDON embeds fundamental seismological laws as learnable constraints:

### Gutenberg-Richter Law
```
log₁₀(N) = a - bM
```
- Learned b-value: **0.752** (literature: ~1.0)

### Omori-Utsu Decay Law
```
n(t) = K / (t + c)^p
```
- Learned p-value: **0.835** (literature: 0.7-1.5)
- Learned c-value: **0.1948** days

### Energy-Magnitude Relation
```
log₁₀(E) = 1.5M + 4.8
```

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 → 1e-5 |
| Batch Size | 512 |
| Epochs | 50 (Stage 1) + 100 (Stage 2) |
| Physics Loss Weight | 0.1 |
| Focal Loss γ | 2.0 |

## 📂 Repository Structure

```
POSEIDON/
├── model.py              # PI-EBM architecture
├── data_fetcher.py       # USGS data pipeline
├── requirements.txt      # Dependencies
├── LICENSE               # MIT License
└── README.md
```

## 🎯 Use Cases

- **Early Warning Systems**: Real-time aftershock probability assessment
- **Tsunami Alerts**: High-precision tsunami generation prediction (AUC: 0.971)
- **Seismic Hazard Analysis**: Regional risk assessment
- **Research**: Studying earthquake triggering mechanisms

## 📜 Citation

```bibtex
@software{poseidon2025,
  author = {Kriuk, Boris},
  title = {POSEIDON: Physics-Optimized Seismic Energy Inference and Detection Operating Network},
  year = {2025},
  url = {https://github.com/BorisKriuk/POSEIDON}
}
```

