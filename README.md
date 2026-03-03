 # HepatoAIM: A Multimodal Deep Learning System for Hepatocellular Carcinoma Drug-Target Interaction Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TorchGeometric](https://img.shields.io/badge/TorchGeometric-2.3+-3b82f6.svg)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-green.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Publication**: Drug Repurposing for Hepatocellular Carcinoma via Multimodal Neural Networks and Virtual Screening

> Computational experiments were performed on a laptop workstation equipped with an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM, driver version 566.07,CUDA 12.7), a 12th Gen Intel Core i7-12700H processor (14 cores: 6 P-cores + 8 E-cores), and 16 GB DDR4 system memory, running Windows 11.


---

## 📑 Table of Contents

- [HepatoAIM: A Multimodal Deep Learning System for Hepatocellular Carcinoma Drug-Target Interaction Prediction](#hepatoaim-a-multimodal-deep-learning-system-for-hepatocellular-carcinoma-drug-target-interaction-prediction)
  - [📑 Table of Contents](#-table-of-contents)
  - [Key Features](#key-features)
    - [1. Dual-Modal Fusion Architecture](#1-dual-modal-fusion-architecture)
    - [2. Self-Supervised Contrastive Learning](#2-self-supervised-contrastive-learning)
    - [3. End-to-End Drug Screening Pipeline](#3-end-to-end-drug-screening-pipeline)
  - [System Architecture](#system-architecture)
  - [Core Modules](#core-modules)
    - [1. Data Preprocessing (`pre_data.py` / `pre_main.ipynb`)](#1-data-preprocessing-pre_datapy--pre_mainipynb)
    - [2. Graph Neural Network Construction (`def_GNN.py`)](#2-graph-neural-network-construction-def_gnnpy)
    - [3. Classification Model (`def_c_323.py` / `main.ipynb`/`def_cls317-1.ipynb`)](#3-classification-model-def_c_323py--mainipynbdef_cls317-1ipynb)
    - [4. Evaluation \& Visualization (`Grade_c` Class)](#4-evaluation--visualization-grade_c-class)
    - [5. Virtual Screening (`screening_c.py` / `screening_r.py`)](#5-virtual-screening-screening_cpy--screening_rpy)
  - [Quick Start](#quick-start)
    - [Environment Setup](#environment-setup)
      - [Method 1: Using environment.yml (Recommended)](#method-1-using-environmentyml-recommended)
      - [Method 2: Using requirements.txt](#method-2-using-requirementstxt)
  - [Usage Guide](#usage-guide)
    - [Method 1: Model Training (def\_cls317-1.ipynb)](#method-1-model-training-def_cls317-1ipynb)
    - [Method 2: Data Preprocessing (pre\_main.ipynb)](#method-2-data-preprocessing-pre_mainipynb)
    - [Method 3: Modular API (Compose in main.ipynb)](#method-3-modular-api-compose-in-mainipynb)
  - [Project Structure](#project-structure)

---

## Key Features

### 1. Dual-Modal Fusion Architecture
- **Molecular Fingerprint Branch**: 12 molecular fingerprints (extracted via PaDELpy) → Transformer Encoder
- **Molecular Graph Branch**: Molecular graph constructed via RDKit → TransformerConv + GAT → Global Average Pooling
- **Fusion Strategy**: Multi-head cross-attention mechanism + learnable weighted concatenation (192-dimensional fused features)

### 2. Self-Supervised Contrastive Learning
- **Dual-View Augmentation**: Random edge dropping (15%) + node feature masking (10%, mean imputation)
- **Dynamic Loss Weighting**: λ = max(0.3×(1-epoch/300), 0.05)
- **NT-Xent Loss**: Temperature coefficient τ=0.6, cosine similarity metric

### 3. End-to-End Drug Screening Pipeline
- **Classification Model**: Predicts drug-target interaction probability (binary classification)
- **Regression Model**: Predicts binding affinity values (IC50, Ki, Kd, etc.)
- **Virtual Screening**: Large-scale FDA drug library screening + molecular docking validation

---

## System Architecture

```
HepatoAIM/
├── Data Preprocessing Layer (pre_data.py / pre_main.ipynb)
│   ├── ChEMBL data cleaning and standardization
│   ├── 12 molecular fingerprint extraction (PaDELpy)
│   ├── Low-variance feature filtering (VarianceThreshold=0.5)
│   └── Data merging and format conversion
│
├── Graph Construction Layer (def_GNN.py)
│   ├── SMILES → Molecular graph conversion (RDKit)
│   ├── 3D conformer generation (MMFF/UFF optimization)
│   ├── Node features: 15-dimensional atomic properties
│   │   ├── Atomic number, degree, charge, hybridization, aromaticity
│   │   ├── Electronegativity, ring information, chirality, hydrogen count
│   │   └── Valence electrons, bond length
│   └── Edge features: 7-dimensional bond properties
│       ├── Bond type, conjugation, ring membership
│       ├── Stereochemistry, aromaticity
│       └── Charge difference, bond length
│
├── Model Training Layer (main.ipynb / def_c_323.py)
│   ├── Classification Model (def_c_323.py / def_c_322.py / def_c_313.py)
│   │   ├── Dual-modal encoder
│   │   ├── Contrastive learning projection head
│   │   └── Early stopping (patience=150)
│   └── Regression Model (train_c.py / def_for_re.py)
│       └── Bioactivity value prediction
│
├── Screening Application Layer (screening_c.py / screening_r.py)
│   ├── Classification Screening (screening_c.py)
│   │   └── Predicts drug-target interaction probability
│   └── Regression Screening (screening_r.py)
│       └── Predicts binding affinity values
│
└── Utility Layer
    ├── Protein feature extraction (protein_bert_pre.py)
    ├── SMILES processing tools (add_smiles.py / add_smiles_s.py)
    └── Model evaluation and visualization
```

---

## Core Modules

### 1. Data Preprocessing (`pre_data.py` / `pre_main.ipynb`)

**Class**: `pre_date`

| Method | Function |
|:---|:---|
| `log_in_data(target)` | Imports ChEMBL data, performs cleaning and standardization |
| `pre_figure_class(target)` | Classification data preprocessing (IC50/AC50/Kd50, etc.) |
| `pre_figure_regression(target)` | Regression data preprocessing (grouped by Standard Type) |
| `fged()` | Core fingerprint extraction function (PaDELpy) |
| `figured(CRS='C')` | Executes fingerprint extraction (C=Classification, R=Regression) |
| `figured_S(screening_name)` | Fingerprint extraction for screening datasets |

**Data Pipeline**:
```
Raw ChEMBL Data
    ↓
Data Cleaning (remove nulls, unit conversion, filter Type B assays)
    ↓
Activity Classification (median split: active/inactive) or Regression Labeling
    ↓
12 Fingerprint Extraction (AtomPairs2D, MACCS, PubChem, etc.)
    ↓
Low-Variance Filtering (threshold=0.5)
    ↓
Standardized CSV Output
```

### 2. Graph Neural Network Construction (`def_GNN.py`)

**Function**: `smiles_to_graph(df)`

**Node Features** (15-dimensional):
```python
[
    atom.GetAtomicNum(),           # Atomic number
    atom.GetDegree(),              # Bond degree
    atom.GetFormalCharge(),        # Formal charge
    atom.GetHybridization(),       # Hybridization type
    atom.GetIsAromatic(),          # Aromaticity
    electronegativity,             # Electronegativity
    atom.IsInRing(),               # Ring membership
    atom.GetTotalDegree(),         # Total degree
    atom.GetChiralTag().real,      # Chirality
    atom.GetNumImplicitHs(),       # Implicit hydrogen count
    atom.GetNumExplicitHs(),       # Explicit hydrogen count
    atom.IsInRingSize(3),          # 3-membered ring
    atom.IsInRingSize(4),          # 4-membered ring
    atom.GetExplicitValence(),     # Explicit valence
    atom.GetImplicitValence()      # Implicit valence
]
```

**Edge Features** (7-dimensional):
```python
[
    bond.GetBondTypeAsDouble(),    # Bond type
    bond.GetIsConjugated(),        # Conjugation
    bond.IsInRing(),               # Ring membership
    bond.GetStereo(),              # Stereochemistry
    bond.GetIsAromatic(),          # Aromaticity
    charge_diff,                   # Electronegativity difference
    bond_length                    # Bond length (calculated from 3D coordinates)
]
```

**3D Conformer Generation Strategy**:
1. Attempt MMFF optimization
2. Fallback to UFF optimization if MMFF fails
3. Generate 2D coordinates if both fail

### 3. Classification Model (`def_c_323.py` / `main.ipynb`/`def_cls317-1.ipynb`)

**Class**: `Class_Bert_NN`

```python
Class_Bert_NN(
    tg,                    # Target ID
    descriptor_size,       # Fingerprint dimension
    test_num,              # Complexity level (0-8)
    num_classes=2          # Binary classification
)
```

**Architecture Details**:

| Component | Configuration |
|:---|:---|
| **Fingerprint Encoder** | Linear(descriptor_size→64) + LayerNorm + GELU |
| | + (test_num+1)×TransformerEncoderLayer(d_model=64, nhead=2) |
| **Graph Encoder** | TransformerConv(15→64, heads=2) + LayerNorm |
| | + Learnable residual connection (res_weight) |
| | + GATConv(128→64) + Global Average Pooling |
| **Fusion Layer** | MultiheadAttention(embed_dim=64, num_heads=4) |
| | + Learnable weighted concatenation (fused_feat_weight, desc_feat_weight, graph_feat_weight) |
| **Classification Head** | Linear(192→512) + GELU + LayerNorm + Dropout(0.01) |
| | + Linear(512→num_classes) + Sigmoid |
| **Projection Head** | Linear(192→512) + GELU + LayerNorm + Dropout(0.3) |
| | + Linear(512→256) + GELU + LayerNorm |
| | + Linear(256→64) |

**Contrastive Learning Loss** (`ContrastiveLoss`):
```python
# NT-Xent loss implementation
sim_matrix = cosine_sim(z.unsqueeze(1), z.unsqueeze(0)) / temperature
labels = torch.cat([torch.arange(batch_size)+batch_size, torch.arange(batch_size)])
loss = F.cross_entropy(sim_matrix.masked_fill(eye_mask, -inf), labels)
```

**Training Configuration**:
- Optimizer: AdamW (lr=1e-4, weight_decay=0.1)
- Learning Rate Scheduler: OneCycleLR
- Gradient Clipping: max_norm=1.0
- Early Stopping: patience=150
- Batch Size: 64
- Train/Test Split: 0.5/0.5 (low-sample targets) or 0.8/0.2

**Data Augmentation** (`Train_C_Data`):
```python
# Online graph augmentation, generating dual views
def apply_graph_augmentation(graph_data):
    edge_index = drop_edge(graph_data.edge_index, p=0.15)  # Random edge dropping
    masked_x, _ = mask_node(graph_data.x, p=0.1)            # Random node masking
    return Data(x=masked_x, edge_index=edge_index, edge_attr=graph_data.edge_attr)
```

### 4. Evaluation & Visualization (`Grade_c` Class)

**Evaluation Metrics**:
- Accuracy, Precision, Recall
- F2-Score (β=2, emphasizing recall)
- AUC, MCC

**Optimal Model Selection**:
```python
# Normalized weighted average: 0.4×F2 + 0.2×Accuracy + 0.4×AUC
weights = [0.4, 0.2, 0.4]
best_index = argmax(weighted_average)
```

**Visualization Output** (9-panel SVG):
1. Loss curves (train/validation)
2. Accuracy curves
3. Precision curves
4. Recall curves
5. AUC curves
6. MCC curves
7. Contrastive feature cosine similarity heatmap (hierarchical clustering)
8. Training set confusion matrix
9. Test set confusion matrix

### 5. Virtual Screening (`screening_c.py` / `screening_r.py`)
*Note: Preprocessing of screening datasets is required before screening*

**Classification Screening Pipeline**:
```python
def screen_c_tg_list(screening_data_pth, target_list):
    for tg in target_list:
        # Load pretrained model
        model.load_state_dict(torch.load(f'./best_model/{tg}_classify_best_model.pth'))
        # Predict interaction probability
        out = screen_for_c(tg, screening_pth)  # Output: [batch_size, 2] probability distribution
        # Merge results
        df_out[tg] = out[:, 1]  # Extract positive class probability
    return df_out
```

**Regression Screening Pipeline**:
```python
def screen_for_r_put_out(out, tg_value_type):
    for tg, types in tg_value_type.items():
        # Filter high-confidence samples (probability>0.95)
        high_score_df = out[out[tg] > 0.95]
        for type00 in types:  # IC50, Ki, Kd, etc.
            predictions = screen_for_r(tg, selected_rows, type00)
        # Save predictions for each target and each type
    return result_r_{tg}.csv
```

---

## Quick Start

> **⚠️ Important**: ChEMBL data preprocessing is complete; you can directly use `main.ipynb` for training. To reprocess data or handle new targets, use `pre_main.ipynb`

### Environment Setup

#### Method 1: Using environment.yml (Recommended)

```bash
# Create new environment named hepatoaim
conda env create -f environment.yml -n hepatoaim

# Activate environment
conda activate hepatoaim
```

#### Method 2: Using requirements.txt


```bash
# 1. Create base Python environment first
conda create -n hepatoaim python=3.9

# 2. Activate environment
conda activate hepatoaim

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note**: This method may miss some dependencies; `environment.yml` is recommended


---

## Usage Guide

### Method 1: Model Training (def_cls317-1.ipynb)

**Use Case**: Quick start for model training, reproducing paper results

**Prerequisites**: `./fingered_c_data/` directory contains preprocessed data (CHEMBL1811_cs_fg.csv, etc.)

**Steps**:
1. Open `main.ipynb`
2. Execute cells in order:
   - Import dependencies (`def_c_323` and other modules)
   - Configure training parameters (target, epochs, learning rate, etc.)
   - Launch training (automatically executes `train_cls`)
   - View evaluation results (auto-generates 9-panel charts)
3. Check outputs:
   - Best model weights: `./model/{tg}_classify_best_model_{t}.pth`
   - Performance metrics: `./model/{tg}_classify_best_model_performance_{test_num}_{t}.txt`
   - Training visualization: `./train_c_putouts/{tg}_combined1_plots_{test_num}_{t}.svg`

**Quick Training Example**:
```python
from def_c_323 import train_cls
import torch

# Train GCGR target
results, model = train_cls(
    size=0.5,                    # Training set ratio
    tg='CHEMBL1985',             # Target ID
    train_epochs=800,            # Training epochs
    device=torch.device('cuda:0'),
    test_num=0,                  # Complexity level (0-8)
    batch_size=64,
    loss_fn=torch.nn.CrossEntropyLoss(),
    num_classes=2,
    l_r=1e-4,                    # Learning rate
    t=1                          # Experiment repetition ID
)
```

**Advantages**:
- Ready to use out of the box, no preprocessing needed
- Built-in optimal hyperparameters
- Automatic early stopping and model selection

`Typical training time was approximately [18] min per target for 800 epochs.`

---

### Method 2: Data Preprocessing (pre_main.ipynb)

**Use Cases**:
- Adding new targets (non-CHEMBL1811/1974/1985/4896)
- Re-extracting fingerprint features
- Processing custom screening databases

**Steps**:
1. Open `pre_main.ipynb`
2. Configure target list and paths:
```python
from pre_data import pre_date

# Configure new targets
tg_list = ['CHEMBL1234', 'CHEMBL5678']  # New targets
fg_list = ['AtomPairs2DCount', 'AtomPairs2D', 'EState', 'CDKextended', 
           'CDK', 'CDKgraphonly', 'KlekotaRothCount', 'KlekotaRoth',
           'MACCS', 'PubChem', 'SubstructureCount', 'Substructure']

# Initialize preprocessor
preprocessor = pre_date(
    tg_list, 
    "./targets/*.csv",           # Raw ChEMBL data path
    fg_list, 
    "./fingerprints_xml/*.xml"   # Fingerprint configuration files
)
```
3. Execute preprocessing:
```python
# Classification data preprocessing
preprocessor.figured(CRS='C')

# Regression data preprocessing  
preprocessor.figured(CRS='R')

# Screening data preprocessing
preprocessor.figured_S(screening_name='FDA_Drug_Library')
```

**Output Results**:
- Classification data: `./fingered_c_data/{tg}_cs_fg.csv`
- Regression data: `./fingered_r_data/{tg}_r_fg.csv`
- Screening data: `./fingered_s_data/{tg}_s_c_fg.csv`, etc.

**Advantages**:
- Visualized preprocessing workflow
- Supports batch processing of multiple targets
- Customizable fingerprint types and filtering thresholds

---

### Method 3: Modular API (Compose in main.ipynb)

**Use Case**: Custom experimental workflows, batch processing, or modifying specific modules

**Example: Complete Workflow**
```python
# === Step 1: Preprocessing (if needed) ===
from pre_data import pre_date
preprocessor = pre_date(['CHEMBL1234'], "./targets/*.csv", fg_list, "./fingerprints_xml/*.xml")
preprocessor.figured(CRS='C')

# === Step 2: Training ===
from def_c_323 import train_cls
results, model = train_cls(
    size=0.8, tg='CHEMBL1234', train_epochs=500,
    device=torch.device('cuda:0'), test_num=2,
    batch_size=32, loss_fn=torch.nn.CrossEntropyLoss(),
    num_classes=2, l_r=5e-5, t=1
)

# === Step 3: Screening ===
# Note: Use same structure as training data
from screening_c import screen_c_tg_list
df_results = screen_c_tg_list('./screening/new_library.csv', ['CHEMBL1234'])

# === Step 4: Prediction (high-confidence samples) ===
from screening_c import screen_for_r_put_out
tg_value_type = {'CHEMBL1234': ['IC50', 'Ki']}
screen_for_c_put_out(df_results, tg_value_type)
```

---

## Project Structure

```
HepatoAIM/
├── README.md
├── requirements.txt
│
├── 📓 Jupyter Notebooks
│   ├── main.ipynb                   ⭐ Model Training (Recommended, preprocessing complete)
│   └── pre_main.ipynb               🔧 Data Preprocessing (Add new targets or reprocess)
│
├── data/
│   ├── targets/                    # Raw ChEMBL data
│   │   ├── CHEMBL1811.csv
│   │   ├── CHEMBL1974.csv
│   │   ├── CHEMBL1985.csv
│   │   └── CHEMBL4896.csv
│   ├── fingerprints_xml/           # PaDELpy fingerprint configuration files
│   ├── screening/                  # Screening databases
│   └── saved_bert_data/            # Pre-computed protein features
│
├── processed_data/
│   ├── fingered_c_data/            # ⭐ Classification fingerprint data (generated, ready to use)
│   │   ├── CHEMBL1811_cs_fg.csv
│   │   ├── CHEMBL1974_cs_fg.csv
│   │   ├── CHEMBL1985_cs_fg.csv
│   │   └── CHEMBL4896_cs_fg.csv
│   ├── fingered_r_data/            # Regression fingerprint data
│   ├── fingered_r_data_dd/         # Regression data split by type
│   └── fingered_s_data/            # Screening data fingerprints
│
├── src/
│   ├── pre_data.py                 # Data preprocessing main program
│   ├── def_GNN.py                  # Graph neural network construction
│   ├── def_c_323.py                # Classification model (latest, recommended)
│   ├── def_c_322.py                # Classification model (stable version)
│   ├── def_c_313.py                # Classification model (experimental version)
│   ├── train_c.py                  # Regression model training
│   ├── def_for_re.py               # Regression model definition
│   ├── def_for_cls.py              # Classification model wrapper
│   ├── screening_c.py              # Classification screening
│   ├── screening_r.py              # Regression screening
│   ├── protein_bert_pre.py         # Protein feature preloading
│   ├── add_smiles.py               # SMILES merging tool
│   └── add_smiles_s.py             # Screening data SMILES processing
│
├── models/
│   ├── best_model/                 # Best model weights
│   │   ├── CHEMBL1811_classify_best_model.pth
│   │   ├── CHEMBL1974_classify_best_model.pth
│   │   ├── CHEMBL1985_classify_best_model.pth
│   │   └── CHEMBL4896_classify_best_model.pth
│   └── checkpoints/                # Training checkpoints
│
├── results/
│   ├── train_c_putouts/            # Training visualization SVGs
│   ├── model_performance/          # Performance metric text files
│   └── screening_results/          # Screening result CSVs
│
└── scripts/
    ├── train_all_targets.py        # Batch training script
    ├── evaluate_models.py          # Model evaluation
    └── run_screening.py            # Execute virtual screening
```