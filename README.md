# Learning Spatial Organization Patterns Within Cells

A PyTorch implementation of a Vector-Quantized Variational Autoencoder (VQ-VAE) for learning protein localization patterns from microscopy images. This model discovers how proteins are spatially organized within cells by learning discrete, interpretable representations of subcellular structure.

## Table of Contents

- [Overview](#overview)
- [Biological Background](#biological-background)
- [Machine Learning Approach](#machine-learning-approach)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Extensions](#extensions)

## Overview

Protein localization—understanding where proteins are located within cells—is crucial to cell biology. A protein's position determines its function. Mislocalization of proteins is implicated in many diseases, even when the protein's structure is normal.

This project trains a deep learning model to:
1. Analyze microscopy images without explicit labels
2. Learn a **latent representation** capturing protein localization patterns
3. Discover meaningful structure in spatial organization

Unlike classification models that predict a single label, this approach learns **representation learning**: an internal map where proteins with similar localization patterns cluster together naturally, enabling downstream tasks like clustering, visualization, and biological discovery.

## Biological Background

### Cellular Organization

Cells contain membrane-bound compartments (organelles) that specialize in different functions:

- **Nucleus**: Contains DNA, the genetic blueprint
- **Mitochondria**: Powerhouses generating cellular energy (ATP)
- **Endoplasmic Reticulum (ER)**: Synthesizes proteins (rough ER) and handles lipids (smooth ER)
- **Golgi Apparatus**: Modifies, packages, and routes proteins
- **Cytoplasm**: Gel-like substance where most cellular activity occurs
- **Plasma Membrane**: Protective lipid bilayer controlling entry/exit

### Protein Localization

After synthesis, proteins don't randomly float around. They're directed to specific locations via molecular "postal codes" (signal sequences):

**Why Location Matters:**
- DNA polymerase only works in the nucleus
- Mitochondrial enzymes only function in mitochondria
- Cell membranes need membrane-anchored proteins

**Disease Implications:**
- **ALS (Amyotrophic Lateral Sclerosis)**: TDP-43 accumulates in cytoplasm instead of nucleus
- **Breast/Ovarian Cancer**: BRCA1 tumor suppressor mislocalizes to cytoplasm
- Protein is normal, but location determines disease

### Dataset: OpenCell

The model trains on the OpenCell database:
- **1,311** fluorescently tagged human proteins
- **~18** fields of view per protein
- **~45** image crops per field of view
- **~800** images per protein
- **Total**: >1,048,800 labeled microscopy images

All images captured with consistent pipeline, enabling focus on biological variation rather than technical artifacts.

## Machine Learning Approach

### Why VQ-VAE?

Unlike continuous autoencoders, VQ-VAEs force the model to describe images using discrete, learned patterns (codebook vectors). Benefits:

1. **Interpretability**: Understand which visual patterns the model learned
2. **Compression**: Efficient discrete representation
3. **Stability**: Prevents representation collapse
4. **Reusability**: Codebook entries can be analyzed across proteins

### Key Concepts

#### Autoencoders
Neural networks that compress input through a bottleneck:
```
Input → Encoder (compression) → Bottleneck → Decoder (reconstruction) → Output
```

The bottleneck forces the model to learn only essential features.

#### Variational Autoencoders (VAEs)
Instead of single fixed encoding, VAEs learn probability distributions:
- Encoder outputs mean and standard deviation
- Model samples from this distribution
- Forces smooth, structured latent space
- Enables meaningful interpolation between points

#### Vector Quantization
Core innovation: Replace continuous embeddings with nearest discrete codebook entry:
```
Continuous encoder output → Find nearest codebook vector → Use that vector
```

Benefits:
- Discrete, interpretable representations
- Better compression
- More structured latent space
- Enables analysis of learned patterns

### Loss Function

The model optimizes four complementary objectives:

```
Total Loss = Reconstruction Loss + Codebook Loss + Commitment Loss + Classification Loss
```

**1. Reconstruction Loss (MSE)**
```
L_recon = ||original_image - reconstructed_image||²
```
Ensures the model compresses only non-essential details.

**2. Codebook Loss**
```
L_codebook = ||stop_gradient[encoder_output] - codebook_vector||²
```
Updates codebook vectors to match encoder outputs. Uses stop_gradient so only codebook updates.

**3. Commitment Loss**
```
L_commit = β × ||encoder_output - stop_gradient[codebook_vector]||²
```
Keeps encoder from diverging away from codebook. Uses stop_gradient so only encoder updates. Parameter β (default 0.25) controls strength.

**4. Classification Loss** (Auxiliary Task)
```
L_class = α × CrossEntropy(logits, protein_id)
```
Predicts protein identity from learned embedding. Controls via classification_weight α.

### Why Classification Helps

The auxiliary protein classification task is crucial:
- Guides model to learn discriminative spatial features
- Prevents codebook collapse (using only few entries)
- Pushes model to encode meaningful patterns, not just minimize reconstruction
- Improves downstream tasks like clustering and protein grouping

## Architecture

### Components

#### Encoder
Converts 100×100 grayscale images to latent feature maps:

```
Input (1, 100, 100)
  ↓
Conv2d(1→32, stride=2)  + ReLU           [32, 50, 50]
  ↓
Conv2d(32→64, stride=2) + ReLU           [64, 25, 25]
  ↓
Conv2d(64→64, stride=1)                  [64, 25, 25]
  ↓
ResNetBlock(64)                          [64, 25, 25]
  ↓
ResNetBlock(64)                          [64, 25, 25]
```

**ResNet Block** (modern design with group normalization):
```
Input
  ↓
GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3
                                          ↓
                            Add skip connection from input
```

#### Vector Quantizer
Discretizes embeddings using learned codebook:

1. Flatten encoder output
2. Compute Euclidean distances to all codebook vectors
3. Find nearest vector (argmin distance)
4. Replace with codebook vector
5. Use straight-through estimator for gradients

**Straight-Through Estimator (STE)**:
```
Forward: output = quantized_vector (discrete)
Backward: gradient flows as if no quantization occurred
```
Clever trick enabling gradient-based training despite non-differentiable quantization.

#### Decoder
Reconstructs images from quantized representations:

```
Quantized [64, 25, 25]
  ↓
ResNetBlock(64)                          [64, 25, 25]
  ↓
ResNetBlock(64)                          [64, 25, 25]
  ↓
Upsample (bilinear 2×) → Conv3×3 + ReLU [32, 50, 50]
  ↓
Upsample (bilinear 2×) → Conv3×3        [1, 100, 100]
```

**Upsampling Design**: Uses bilinear interpolation + convolution (not transposed convolution) to avoid checkerboard artifacts common in deconvolutions.

#### Classification Head
Auxiliary task for protein identification:

```
Quantized [64, 25, 25]
  ↓
Flatten to [batch, 64×25×25 = 40,000]
  ↓
Dense(40,000 → 1,000) → ReLU → Dropout(0.45)
  ↓
Dense(1,000 → 1,000) → ReLU → Dropout(0.45)
  ↓
Dense(1,000 → num_proteins)
```

### Hyperparameters

- `embedding_dim`: 64 (dimension of codebook vectors)
- `num_embeddings`: 512 (number of discrete vectors)
- `commitment_cost`: 0.25 (scales commitment loss)
- `dropout_rate`: 0.45 (regularization)
- `classification_head_layers`: 2 (depth of classifier)

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy pandas tqdm matplotlib scikit-learn pillow

# Or use requirements.txt
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from training import create_model_and_train

# Train with 50 proteins
model, metrics = create_model_and_train(
    data_path="./data",
    num_proteins=50,
    num_epochs=10,
    batch_size=256,
    learning_rate=0.001,
)
```

### Training from Scratch

```python
from model import LocalizationModel
from dataset import DatasetBuilder
from training import train
import torch
import torch.optim as optim

# Build dataset
builder = DatasetBuilder("./data")
dataset_splits = builder.build(
    splits={"train": 0.8, "valid": 0.1, "test": 0.1},
    exclusive_by="fov_id",
    n_proteins=50,
)

# Create model
model = LocalizationModel(
    embedding_dim=64,
    num_embeddings=512,
    commitment_cost=0.25,
    num_classes=50,
    dropout_rate=0.45,
    classification_head_layers=2,
)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model, metrics = train(
    model=model,
    dataset_splits=dataset_splits,
    num_epochs=10,
    batch_size=256,
    learning_rate=0.001,
    classification_weight=1.0,
    device=device,
)
```

### Analyze Reconstructions

```python
from visualization import visualize_reconstructions

fig = visualize_reconstructions(
    model=trained_model,
    dataset=dataset_splits['valid'],
    num_samples=8,
)
fig.savefig('reconstructions.png')
```

### Monitor Training

```python
from visualization import plot_training_metrics

fig = plot_training_metrics(
    metrics=metrics,
    save_path='training_metrics.png',
)
```

### Analyze Codebook Usage

```python
from visualization import plot_codebook_usage

fig = plot_codebook_usage(
    model=trained_model,
    dataset=dataset_splits['valid'],
)
fig.savefig('codebook_usage.png')
```

### Visualize Latent Space

```python
from visualization import visualize_latent_space

fig = visualize_latent_space(
    model=trained_model,
    dataset=dataset_splits['valid'],
    num_samples=500,
)
fig.savefig('latent_space.png')
```

## Results

### Small Model (50 Proteins)

#### Training Dynamics
- **Reconstruction Loss**: Decreases from ~0.15 → ~0.05
- **Classification Loss**: Dominant initially, decreases to ~0.02
- **Codebook Loss**: Stabilizes around 0.01
- **Commitment Loss**: Stabilizes around 0.001
- **Validation Gap**: Minimal, indicating good generalization

#### Codebook Utilization
- **With Classification Head**: Perplexity rises to ~350
  - Uses broad vocabulary of codebook entries
  - Diverse, discriminative representations

- **Without Classification Head**: Perplexity collapses to ~30
  - Uses only ~30 of 512 codebook entries
  - Focused solely on reconstruction
  - Overfits to pixel details

#### Reconstruction Quality
- **With Classification Head**: Blurry but meaningful (captures structure)
- **Without Classification Head**: Sharper but less meaningful (memorizes details)
- Trade-off: Better representations → Slightly blurrier reconstructions

### Embedding Structure (UMAP Analysis)

Visualizing learned embeddings using UMAP reveals:

**With Classification Head** :
- Chromatin, mitochondria, nucleolus form tight, distinct clusters
- Clear separation between organelles
- Model learned meaningful biological features
- Shared subcellular structures cluster together

**Without Classification Head** :
- Less distinct clusters
- ER, vesicles, cytoplasm more intermixed
- Weaker compartment discrimination
- Structure exists but less organized

**Conclusion**: Auxiliary classification task significantly improves learned representation structure.

### Feature Spectra Analysis

For each protein, we compute a histogram of codebook entry usage (feature spectrum—a "fingerprint" of localization patterns).

#### Codebook Specialization
Hierarchical clustering of proteins reveals:
- Codebook entries organize into 8 functional groups
- Groups correlate with biological function
- Proteins with similar localization use overlapping codebook subsets
- **Unsupervised learning**: No localization labels used!

#### Localization Signatures (Figure 6-16)
Aggregating feature spectra by known localization:

| Compartment | Pattern |
|------------|---------|
| **Nucleolus** | Narrow activation (group VIII only) - structured organelle |
| **Chromatin** | Similar to nucleoplasm - shared features |
| **Mitochondria** | Distributed across multiple groups, high group VI - variable features |
| **ER** | Broader activation - more heterogeneous structure |
| **Cytoplasm** | Broadest activation - most variable, diffuse localization |

Key insight: **Model learned biological organization without supervision**, reflecting true subcellular structure heterogeneity.

### Scaling Results (500 Proteins)

Expanding from 50 to 500 proteins:
- **Clearer localization signatures** across broader protein range
- **More codebook groups** (12 vs 8) capturing finer distinctions
- **Better clustering quality** of biological compartments
- **Improved generalization** to larger, more diverse protein set

Evidence that scaling improves representation quality and biological meaningfulness.

## Key Findings

### 1. Auxiliary Tasks Drive Learning
- Classification head critical for learning meaningful representations
- Without it: codebook collapses, weaker embeddings
- With it: rich, structured, interpretable latent space
- Blurrier reconstructions reflect learning meaningful abstractions, not just pixel details

### 2. Unsupervised Learning Works
- Model discovers protein organization **without localization labels**
- Learned codebook entries correspond to biological compartments
- Embeddings cluster by compartment type naturally
- Framework transfers to unlabeled or weakly labeled data

### 3. Codebook as Biological Vocabulary
- Discrete codebook acts as learned visual vocabulary
- Different codebook entries specialize for different organelles
- Feature spectra reveal protein's biological role
- Analysis enables biological discovery (e.g., protein function assignment)

### 4. Trade-offs in Model Design
| Objective | Reconstruction | Representation | Codebook Use |
|-----------|---|---|---|
| Reconstruction-focused | Sharp | Weak | Collapsed |
| Classification-focused | Blurry | Strong | Diverse |
| Balanced | Moderate | Strong | Healthy |

### 5. Scaling Improves Quality
- More proteins → better learned features
- Larger dataset → broader coverage of biological variation
- Richer codebook organization emerges
- Generalization improves

## Applications

### Biological Discovery
1. **Protein Function**: Proteins clustering with known-function proteins likely have similar role
2. **Protein Complexes**: Identify functionally-related proteins via embedding similarity
3. **Disease Markers**: Monitor protein mislocalization changes in disease progression
4. **Novel Compartments**: Discover unknown cellular structures from embedding patterns

### Downstream Tasks
1. **Clustering**: Group proteins by localization using learned embeddings
2. **Classification**: Fine-tune on localization prediction task
3. **Similarity Search**: Find similar proteins in embedding space
4. **Anomaly Detection**: Identify proteins with unusual localization patterns

### Comparative Analysis
1. **Cell States**: How localization changes across cell types
2. **Drug Effects**: Monitor protein localization changes under treatment
3. **Disease Progression**: Track mislocalization as early disease indicator
4. **Evolution**: Compare protein localization across species


### Ablation Study Results

| Configuration | Reconstruction | Perplexity | Accuracy | Notes |
|---|---|---|---|---|
| With Classification | 0.05 | 350 | 78% | Rich, diverse codebook |
| Without Classification | 0.03 | 30 | - | Collapsed codebook |
| Different commitment_cost=0.1 | 0.06 | 280 | 75% | More codebook drift |
| Different commitment_cost=0.5 | 0.04 | 320 | 79% | Less encoder flexibility |


## References

### Key Papers
1. **VQ-VAE**: van den Oord et al. (2017) - "Neural Discrete Representation Learning"
2. **VAE**: Kingma & Welling (2013) - "Auto-Encoding Variational Bayes"
3. **Group Norm**: Yuxin Wu & Kaiming He (2018) - "Group Normalization"
4. **Straight-Through Estimator**: Bengio et al. (2013) - "Estimating and Backpropagating Gradients"
5. **Cytoself**: Kobayashi et al. (2022, Nature Methods) - "Self-supervised representation learning of microscopy images"
