"""
Visualization utilities for analyzing VQ-VAE model outputs and training metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from model import LocalizationModel
from dataset import Dataset


def visualize_reconstructions(
    model: LocalizationModel,
    dataset: Dataset,
    num_samples: int = 8,
    device: Optional[torch.device] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Visualize original and reconstructed images.

    Args:
        model: Trained LocalizationModel
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to use for inference
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Sample random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            image = sample['image'].unsqueeze(0).to(device)

            # Get reconstruction
            outputs = model(image, training=False)
            reconstruction = outputs['reconstruction']

            # Denormalize for display
            orig_display = image.squeeze().cpu().numpy()
            recon_display = reconstruction.squeeze().cpu().numpy()

            # Plot original
            axes[idx, 0].imshow(orig_display, cmap='gray')
            axes[idx, 0].set_title(f'Original (ID: {sample["frame_id"]})')
            axes[idx, 0].axis('off')

            # Plot reconstruction
            axes[idx, 1].imshow(recon_display, cmap='gray')
            axes[idx, 1].set_title('Reconstruction')
            axes[idx, 1].axis('off')

    plt.tight_layout()
    return fig


def plot_training_metrics(
    metrics: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot training metrics over epochs.

    Args:
        metrics: Dictionary of metrics from training
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Extract metrics
    train_metrics = metrics.get('train', {})
    valid_metrics = metrics.get('valid', {})

    # Prepare data
    epochs = np.arange(len(train_metrics.get('total_loss', [])))

    # Plot 1: Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if 'total_loss' in train_metrics:
        ax1.plot(epochs, train_metrics['total_loss'], 'b-', label='Train', linewidth=2)
    if 'total_loss' in valid_metrics:
        ax1.plot(epochs, valid_metrics['total_loss'], 'r--', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss Components
    ax2 = fig.add_subplot(gs[0, 1])
    loss_components = ['recon_loss', 'codebook_loss', 'commitment_loss']
    for component in loss_components:
        if component in train_metrics:
            values = train_metrics[component]
            if isinstance(values[0], np.ndarray):
                values = [np.mean(v) if isinstance(v, np.ndarray) else v for v in values]
            ax2.plot(epochs, values, label=component, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components (Training)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Perplexity
    ax3 = fig.add_subplot(gs[1, 0])
    if 'perplexity' in train_metrics:
        train_perp = train_metrics['perplexity']
        if isinstance(train_perp[0], np.ndarray):
            train_perp = [np.mean(p) if isinstance(p, np.ndarray) else p for p in train_perp]
        ax3.plot(epochs, train_perp, 'b-', label='Train', linewidth=2)
    if 'perplexity' in valid_metrics:
        valid_perp = valid_metrics['perplexity']
        if isinstance(valid_perp[0], np.ndarray):
            valid_perp = [np.mean(p) if isinstance(p, np.ndarray) else p for p in valid_perp]
        ax3.plot(epochs, valid_perp, 'r--', label='Validation', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('Codebook Usage Perplexity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Classification Accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    if 'accuracy' in train_metrics:
        train_acc = train_metrics['accuracy']
        if isinstance(train_acc[0], np.ndarray):
            train_acc = [np.mean(a) if isinstance(a, np.ndarray) else a for a in train_acc]
        ax4.plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    if 'accuracy' in valid_metrics:
        valid_acc = valid_metrics['accuracy']
        if isinstance(valid_acc[0], np.ndarray):
            valid_acc = [np.mean(a) if isinstance(a, np.ndarray) else a for a in valid_acc]
        ax4.plot(epochs, valid_acc, 'r--', label='Validation', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Classification Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Classification Loss
    ax5 = fig.add_subplot(gs[2, 0])
    if 'classification_loss' in train_metrics:
        train_clf = train_metrics['classification_loss']
        if isinstance(train_clf[0], np.ndarray):
            train_clf = [np.mean(c) if isinstance(c, np.ndarray) else c for c in train_clf]
        ax5.plot(epochs, train_clf, 'b-', label='Train', linewidth=2)
    if 'classification_loss' in valid_metrics:
        valid_clf = valid_metrics['classification_loss']
        if isinstance(valid_clf[0], np.ndarray):
            valid_clf = [np.mean(c) if isinstance(c, np.ndarray) else c for c in valid_clf]
        ax5.plot(epochs, valid_clf, 'r--', label='Validation', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Classification Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Reconstruction Loss
    ax6 = fig.add_subplot(gs[2, 1])
    if 'recon_loss' in train_metrics:
        train_recon = train_metrics['recon_loss']
        if isinstance(train_recon[0], np.ndarray):
            train_recon = [np.mean(r) if isinstance(r, np.ndarray) else r for r in train_recon]
        ax6.plot(epochs, train_recon, 'b-', label='Train', linewidth=2)
    if 'recon_loss' in valid_metrics:
        valid_recon = valid_metrics['recon_loss']
        if isinstance(valid_recon[0], np.ndarray):
            valid_recon = [np.mean(r) if isinstance(r, np.ndarray) else r for r in valid_recon]
        ax6.plot(epochs, valid_recon, 'r--', label='Validation', linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss')
    ax6.set_title('Reconstruction Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Training Metrics Overview', fontsize=16, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_codebook_usage(
    model: LocalizationModel,
    dataset: Dataset,
    device: Optional[torch.device] = None,
    num_samples: int = 1000,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Visualize codebook usage distribution.

    Args:
        model: Trained LocalizationModel
        dataset: Dataset to sample from
        device: Device to use for inference
        num_samples: Number of samples to use for analysis
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Collect encoding indices
    all_indices = []
    num_batches = min(100, (len(dataset) + 31) // 32)  # 32-sample batches

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * 32
            end_idx = min((batch_idx + 1) * 32, len(dataset))

            images = []
            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                images.append(sample['image'])

            batch_images = torch.stack(images).to(device)
            indices = model.get_encoding_indices(batch_images)
            all_indices.extend(indices.cpu().numpy().flatten())

    all_indices = np.array(all_indices)

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histogram of codebook usage
    axes[0].hist(all_indices, bins=model.num_embeddings, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Codebook Index')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Codebook Usage Distribution')
    axes[0].grid(True, alpha=0.3)

    # Unique entries used
    unique_entries = len(np.unique(all_indices))
    axes[1].bar([0], [unique_entries], color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Unique Codebook Entries Used: {unique_entries}/{model.num_embeddings}')
    axes[1].set_ylim([0, model.num_embeddings])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Usage distribution
    unique, counts = np.unique(all_indices, return_counts=True)
    usage_dist = counts / counts.sum()
    entropy = -np.sum(usage_dist * np.log(usage_dist + 1e-10))
    perplexity = np.exp(entropy)

    axes[2].bar([0], [perplexity], color='coral', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title(f'Codebook Perplexity: {perplexity:.2f}')
    axes[2].set_ylim([0, model.num_embeddings])
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def visualize_latent_space(
    model: LocalizationModel,
    dataset: Dataset,
    device: Optional[torch.device] = None,
    num_samples: int = 500,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Visualize latent space structure using t-SNE or PCA.

    Args:
        model: Trained LocalizationModel
        dataset: Dataset to sample from
        device: Device to use for inference
        num_samples: Number of samples to visualize
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn required for latent space visualization")
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Collect embeddings and labels
    embeddings = []
    labels = []

    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)

            outputs = model(image, training=False)
            embedding = outputs['quantized'].squeeze(0).permute(1, 2, 0).reshape(-1, model.embedding_dim)

            # Take mean embedding across spatial dimensions
            mean_embedding = embedding.mean(dim=0).cpu().numpy()

            embeddings.append(mean_embedding)
            labels.append(sample['protein_id'])

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Reduce dimensionality
    if embeddings.shape[1] > 2:
        # Use PCA first to reduce computational cost
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        embeddings_reduced = pca.fit_transform(embeddings)

        # Then apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_reduced)
    else:
        embeddings_2d = embeddings

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20', alpha=0.6, s=50)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Latent Space Structure (t-SNE + PCA)')
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Protein ID')
    plt.tight_layout()

    return fig
