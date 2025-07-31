import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import json
from sklearn.metrics import accuracy_score
from model import LocalizationModel
from dataset import Dataset, DatasetBuilder


class MetricsLogger:
    """Logs training metrics and provides export functionality."""

    def __init__(self):
        self.metrics = {}
        self.current_epoch = None

    def log_step(self, split: str, **kwargs):
        """Log metrics for a step."""
        if split not in self.metrics:
            self.metrics[split] = {}

        for key, value in kwargs.items():
            if key not in self.metrics[split]:
                self.metrics[split][key] = []

            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.detach().cpu().numpy()

            self.metrics[split][key].append(value)

    def flush(self, epoch: int):
        """Aggregate metrics for the epoch."""
        self.current_epoch = epoch

    def latest(self, keys: list) -> str:
        """Get latest metrics as string."""
        results = []
        for key in keys:
            for split in self.metrics:
                if key in self.metrics[split]:
                    latest_val = self.metrics[split][key][-1]
                    if isinstance(latest_val, np.ndarray):
                        latest_val = latest_val.mean()
                    results.append(f"{key}={latest_val:.4f}")
                    break
        return ", ".join(results)

    def export(self) -> Dict[str, Dict]:
        """Export metrics as dictionary."""
        exported = {}
        for split, split_metrics in self.metrics.items():
            exported[split] = {}
            for key, values in split_metrics.items():
                # Average over batches for each epoch
                if isinstance(values[0], np.ndarray):
                    exported[split][key] = [np.mean(values)]
                else:
                    exported[split][key] = [values[0]] if len(values) == 1 else values
        return exported


def train_step(
    model: LocalizationModel,
    optimizer: optim.Optimizer,
    batch: Dict,
    device: torch.device,
    classification_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Perform a single training step.

    Args:
        model: The LocalizationModel
        optimizer: Optimizer for parameter updates
        batch: Dictionary with 'images' and 'labels'
        device: Device to run on
        classification_weight: Weight for classification loss

    Returns:
        Dictionary of loss components and metrics
    """
    model.train()

    images = batch['images'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass
    outputs = model(images, training=True)

    reconstruction = outputs['reconstruction']
    perplexity = outputs['perplexity']
    codebook_loss = outputs['codebook_loss']
    commitment_loss = outputs['commitment_loss']
    logits = outputs['logits']

    # Compute losses
    recon_loss = nn.MSELoss()(reconstruction, images)

    total_loss = recon_loss + codebook_loss + commitment_loss

    # Classification loss (auxiliary task)
    classification_loss = torch.zeros(1, device=device)
    accuracy = torch.tensor(0.0, device=device)

    if logits is not None:
        classification_loss = classification_weight * nn.CrossEntropyLoss()(logits, labels)
        total_loss += classification_loss

        # Calculate accuracy
        preds = logits.argmax(dim=1).cpu().numpy()
        accuracy = torch.tensor(accuracy_score(labels.cpu().numpy(), preds), device=device)

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.detach(),
        'recon_loss': recon_loss.detach(),
        'codebook_loss': codebook_loss.detach(),
        'commitment_loss': commitment_loss.detach(),
        'classification_loss': classification_loss.detach(),
        'perplexity': perplexity.detach(),
        'accuracy': accuracy.detach(),
    }


@torch.no_grad()
def eval_step(
    model: LocalizationModel,
    batch: Dict,
    device: torch.device,
    classification_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Perform a single evaluation step.

    Args:
        model: The LocalizationModel
        batch: Dictionary with 'images' and 'labels'
        device: Device to run on
        classification_weight: Weight for classification loss

    Returns:
        Dictionary of loss components and metrics
    """
    model.eval()

    images = batch['images'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass
    outputs = model(images, training=False)

    reconstruction = outputs['reconstruction']
    perplexity = outputs['perplexity']
    codebook_loss = outputs['codebook_loss']
    commitment_loss = outputs['commitment_loss']
    logits = outputs['logits']

    # Compute losses
    recon_loss = nn.MSELoss()(reconstruction, images)

    total_loss = recon_loss + codebook_loss + commitment_loss

    # Classification loss (auxiliary task)
    classification_loss = torch.zeros(1, device=device)
    accuracy = torch.tensor(0.0, device=device)

    if logits is not None:
        classification_loss = classification_weight * nn.CrossEntropyLoss()(logits, labels)
        total_loss += classification_loss

        # Calculate accuracy
        preds = logits.argmax(dim=1).cpu().numpy()
        accuracy = torch.tensor(accuracy_score(labels.cpu().numpy(), preds), device=device)

    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'codebook_loss': codebook_loss,
        'commitment_loss': commitment_loss,
        'classification_loss': classification_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
    }


def train(
    model: LocalizationModel,
    dataset_splits: Dict[str, Dataset],
    num_epochs: int,
    batch_size: int,
    learning_rate: float = 0.001,
    classification_weight: float = 1.0,
    eval_every: int = 1,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Tuple[LocalizationModel, Dict]:
    """
    Train the VQ-VAE model.

    Args:
        model: LocalizationModel instance
        dataset_splits: Dictionary of dataset splits
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        classification_weight: Weight for classification loss
        eval_every: Evaluate every N epochs
        device: Device to use
        save_path: Path to save model checkpoints

    Returns:
        Tuple of trained model and metrics dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup metrics logging
    metrics_logger = MetricsLogger()

    # Create data loaders
    train_loader = DataLoader(
        dataset_splits['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        dataset_splits['valid'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    ) if 'valid' in dataset_splits else None

    test_loader = DataLoader(
        dataset_splits['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    ) if 'test' in dataset_splits else None

    # Training loop
    epochs = tqdm(range(num_epochs), desc="Training")

    for epoch in epochs:
        # Training phase
        train_metrics_list = []
        for batch_idx, batch in enumerate(train_loader):
            step_metrics = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
                classification_weight=classification_weight,
            )

            train_metrics_list.append(step_metrics)
            metrics_logger.log_step(split="train", **step_metrics)

        # Validation phase
        if valid_loader and epoch % eval_every == 0:
            valid_metrics_list = []
            for batch_idx, batch in enumerate(valid_loader):
                step_metrics = eval_step(
                    model=model,
                    batch=batch,
                    device=device,
                    classification_weight=classification_weight,
                )
                valid_metrics_list.append(step_metrics)
                metrics_logger.log_step(split="valid", **step_metrics)

        # Update progress bar
        metrics_logger.flush(epoch=epoch)
        postfix_str = metrics_logger.latest(['total_loss', 'perplexity'])
        epochs.set_postfix_str(postfix_str)

        # Save checkpoint if requested
        if save_path:
            checkpoint_dir = Path(save_path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics_logger.export(),
            }, checkpoint_path)

    return model, metrics_logger.export()


def create_model_and_train(
    data_path: str,
    num_proteins: int = 50,
    embedding_dim: int = 64,
    num_embeddings: int = 512,
    commitment_cost: float = 0.25,
    dropout_rate: float = 0.45,
    classification_head_layers: int = 2,
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    classification_weight: float = 1.0,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Tuple[LocalizationModel, Dict]:
    """
    Create and train a LocalizationModel from scratch.

    Args:
        data_path: Path to dataset
        num_proteins: Number of proteins to use
        embedding_dim: Dimension of embeddings
        num_embeddings: Number of codebook entries
        commitment_cost: VQ loss scaling factor
        dropout_rate: Dropout rate
        classification_head_layers: Number of classification head layers
        num_epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate
        classification_weight: Weight for classification loss
        device: Device to use
        save_path: Path to save checkpoints

    Returns:
        Trained model and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    builder = DatasetBuilder(data_path)
    dataset_splits = builder.build(
        splits={"train": 0.8, "valid": 0.1, "test": 0.1},
        exclusive_by="fov_id",
        n_proteins=num_proteins,
    )

    # Count unique proteins for classification head
    unique_proteins = set()
    for split in dataset_splits.values():
        unique_proteins.update(split.labels['code'].unique())
    num_classes = len(unique_proteins)

    # Create model
    model = LocalizationModel(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        classification_head_layers=classification_head_layers,
    )

    # Train
    trained_model, metrics = train(
        model=model,
        dataset_splits=dataset_splits,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        classification_weight=classification_weight,
        device=device,
        save_path=save_path,
    )

    return trained_model, metrics
