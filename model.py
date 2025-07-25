import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class VectorQuantizer(nn.Module):
    """Discretizes continuous embeddings using a learned codebook of discrete vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook as a learnable parameter
        self.codebook = nn.Parameter(
            torch.randn(embedding_dim, num_embeddings) * 0.01
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: quantize inputs and return quantized output with losses.

        Args:
            inputs: Tensor of shape (batch, channels, height, width)

        Returns:
            ste: Straight-through estimator output
            perplexity: Metric for codebook usage diversity
            codebook_loss: Loss for updating codebook
            commitment_loss: Loss for committing encoder to codebook
            encoding_indices: Indices of selected codebook entries
        """
        # Flatten input for quantization
        flat_inputs = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances to codebook entries
        distances = self._calculate_distances(flat_inputs)

        # Find nearest codebook entry for each input vector
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize: replace inputs with nearest codebook entries
        quantized = self.codebook[:, encoding_indices].t().reshape(inputs.shape)

        # Calculate losses
        codebook_loss, commitment_loss = self._compute_losses(inputs, quantized, flat_inputs)

        # Calculate perplexity as a measure of codebook usage
        perplexity = self._calculate_perplexity(encoding_indices)

        # Straight-through estimator: copy gradient through quantization
        ste = inputs + (quantized - inputs).detach()

        return ste, perplexity, codebook_loss, commitment_loss, encoding_indices.reshape(inputs.shape[0], -1)

    def _calculate_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distances between inputs and codebook vectors."""
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 * x^T * e
        distances = (
            torch.sum(inputs ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook ** 2, dim=0, keepdim=True) -
            2 * torch.matmul(inputs, self.codebook)
        )
        return distances

    def _compute_losses(self, inputs: torch.Tensor, quantized: torch.Tensor, flat_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute codebook and commitment losses."""
        # Codebook loss: move codebook entries toward encoder outputs
        codebook_loss = F.mse_loss(quantized.detach(), inputs)

        # Commitment loss: move encoder outputs toward codebook entries
        commitment_loss = self.commitment_cost * F.mse_loss(quantized, inputs.detach())

        return codebook_loss, commitment_loss

    def _calculate_perplexity(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """Calculate perplexity as a measure of how many codebook entries are used."""
        # One-hot encode indices
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Average usage probability
        avg_probs = encodings.mean(dim=0)

        # Perplexity = exp(entropy)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(entropy)

        return perplexity

    def get_closest_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get indices of closest codebook entries without computing losses."""
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        distances = self._calculate_distances(flat_inputs)
        return torch.argmin(distances, dim=1)


class ResNetBlock(nn.Module):
    """Residual block with group normalization and swish activation."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=latent_dim)
        self.conv1 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=latent_dim)
        self.conv2 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block with skip connection."""
        h = F.silu(self.norm1(x))  # SiLU is the modern equivalent of Swish
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class Encoder(nn.Module):
    """Convolutional encoder that compresses images into latent feature maps."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Downsampling layers: reduce spatial dimensions by 4x, increase channels
        self.conv1 = nn.Conv2d(1, latent_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)

        # Residual refinement blocks
        self.res_block1 = ResNetBlock(latent_dim)
        self.res_block2 = ResNetBlock(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image into latent feature maps."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class Upsample(nn.Module):
    """Upsampling block using bilinear interpolation followed by convolution."""

    def __init__(self, latent_dim: int, upfactor: int):
        super().__init__()
        self.upfactor = upfactor
        self.conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample using bilinear interpolation and refine with convolution."""
        # Use bilinear interpolation to avoid checkerboard artifacts
        x = F.interpolate(x, scale_factor=self.upfactor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """Decoder that reconstructs images from quantized latent representations."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Refinement blocks
        self.res_block1 = ResNetBlock(latent_dim)
        self.res_block2 = ResNetBlock(latent_dim)

        # Upsampling layers: restore spatial dimensions
        self.upsample1 = Upsample(latent_dim, upfactor=2)
        self.upsample2 = Upsample(1, upfactor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode quantized representations back to image space."""
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.upsample1(x)
        x = F.relu(x)

        # After first upsample, reduce to single channel
        x = nn.Conv2d(self.latent_dim // 2, 1, kernel_size=3, stride=1, padding=1)(x)
        x = self.upsample2(x)

        return x


class ClassificationHead(nn.Module):
    """Fully connected classification head for auxiliary protein identification task."""

    def __init__(self, latent_size: int, num_classes: int, dropout_rate: float = 0.45, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        input_size = latent_size

        # Build dense layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_size, 1000))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_size = 1000

        # Output layer
        layers.append(nn.Linear(input_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict protein class from latent embeddings."""
        return self.network(x)


class LocalizationModel(nn.Module):
    """Complete VQ-VAE model for learning protein localization patterns."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        num_classes: int = None,
        dropout_rate: float = 0.45,
        classification_head_layers: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.num_classes = num_classes

        # Core VQ-VAE components
        self.encoder = Encoder(latent_dim=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = Decoder(latent_dim=embedding_dim)

        # Classification head for auxiliary task
        if num_classes is not None:
            # Calculate latent size after encoder (100x100 -> 25x25 after 2 strides of 2)
            latent_spatial_size = 25 * 25
            latent_size = embedding_dim * latent_spatial_size

            self.classification_head = ClassificationHead(
                latent_size=latent_size,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                num_layers=classification_head_layers,
            )
        else:
            self.classification_head = None

    def forward(self, x: torch.Tensor, training: bool = True) -> Dict:
        """
        Forward pass through the model.

        Args:
            x: Input images of shape (batch, 1, height, width)
            training: Whether in training mode (affects dropout)

        Returns:
            Dictionary containing reconstruction, losses, and logits
        """
        # Encode
        encoded = self.encoder(x)

        # Quantize
        quantized, perplexity, codebook_loss, commitment_loss, indices = self.quantizer(encoded)

        # Decode
        decoded = self.decoder(quantized)

        # Classification (if applicable)
        logits = None
        if self.classification_head is not None:
            # Flatten quantized representation for classification
            flat_z = quantized.reshape(quantized.shape[0], -1)
            logits = self.classification_head(flat_z)

        return {
            'reconstruction': decoded,
            'quantized': quantized,
            'perplexity': perplexity,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'logits': logits,
        }

    def get_encoding_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Get quantized indices for input images without computing losses."""
        encoded = self.encoder(x)
        indices = self.quantizer.get_closest_indices(encoded)
        return indices
