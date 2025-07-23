import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import os


class ImageLoader:
    """Handles loading and caching of microscopy images."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.images = None

    def load(self, force_recreate: bool = False) -> np.ndarray:
        """Load images from disk or return cached version."""
        cache_path = os.path.join(self.data_path, 'images_cache.npy')

        if os.path.exists(cache_path) and not force_recreate:
            self.images = np.load(cache_path, mmap_mode='r')
        else:
            # If no cached images exist, initialize empty array
            # In practice, images would be loaded from tiff/png files
            self.images = np.zeros((1000, 100, 100), dtype=np.uint8)

        return self.images


class LabelLoader:
    """Handles loading and caching of labels and metadata."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.lookup = None

    def load(self, force_recreate: bool = False) -> pd.DataFrame:
        """Load labels from disk or return cached version."""
        cache_path = os.path.join(self.data_path, 'labels_cache.csv')

        if os.path.exists(cache_path) and not force_recreate:
            self.lookup = pd.read_csv(cache_path)
        else:
            # Initialize with sample data structure
            self.lookup = pd.DataFrame({
                'frame_id': range(1000),
                'protein_id': np.random.randint(0, 100, 1000),
                'code': np.random.randint(0, 100, 1000),
                'fov_id': np.repeat(np.arange(50), 20),
                'gene_symbol': [f'PROTEIN_{i % 100}' for i in range(1000)],
            })

        return self.lookup

    def get_n_proteins(self) -> int:
        """Get number of unique proteins."""
        return self.lookup['protein_id'].nunique()

    def get_frame_ids(self) -> np.ndarray:
        """Get all frame IDs."""
        return self.lookup['frame_id'].values

    def get_frames_of_random_proteins(self, n_proteins: int) -> pd.DataFrame:
        """Get frames of randomly selected proteins."""
        unique_proteins = self.lookup['protein_id'].unique()
        selected = np.random.choice(unique_proteins, min(n_proteins, len(unique_proteins)), replace=False)
        return self.lookup[self.lookup['protein_id'].isin(selected)]


class Labels:
    """Container for label information."""

    def __init__(self, lookup: pd.DataFrame):
        self.lookup = lookup

    def get_frame_ids(self) -> np.ndarray:
        """Get frame IDs from lookup table."""
        return self.lookup['frame_id'].values


class Images:
    """Container for image data with memory-mapped access."""

    def __init__(self, frames: np.ndarray):
        self.frames = frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get image by index."""
        return torch.FloatTensor(self.frames[idx] / 255.0).unsqueeze(0)  # Normalize and add channel


class Dataset(TorchDataset):
    """Dataset class for protein localization images."""

    def __init__(self, images: np.ndarray, labels: pd.DataFrame):
        self.images = images
        self.labels = labels
        self.frame_ids = labels['frame_id'].values

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        frame_id = self.frame_ids[idx]
        image = torch.FloatTensor(self.images[frame_id] / 255.0).unsqueeze(0)  # Normalize
        label_row = self.labels[self.labels['frame_id'] == frame_id].iloc[0]

        return {
            'image': image,
            'protein_id': torch.tensor(label_row['code'], dtype=torch.long),
            'frame_id': frame_id,
        }

    def get_batches(self, batch_size: int, shuffle: bool = True):
        """Yield batches of data."""
        indices = np.arange(len(self))

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self), batch_size):
            end_idx = min(start_idx + batch_size, len(self))
            batch_indices = indices[start_idx:end_idx]

            images = []
            labels = []
            frame_ids = []

            for idx in batch_indices:
                sample = self.__getitem__(idx)
                images.append(sample['image'])
                labels.append(sample['protein_id'])
                frame_ids.append(sample['frame_id'])

            yield {
                'images': torch.stack(images),
                'labels': torch.stack(labels),
                'frame_ids': np.array(frame_ids),
            }

    def get_dummy_input(self) -> torch.Tensor:
        """Get a dummy input for model initialization."""
        return torch.randn(1, 1, 100, 100)

    def plot_random_frames(self, n: int = 16, **kwargs):
        """Placeholder for visualization (would use matplotlib in practice)."""
        print(f"Plotting {n} random frames from dataset")


class DatasetBuilder:
    """Builds datasets with train/val/test splits."""

    def __init__(self, data_path: str, force_recreate: bool = False):
        self.data_path = data_path
        self.images_loader = ImageLoader(data_path)
        self.labels_loader = LabelLoader(data_path)

        self.images = self.images_loader.load(force_recreate=force_recreate)
        self.labels = self.labels_loader.load(force_recreate=force_recreate)

    def build(
        self,
        splits: Dict[str, float],
        exclusive_by: str = "fov_id",
        n_proteins: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Dataset]:
        """
        Build dataset splits.

        Args:
            splits: Dictionary of split names and their fractional sizes
            exclusive_by: Field to use for exclusive split assignment
            n_proteins: Limit to this many proteins
            max_frames: Limit to this many frames

        Returns:
            Dictionary of split names to Dataset objects
        """
        # Sample proteins if specified
        if n_proteins:
            unique_proteins = self.labels['protein_id'].unique()
            selected_proteins = np.random.choice(
                unique_proteins,
                min(n_proteins, len(unique_proteins)),
                replace=False
            )
            frames = self.labels[self.labels['protein_id'].isin(selected_proteins)].reset_index(drop=True)
        else:
            frames = self.labels.reset_index(drop=True)

        # Limit frames if specified
        if max_frames and len(frames) > max_frames:
            frames = frames.head(max_frames)

        # Get unique values for exclusive assignment
        unique_ids = frames[exclusive_by].unique()
        np.random.shuffle(unique_ids)

        # Assign frames to splits
        dataset_splits = {}
        start_idx = 0

        split_names = list(splits.keys())
        split_sizes = []
        for name in split_names[:-1]:
            split_sizes.append(int(len(unique_ids) * splits[name]))
        split_sizes.append(len(unique_ids) - sum(split_sizes))  # Remainder to last split

        # Encode proteins to consecutive integers
        unique_protein_ids = sorted(frames['protein_id'].unique())
        protein_to_code = {pid: idx for idx, pid in enumerate(unique_protein_ids)}
        frames['code'] = frames['protein_id'].map(protein_to_code)

        for split_name, split_size in zip(split_names, split_sizes):
            selected_ids = unique_ids[start_idx:start_idx + split_size]
            split_frames = frames[frames[exclusive_by].isin(selected_ids)].reset_index(drop=True)

            dataset_splits[split_name] = Dataset(
                images=self.images,
                labels=split_frames,
            )

            start_idx += split_size

        return dataset_splits


def get_dataset(data_path: str, **kwargs) -> Dataset:
    """Utility function to get a dataset."""
    builder = DatasetBuilder(data_path)
    splits = builder.build(
        splits={"train": 0.8, "valid": 0.1, "test": 0.1},
        exclusive_by="fov_id",
        n_proteins=kwargs.get('n_proteins', None),
        max_frames=kwargs.get('max_frames', None),
    )
    return splits['train']


def count_unique_proteins(dataset_splits: Dict[str, Dataset]) -> int:
    """Count unique proteins across all splits."""
    all_codes = []
    for split in dataset_splits.values():
        all_codes.extend(split.labels['code'].unique())
    return len(set(all_codes))
