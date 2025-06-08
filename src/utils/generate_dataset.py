import numpy as np
import torch
from PIL import Image, ImageDraw  # type: ignore
from sklearn.datasets import make_classification  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from .dataset import MultimodalSyntheticDataset


def generate_multimodal_dataset(num_samples=500, num_features=10, image_size=(64, 64), num_classes=2):
    """
    Generate a multimodal dataset using the MultimodalSyntheticDataset class.

    Args:
        num_samples (int): Number of samples.
        num_features (int): Number of structured features.
        image_size (tuple): Dimensions of the image (H, W).
        num_classes (int): Number of output classes.

    Returns:
        DataLoader: Dataloader containing structured data, image data, and labels.
    """

    print('Generate synthetic image data')

    dataset = MultimodalSyntheticDataset(
        num_samples=num_samples,
        num_features=num_features,
        image_size=image_size,
        num_classes=num_classes
    )
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    return next(iter(dataloader))
