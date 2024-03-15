from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import List
import cv2
from sklearn.utils.class_weight import compute_class_weight


@dataclass
class ImageMaskPairPaths:
    image_path: str
    mask_path: str


def get_image_mask_pair_paths(
    directory_path: str, input_suffix: str = "_input", mask_suffix: str = "_mask"
) -> list[ImageMaskPairPaths]:
    """Get a list of image/mask pairs from a directory"""
    pair_paths = []
    directory_path = Path(directory_path)

    # Iterate over all files in the provided directory
    for file_path in directory_path.iterdir():
        if file_path.stem.endswith(input_suffix):
            mask_path = file_path.with_name(
                file_path.stem.replace(input_suffix, mask_suffix) + file_path.suffix
            )
            if mask_path.exists():
                pair_paths.append(ImageMaskPairPaths(file_path, mask_path))

    return pair_paths


def split_mask_into_binary(mask: np.ndarray, colormap: List[int]) -> np.ndarray:
    return np.stack([np.all(mask == c, axis=-1) * 1 for c in colormap], axis=-1)


def join_binary_masks(mask: np.ndarray, colormap: List[int]) -> np.ndarray:
    return np.sum([mask[..., i, None] * c for i, c in enumerate(colormap)], axis=0)


def softmax_to_argmax(mask: np.ndarray) -> np.ndarray:
    binary_mask = np.zeros_like(mask)
    max_indices = np.argmax(mask, axis=-1)

    for i in range(mask.shape[-1]):
        binary_mask[:, :, i] = (max_indices == i).astype(int)

    return binary_mask


def get_class_weights(mask_paths: List[str], colormap: List[int]) -> dict[int, float]:
    class_labels = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique_classes = np.unique(mask)
        # Ignore classes not in colormap
        unique_classes = list(filter(lambda cls: cls in colormap, unique_classes))
        class_labels.extend(unique_classes)

    # Compute class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.array(colormap), y=class_labels
    )

    return {cls: weight for cls, weight in zip(colormap, class_weights)}
