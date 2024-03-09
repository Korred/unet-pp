from unetpp.utils.images import ImageMaskPairPaths

from typing import List, Tuple
import numpy as np


def train_test_val_split(
    paths: List[ImageMaskPairPaths],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[
    List[ImageMaskPairPaths], List[ImageMaskPairPaths], List[ImageMaskPairPaths]
]:
    # Ensure that the ratios are valid
    tolerance = 1e-7
    if abs((train_ratio + val_ratio + test_ratio) - 1) > tolerance:
        raise ValueError("The sum of the train, validation and test ratios must be 1")

    # Provide at least 10 path pairs
    if len(paths) < 10:
        raise ValueError("There must be at least 10 path pairs")

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(paths)

    total_cnt = len(paths)
    train_cnt = int(total_cnt * train_ratio)
    val_cnt = int(total_cnt * val_ratio)

    train_paths = paths[:train_cnt]
    val_paths = paths[train_cnt : train_cnt + val_cnt]
    test_paths = paths[train_cnt + val_cnt :]

    return train_paths, val_paths, test_paths
