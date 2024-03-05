import numpy as np
from keras.utils import Sequence

from keras.preprocessing.image import load_img, img_to_array
from typing import Tuple


from utils.images import ImageMaskPairPaths, split_mask_into_binary


class SegmentationGenerator(Sequence):
    def __init__(
        self,
        paths: list[ImageMaskPairPaths],
        colormap: list[int],
        target_size: tuple[int, int],
        batch_size: int = 32,
        shuffle=True,
    ):
        self.paths = paths
        self.colormap = colormap
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.paths))
        self.steps_per_epoch = self._get_steps_per_epoch()

    def _get_steps_per_epoch(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    @staticmethod
    def read_image_and_resize(
        path: str, channels: int, target_size: Tuple[int, int]
    ) -> np.ndarray:
        CHANNEL_LKP = {1: "grayscale", 3: "rgb", 4: "rgba"}
        try:
            img = img_to_array(
                load_img(
                    path, color_mode=CHANNEL_LKP[channels], target_size=target_size
                )
            )
            return img
        except KeyError:
            raise ValueError("Invalid number of channels. Must be 1, 3 or 4.")

    def _parse_images_and_masks(self, batch_indexes) -> tuple[np.ndarray, np.ndarray]:
        images = []
        masks = []

        for index in batch_indexes:
            image_path = self.paths[index].image_path
            mask_path = self.paths[index].mask_path

            # Read and resize the image and mask
            image = self.read_image_and_resize(image_path, 3, self.target_size)
            images.append(image)
            # Raw mask
            mask = self.read_image_and_resize(mask_path, 1, self.target_size)
            # Split the mask into binary masks
            mask = split_mask_into_binary(mask, self.colormap)

            masks.append(mask)

        return (images, masks)

    def __len__(self):
        return self._get_steps_per_epoch()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        images, masks = self._parse_images_and_masks(batch_indexes)
        images = np.stack(images, axis=0)
        masks = np.stack(masks, axis=0)

        return (images, masks)
