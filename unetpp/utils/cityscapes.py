import os
import random
import numpy as np
import cv2

import tensorflow as tf


class SegmentationGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        input_path,
        batch_size=32,
        image_size=(256, 256),
        validation_split=0.2,
        test_split=0.1,
        seed=42,
    ):
        self.input_path = input_path
        self.batch_size = batch_size
        self.image_size = image_size

        # Get list of input files
        input_files = os.listdir(self.input_path)
        input_files = [file for file in input_files if file.endswith("_input.png")]

        # Shuffle the list of input files
        random.seed(seed)
        random.shuffle(input_files)

        # Calculate the number of samples for each split
        num_samples = len(input_files)
        num_val = int(num_samples * validation_split)
        num_test = int(num_samples * test_split)
        num_train = num_samples - num_val - num_test

        # Assign the files to the respective splits
        self.train_files = input_files[:num_train]
        self.val_files = input_files[num_train : num_train + num_val]
        self.test_files = input_files[num_train + num_val :]

    def load_image(self, filename):
        img = cv2.imread(os.path.join(self.input_path, filename))
        img = cv2.resize(img, self.image_size)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        return img

    def load_mask(self, filename):
        mask = cv2.imread(os.path.join(self.input_path, filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size)
        mask = mask / 255.0  # Normalize pixel values to [0, 1]
        return np.expand_dims(mask, axis=-1)  # Add channel dimension

    def __len__(self):
        return int(np.ceil(len(self.train_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.train_files[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = [filename.replace("_input.png", "_mask.png") for filename in batch_x]

        X = [self.load_image(filename) for filename in batch_x]
        y = [self.load_mask(mask_filename) for mask_filename in batch_y]

        return np.array(X), np.array(y)


generator = SegmentationGenerator(
    input_path=input_path,
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    test_split=0.1,
)
