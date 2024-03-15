import os
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# Function to read images and extract class labels
def class_weigt(input_path):

    TARGET_MASK_SUFFIX = "_mask.png"
    class_labels = []

    for image_file in os.listdir(input_path):
        if image_file.endswith(TARGET_MASK_SUFFIX):
            image_path = os.path.join(input_path, image_file)
            # Read the image using cv2.imread
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            unique_classes = np.unique(image)
            class_labels.extend(unique_classes)

    # Compute class weights
    unique_classes = np.unique(class_labels)
    class_weights = compute_class_weight(
        "balanced", classes=unique_classes, y=class_labels
    )
    class_weights_dict = {
        cls: weight for cls, weight in zip(unique_classes, class_weights)
    }
    return class_weights_dict
