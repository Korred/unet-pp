# We need to add the unetpp package to the system path so that we can import it
import sys
from pathlib import PurePath

unetpp_path = PurePath(__file__).parent.parent
sys.path.append(str(unetpp_path))

# Import the necessary packages
from unetpp.model.unetpp import UNetPlusPlus
from unetpp.utils.functions import dice_coefficient
from unetpp.utils.images import (
    split_mask_into_binary,
    join_binary_masks,
    softmax_to_argmax,
)
from unetpp.generators.default import SegmentationGenerator
from keras.saving import load_model
from PIL import Image

from datetime import datetime


NOW_STR = datetime.now().strftime("%y%m%d_%H%M%S")

# Replace the following paths with the actual paths
MODEL_PATH = "path/to/model/folder/with.model.hdf5"
IMAGE_PATH = "path/to/image.png"
MASK_PATH = "path/to/mask.png"

INPUT_SHAPE = (256, 512, 3)  # (height, width, channels)

# We assume the model was trained on the classes [0, 24, 26] (background, person, car)
CLASSES = [0, 24, 26]


# Load model
model = load_model(MODEL_PATH, custom_objects={"dice_coefficient": dice_coefficient})

# When loding checkpoints, use the following code instead
"""
model = UNetPlusPlus(INPUT_SHAPE, 3).model
model.load_weights(MODEL_PATH)
"""

# Load image and mask for prediction (and comparison)
image = SegmentationGenerator.read_image_and_resize(IMAGE_PATH, 3, INPUT_SHAPE[:2])
mask = SegmentationGenerator.read_image_and_resize(MASK_PATH, 1, INPUT_SHAPE[:2])
mask = split_mask_into_binary(mask, CLASSES)

# Predict
pred = model.predict(image.reshape(1, *INPUT_SHAPE))[0]
pred = softmax_to_argmax(pred)
pred = join_binary_masks(pred, CLASSES)
pred = pred.astype("uint8")

# Save the input image, mask and prediction
image = image.astype("uint8")
input_image = Image.fromarray(image)

mask = join_binary_masks(mask, CLASSES)
mask = (mask * [1, 1, 1]).astype("uint8")
input_mask = Image.fromarray(mask)

pred = (pred * [1, 1, 1]).astype("uint8")
pred_mask = Image.fromarray(pred)

# Join the images from left to right
# Add a white line (2px) between all images
joined_image = Image.new("RGB", (3 * INPUT_SHAPE[1] + 2 * 2, INPUT_SHAPE[0]), "white")
joined_image.paste(input_image, (0, 0))
joined_image.paste(input_mask, (INPUT_SHAPE[1] + 2, 0))
joined_image.paste(pred_mask, (2 * INPUT_SHAPE[1] + 2 * 2, 0))

# Save the joined image
joined_image.save(f"result_{NOW_STR}.png")
