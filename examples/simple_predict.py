# We need to add the unetpp package to the system path so that we can import it
import sys
from pathlib import PurePath

unetpp_path = PurePath(__file__).parent.parent
sys.path.append(str(unetpp_path))

# Import the necessary packages
from unetpp.utils.functions import dice_coefficient
from unetpp.utils.images import (
    join_binary_masks,
    softmax_to_argmax,
)
from unetpp.generators.default import SegmentationGenerator
from keras.saving import load_model
from PIL import Image

from unetpp.utils.images import get_image_mask_pair_paths

from datetime import datetime
from unetpp.utils.datasets import train_test_val_split


NOW_STR = datetime.now().strftime("%y%m%d_%H%M%S")

# Replace the following paths with the actual paths
MODEL_PATH = "path/to/model_folder/with.model.h5"
DATASET_FOLDER_PATH = "path/to/dataset_folder/"

INPUT_SHAPE = (256, 512, 3)  # (height, width, channels)
CLASSES = [0, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
BATCH_SIZE = 10

# Load saved Unet++ model
model = load_model(MODEL_PATH, custom_objects={"dice_coefficient": dice_coefficient})

# Get image/mask paths
paths = get_image_mask_pair_paths(PurePath(DATASET_FOLDER_PATH))

# Split the paths into training, validation and test sets
_, _, test_paths = train_test_val_split(
    paths, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)

# Create a test generator
test_generator = SegmentationGenerator(
    test_paths,
    colormap=CLASSES,
    target_size=INPUT_SHAPE[:2],
    batch_size=10,
    shuffle=True,
)

# Shuffle the test set
test_generator.on_epoch_end()

# Get one batch and predict
images, masks = test_generator[0]

# Predict
for i in range(0, BATCH_SIZE):
    pred = model.predict(images[i].reshape(1, *INPUT_SHAPE))[0]
    # Each pixel has 11 probabilities (11 classes)
    # However, we are only interested in the class with the highest probability!
    # Use the argmax to convert the softmax output (probabilities) to either 0 or 1 (binary mask)
    # Then, join the binary masks into a single mask
    pred = softmax_to_argmax(pred)
    pred = join_binary_masks(pred, CLASSES)
    pred = pred.astype("uint8")

    # Save the (resized) input image, mask and prediction
    image = images[i].astype("uint8")
    input_image = Image.fromarray(image)

    mask = join_binary_masks(masks[i], CLASSES)
    mask = (mask * [1, 1, 1]).astype("uint8")
    input_mask = Image.fromarray(mask)

    pred = (pred * [1, 1, 1]).astype("uint8")
    pred_mask = Image.fromarray(pred)

    # Join the images from left to right (input, ground truth mask, prediction mask)
    # Add a padding of 2 pixels between the images
    joined_image = Image.new(
        "RGB", (3 * INPUT_SHAPE[1] + 2 * 2, INPUT_SHAPE[0]), "white"
    )
    joined_image.paste(input_image, (0, 0))
    joined_image.paste(input_mask, (INPUT_SHAPE[1] + 2, 0))
    joined_image.paste(pred_mask, (2 * INPUT_SHAPE[1] + 2 * 2, 0))

    # Save the joined image
    joined_image.save(f"result_{NOW_STR}_{i}.png")
