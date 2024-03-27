# We need to add the unetpp package to the system path so that we can import it
import sys
from pathlib import PurePath

unetpp_path = PurePath(__file__).parent.parent
sys.path.append(str(unetpp_path))

from keras.saving import load_model

from unetpp.generators.default import SegmentationGenerator
from unetpp.utils.datasets import train_test_val_split
from unetpp.utils.functions import dice_coefficient
from unetpp.utils.images import get_image_mask_pair_paths

DATASET_FOLDER_PATH = "/home/korred/repos/unet-pp/data/output/"

MODEL_PATHS = [
    "/home/korred/repos/unet-pp/data/models/model_1_240312_092929.h5",
    "/home/korred/repos/unet-pp/data/models/model_3_240312_092929.h5",
    "/home/korred/repos/unet-pp/data/models/model_4_240312_092929.h5",
]

INPUT_SHAPE = (256, 512, 3)  # (height, width, channels)
CLASSES = [0, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]


# Get image/mask paths
paths = get_image_mask_pair_paths(PurePath(DATASET_FOLDER_PATH))

# Split the paths into training, validation and test sets
# Ignore train and val sets
_, _, test_paths = train_test_val_split(
    paths, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)


for model_path in MODEL_PATHS:
    # Load model
    model = load_model(
        model_path, custom_objects={"dice_coefficient": dice_coefficient}
    )

    # Create generator
    test_generator = SegmentationGenerator(
        test_paths,
        colormap=CLASSES,
        target_size=INPUT_SHAPE[:2],
        batch_size=2,
        shuffle=False,
    )

    # Evaluate the model
    scores = model.evaluate(test_generator)

    print(f"Model: {model_path}")
    print(f"Test loss: {scores[0]}")
    print(f"Test dice coefficient: {scores[1]}")
    print("\n")
