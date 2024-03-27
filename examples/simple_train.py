# We need to add the unetpp package to the system path so that we can import it
import sys
from pathlib import PurePath

unetpp_path = PurePath(__file__).parent.parent
sys.path.append(str(unetpp_path))

# Import the necessary packages
from datetime import datetime

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from unetpp.generators.default import SegmentationGenerator
from unetpp.model.unetpp import UNetPlusPlus
from unetpp.utils.datasets import train_test_val_split
from unetpp.utils.functions import dice_coefficient
from unetpp.utils.images import get_image_mask_pair_paths

# Replace the following paths with the actual paths
DATASET_FOLDER_PATH = "path/to/dataset/folder"
CHECKPOINT_FOLDER_PATH = "path/to/checkpoint/folder"
MODEL_FOLDER_PATH = "path/to/model/folder"

EPOCHS = 100
INPUT_SHAPE = (256, 512, 3)  # (height, width, channels)
COLORMAPS = [
    0,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
]  # Colormaps for the masks / classes / labels
BATCH_SIZE = 4  # Adjust this to your system's capabilities e.g. INPUT_SHAPE + COLORMAPS + BATCH_SIZE affects memory usage

NOW_STR = datetime.now().strftime("%y%m%d_%H%M%S")

# Build the model and print the summary
model = UNetPlusPlus(INPUT_SHAPE, len(COLORMAPS)).model
model.summary()

# Get image/mask paths
paths = get_image_mask_pair_paths(PurePath(DATASET_FOLDER_PATH))

# Split the paths into training, validation and test sets
train_paths, val_paths, test_paths = train_test_val_split(
    paths, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)

# Create a generators
train_generator = SegmentationGenerator(
    train_paths,
    colormap=COLORMAPS,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    shuffle=True,
)
validation_generator = SegmentationGenerator(
    val_paths,
    colormap=COLORMAPS,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Choose an optimizer (default settings)
optimizer = Adam()

# Compile the model
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=[dice_coefficient]
)

# Create a ModelCheckpoint callback
cp_filename = str(
    PurePath(CHECKPOINT_FOLDER_PATH) / f"unetpp_{NOW_STR}" / f"weights_{NOW_STR}.hdf5"
)
model_checkpoint_callback = ModelCheckpoint(
    filepath=cp_filename,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

# Create an EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=6, mode="min", verbose=1, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_generator.steps_per_epoch,
    callbacks=[model_checkpoint_callback, early_stopping_callback],
)

# Save the model
m_path = str(PurePath(MODEL_FOLDER_PATH) / f"unetpp_{NOW_STR}.h5")
model.save(m_path)
