# We need to add the unetpp package to the system path so that we can import it
import sys
from pathlib import PurePath

unetpp_path = PurePath(__file__).parent.parent
sys.path.append(str(unetpp_path))


import keras_tuner as kt
from unetpp.utils.images import get_image_mask_pair_paths, get_class_weights
from unetpp.utils.datasets import train_test_val_split
from unetpp.generators.default import SegmentationGenerator
from unetpp.model.unetpp import UNetPlusPlus
from unetpp.utils.functions import dice_coefficient
from keras.callbacks import EarlyStopping
from datetime import datetime

from keras.optimizers import Adam, SGD, Adamax
from keras.callbacks import TensorBoard
from keras.losses import CategoricalCrossentropy, CategoricalFocalCrossentropy

# Replace the following paths with the actual paths
DATASET_FOLDER_PATH = "/home/korred/repos/unet-pp/data/output/"
MODEL_FOLDER_PATH = "/home/korred/repos/unet-pp/data/model/"
TUNER_PROJECT_FOLDER = "/home/korred/repos/unet-pp/data/tuner/"

NOW_STR = datetime.now().strftime("%y%m%d_%H%M%S")

INPUT_SHAPE = (256, 512, 3)
EPOCHS = 200

# Cityscapes classes for segmentation
# background, person, rider, car, truck, bus, caravan, trailer, train, motorcycle, bicycle
COLORMAPS = [0, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

# Get image/mask paths
paths = get_image_mask_pair_paths(PurePath(DATASET_FOLDER_PATH))

# Split the paths into training, validation and test sets
train_paths, val_paths, test_paths = train_test_val_split(
    paths, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)

# Create generators
train_generator = SegmentationGenerator(
    train_paths,
    colormap=COLORMAPS,
    target_size=INPUT_SHAPE[:2],
    batch_size=2,
    shuffle=True,
)

validation_generator = SegmentationGenerator(
    val_paths,
    colormap=COLORMAPS,
    target_size=INPUT_SHAPE[:2],
    batch_size=2,
    shuffle=True,
)


# Get class weights
print("Computing class weights...")
class_weights = get_class_weights([p.mask_path for p in train_paths], COLORMAPS)
print(class_weights)
class_weights = list(class_weights.values())


# Define the model building function
def model_builder(hp):

    # Optimizers lookup
    OPTIMIZER_LKP = {
        "adam": Adam,
        "sgd": SGD,
        "adamax": Adamax,
    }

    # Loss functions lookup
    LOSS_FUNCTION_LKP = {
        "categorical_crossentropy": CategoricalCrossentropy,
        "categorical_focal_crossentropy": CategoricalFocalCrossentropy,
    }

    # Model params
    deep_supervision = hp.Boolean("deep_supervision", default=False)
    batch_normalization = hp.Boolean("batch_normalization", default=False)
    activation = hp.Choice("activation", values=["relu", "leaky_relu", "elu"])
    optimizer = hp.Choice("optimizer", values=list(OPTIMIZER_LKP.keys()))
    learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])


    # Dropout params
    dropout = hp.Boolean("dropout", default=False)
    with hp.conditional_scope("dropout", [True]):
        if dropout:
            dropout_rate = hp.Choice("dropout_rate", values=[0.1, 0.2, 0.5])
            dropout_type = hp.Choice("dropout_type", values=["simple", "spatial"])

    with hp.conditional_scope("dropout", [False]):
        if not dropout:
            dropout_type = None
            dropout_rate = 0.0


    # Loss function params
    loss_function_choice = hp.Choice("loss_function_choice", values=list(LOSS_FUNCTION_LKP.keys()))
    with hp.conditional_scope("loss_function_choice", ["categorical_crossentropy"]):
        if loss_function_choice == "categorical_crossentropy":
            loss_function = LOSS_FUNCTION_LKP[loss_function_choice]()

    with hp.conditional_scope("loss_function_choice", ["categorical_focal_crossentropy"]):
        if loss_function_choice == "categorical_focal_crossentropy":
            gamma = hp.Choice("gamma", values=[0.5, 1.0, 2.0])
            loss_function = LOSS_FUNCTION_LKP[loss_function_choice](alpha=class_weights, gamma=gamma)


    # Build the model
    model = UNetPlusPlus(
        INPUT_SHAPE,
        len(COLORMAPS),
        deep_supervision=deep_supervision,
        batch_normalization=batch_normalization,
        conv_activation=activation,
        dropout=dropout,
        dropout_type=dropout_type,
        dropout_rate=dropout_rate,
    ).model

    # Compile the model
    model.compile(
        optimizer=OPTIMIZER_LKP[optimizer](learning_rate),
        loss=loss_function,
        metrics=[dice_coefficient],
    )

    return model


# Instantiate the tuner
tuner = kt.Hyperband(
    model_builder,
    objective=kt.Objective("val_dice_coefficient", 'max'), # When using models with different loss functions, use the same metric to compare/maximize them
    max_epochs=EPOCHS,
    hyperband_iterations=1,
    directory=str(PurePath(TUNER_PROJECT_FOLDER)),
    project_name="unetpp_tuning"
)

early_stopping_callback = EarlyStopping(
    monitor="val_dice_coefficient", patience=6, mode="max", verbose=1
)

# Search for the best hyperparameters
tuner.search(
    train_generator,
    steps_per_epoch=train_generator.steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_generator.steps_per_epoch,
    callbacks=[early_stopping_callback],
)

print("TUNING DONE")
tuner.results_summary()

"""
top_100_params = tuner.get_best_hyperparameters(100)
distinct_top_3_models = []

param_keys = ["deep_supervision", "batch_normalization", "activation", "optimizer", "learning_rate", "loss_function"]
for i, hp in enumerate(top_100_params):


    values = [hp.values[key] for key in param_keys]
    if values not in distinct_top_3_models:
        
        print(f"Model Hyperparams {i+1}:")
        print(hp.values)
        print("\n")
        distinct_top_3_models.append(values)



        model = tuner.hypermodel.build(hp)
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            steps_per_epoch=train_generator.steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_generator.steps_per_epoch,
            callbacks=[early_stopping_callback],
        )

        val_loss_min = min(history.history["val_loss"])
        best_epoch = history.history["val_loss"].index(val_loss_min) + 1

        hypermodel = tuner.hypermodel.build(hp)
        hypermodel.fit(
            train_generator,
            epochs=best_epoch,
            steps_per_epoch=train_generator.steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_generator.steps_per_epoch,
            callbacks=[TensorBoard(log_dir=f"logs/hparam_tuning/hypermodel_fit/{NOW_STR}")]
        )

        # Save the model
        model_filename = str(
            PurePath(MODEL_FOLDER_PATH) / f"unetpp_{NOW_STR}" / f"model_{i+1}_{NOW_STR}.h5"
        )

        hypermodel.save(model_filename)



    if len(distinct_top_3_models) == 3:
        break
"""