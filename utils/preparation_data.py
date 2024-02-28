import shutil
import zipfile
import typer
import os
from typing import Tuple, List
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Optional
from typing_extensions import Annotated


app = typer.Typer()


@dataclass
class CityScapesClass:
    name: str
    id: int
    rgb_mask: Tuple[int, int, int]


CityScapesClasses = [
    CityScapesClass("unlabeled", 0, (0, 0, 0)),
    CityScapesClass("ego vehicle", 1, (0, 0, 0)),
    CityScapesClass("rectification border", 2, (0, 0, 0)),
    CityScapesClass("out of roi", 3, (0, 0, 0)),
    CityScapesClass("static", 4, (0, 0, 0)),
    CityScapesClass("dynamic", 5, (111, 74, 0)),
    CityScapesClass("ground", 6, (81, 0, 81)),
    CityScapesClass("road", 7, (128, 64, 128)),
    CityScapesClass("sidewalk", 8, (244, 35, 232)),
    CityScapesClass("parking", 9, (250, 170, 160)),
    CityScapesClass("rail track", 10, (230, 150, 140)),
    CityScapesClass("building", 11, (70, 70, 70)),
    CityScapesClass("wall", 12, (102, 102, 156)),
    CityScapesClass("fence", 13, (190, 153, 153)),
    CityScapesClass("guard rail", 14, (180, 165, 180)),
    CityScapesClass("bridge", 15, (150, 100, 100)),
    CityScapesClass("tunnel", 16, (150, 120, 90)),
    CityScapesClass("pole", 17, (153, 153, 153)),
    CityScapesClass("polegroup", 18, (153, 153, 153)),
    CityScapesClass("traffic light", 19, (250, 170, 30)),
    CityScapesClass("traffic sign", 20, (220, 220, 0)),
    CityScapesClass("vegetation", 21, (107, 142, 35)),
    CityScapesClass("terrain", 22, (152, 251, 152)),
    CityScapesClass("sky", 23, (70, 130, 180)),
    CityScapesClass("person", 24, (220, 20, 60)),
    CityScapesClass("rider", 25, (225, 0, 0)),
    CityScapesClass("car", 26, (0, 0, 142)),
    CityScapesClass("truck", 27, (0, 0, 70)),
    CityScapesClass("bus", 28, (0, 60, 100)),
    CityScapesClass("caravan", 29, (0, 0, 90)),
    CityScapesClass("trailer", 30, (0, 0, 110)),
    CityScapesClass("train", 31, (0, 80, 100)),
    CityScapesClass("motorcycle", 32, (0, 0, 230)),
    CityScapesClass("bicycle", 33, (119, 11, 32)),
    CityScapesClass("license plate", -1, (0, 0, 142)),
]


def open_zip_files(zip_dataset_paths: list[str], output_path: str) -> None:
    """
    Extract contents of multiple zip files to a directory.
    """
    for zip_dataset_path in zip_dataset_paths:
        with zipfile.ZipFile(zip_dataset_path, "r") as zip_ref:
            zip_ref.extractall(output_path + "\\temp\\")


def copy_images(output_path: str) -> None:
    """
    Copy images and masks from input_zip_file to input_path.
    """
    TEMP_PATH = output_path + "\\temp\\"
    IMG_SUFFIX = "_leftImg8bit.png"
    MASK_SUFFIX = "_gtFine_labelIds.png"

    SUFFIX_REPLACEMENT = {
        IMG_SUFFIX: "_input.png",
        MASK_SUFFIX: "_mask.png",
    }

    for root, _, files in os.walk(TEMP_PATH):
        # Skip test folder- test folder
        if "test" not in root:
            for file in files:
                # Iterate over all images/masks, rewrite their suffix and move them
                for suffix, replacement in SUFFIX_REPLACEMENT.items():
                    if file.endswith(suffix):
                        new_filename = file.replace(suffix, replacement)
                        original_path = os.path.join(root, file)
                        destination_path = os.path.join(output_path, new_filename)
                        shutil.move(original_path, destination_path)
                        break

    # Remove the empty directories
    shutil.rmtree(TEMP_PATH)


def create_masks_by_class(output_path: str, class_to_keep: list[str] = []):

    MASK_SUFFIX = "_mask.png"
    IMG_SUFFIX = "_input.png"

    for image_file in os.listdir(output_path):
        if image_file.endswith(MASK_SUFFIX):
            image_path = os.path.join(output_path, image_file)
            # Read the image using cv2.imread
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if class_to_keep:
                # Get the labels of the classes to keep
                labels = [
                    cls.id for cls in CityScapesClasses if cls.name in class_to_keep
                ]
                mask = np.isin(image, labels)
                mask_value = 0
                cleaned_mask = np.where(mask, image, mask_value)
                mask_file_path = os.path.join(output_path, image_file)
            else:
                mask_file_path = os.path.join(output_path, image_file)
                cleaned_mask = image

            input_file = image_file.replace(MASK_SUFFIX, IMG_SUFFIX)
            input_file_path = os.path.join(output_path, input_file)
            if np.all(cleaned_mask == mask_value):
                # Delete input and mask file
                if os.path.exists(input_file_path):
                    os.remove(input_file_path)
                if os.path.exists(mask_file_path):
                    os.remove(mask_file_path)
            else:
                if os.path.exists(mask_file_path):
                    # Delete old mask file
                    os.remove(mask_file_path)
                # Save new mask
                cv2.imwrite(mask_file_path, cleaned_mask)


@app.command()
def process_data(
    dataset: str,
    input_zip_path: str,
    mask_zip_path: str,
    output_path: str,
    class_to_keep: Annotated[Optional[List[str]], typer.Argument()] = None,
):
    # TODO: Add support for other datasets
    # TODO: Validate input paths and class_to_keep
    if dataset in ["cityscapes"]:
        """
        Process zip dataset to extract images and masks.
        """
        typer.echo("Extracting contents of the zip file")
        open_zip_files([input_zip_path, mask_zip_path], output_path)

        typer.echo("Copying images and masks")
        copy_images(output_path)

        if class_to_keep:
            typer.echo(f"Creating masks for classes: {class_to_keep}")
            create_masks_by_class(output_path, class_to_keep)

        typer.echo("Data processing complete.")

    else:
        typer.echo("Dataset not supported.")


if __name__ == "__main__":
    app()
