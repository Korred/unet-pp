import os
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from collections import defaultdict
from typing import Dict
import typer

# We need to add the unetpp package to the system path so that we can import it
import sys
from pathlib import PurePath

unetpp_path = PurePath(__file__).parent.parent
sys.path.append(str(unetpp_path))

from utils.preparation_data import CityScapesClasses

TARGET_MASK_SUFFIX = "_mask.png"
app = typer.Typer()


def get_class_distribution(input_path: str, classes: Dict[int, str]) -> Table:
    """
    Get the class distribution of the masks in the input_path
    """
    class_distribution_pixelwise = defaultdict(int)
    images_per_class = defaultdict(int)
    all_pixels = 0
    total_images = 0

    for image_file in os.listdir(input_path):
        if image_file.endswith(TARGET_MASK_SUFFIX):
            total_images += 1
            image_path = os.path.join(input_path, image_file)
            # Read the image using cv2.imread
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Count all pixels
            all_pixels += image.size
            # Get the unique classes and their counts
            unique, counts = np.unique(image, return_counts=True)
            # Add the counts to the dictionary
            for cls, count in zip(unique, counts):
                class_distribution_pixelwise[cls] += count
                images_per_class[cls] += 1

    # Create class distribution table
    class_distribution_table = Table(title="Class Distribution")
    class_distribution_table.add_column("Class ID", justify="center")
    class_distribution_table.add_column("Class Name", justify="center")
    class_distribution_table.add_column("Percentage (Pixels)", justify="center")
    class_distribution_table.add_column("Images Count", justify="center")
    class_distribution_table.add_column("Percentage (Images)", justify="center")

    sorted_data = sorted(
        class_distribution_pixelwise.items(), key=lambda x: x[1], reverse=True
    )

    for cls, count in sorted_data:
        pixel_percentage = count / all_pixels
        image_percentage = images_per_class[cls] / total_images
        class_name = classes.get(cls, "Unknown")
        class_distribution_table.add_row(
            str(cls),
            class_name,
            f"{pixel_percentage:.2%}",
            str(images_per_class[cls]),
            f"{image_percentage:.2%}",
        )

    return class_distribution_table


@app.command()
def calculate_class_distribution(input_path: str):
    """
    Calculate class distribution from masks in the input_path.
    """
    console = Console()
    typer.echo(f"Calculating class distribution for masks in {input_path}")
    class_distribution_table = get_class_distribution(
        input_path, {cls.id: cls.name for cls in CityScapesClasses}
    )

    console.print("Class Distribution:")
    console.print(class_distribution_table)


if __name__ == "__main__":
    app()
