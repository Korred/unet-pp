# unet-pp
A simple UNet++ implementation for educational purposes

## Preparation data

The process_data function is designed to process a "cityscapes" dataset. Here's a step-by-step description of what this function does:

The function accepts five parameters:

dataset: The name of the dataset to process.
input_zip_path: The path to the zip file containing the input data.
mask_zip_path: The path to the zip file containing the mask data.
output_path: The path where the processed data should be saved.
class_to_keep: An optional list of classes to keep. If specified, masks will be created only for these classes.
The function first checks if the provided dataset is "cityscapes". If it's not, it prints "Dataset not supported" and exits.

Run the file
python preprocessing_data.py --dataset "cityscapes" --input_zip_path "/path/to/input.zip" --mask_zip_path "/path/to/mask.zip" --output_path "/path/to/output" --class_to_keep "class1 class2 class3"

Please replace /path/to/your/script, /path/to/input.zip, /path/to/mask.zip, and /path/to/output with the actual paths on your system. Replace "class1 class2 class3" with the actual classes you want to keep.