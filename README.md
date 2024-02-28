# unet-pp
A simple UNet++ implementation for educational purposes

## Dataset Preparation

To facilitate testing our U-Net++ implementation, specific dataset requirements and tools have been established. Below are the guidelines and steps to prepare your dataset, focusing on using the Cityscapes dataset.

#### Dataset Requirements
- **Cityscapes Dataset:** We decided to use the [Cityscapes dataset](https://www.cityscapes-dataset.com/) for our implementation testing.
- **General Dataset Use:** You may use any dataset of your choice, provided it includes a single folder containing both:
  - Input images in RGB format.
  - Image masks in grayscale.

#### Cityscapes Dataset Preparation
For convenience, we added a CLI tool designed for preparing the Cityscapes dataset to meet the specified criteria.

- **Required Downloads:**
  - Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainval.zip` from the Cityscapes dataset.
- **Using the CLI Tool:**
  - Execute the following command to process the Cityscapes dataset:
```shell
python preprocessing_data.py --dataset "cityscapes" --input_zip_path "./path/to/leftImg8bit_trainvaltest.zip" --mask_zip_path "./path/to/gtFine_trainval.zip" --output_path "/path/to/output"
```


#### CLI Tool Parameters
- `--dataset`: Specifies the dataset to process (currently supports "cityscapes" only).
- `--input_zip_path`: The path to the ZIP file containing the input data.
- `--mask_zip_path`: The path to the ZIP file containing the mask data.
- `--output_path`: The path where the processed/prepared dataset will be saved.
- `--class_to_keep`: (Optional) A list of classes to retain. If specified, the tool will exclude any inputs/masks where none of the specified classes are present, and it will create grayscale masks exclusively for the specified classes (setting all other classes to 0 or black).

#### Additional Notes
- The `class_to_keep` parameter allows for selective data processing, which can be particularly useful for focused model training or testing.
* Skip the `test`  folder from dataset. Test folder: used for testing on evaluation server. The annotations are not public, but we include annotations of ego-vehicle and rectification border for convenience.

Please ensure to follow these guidelines closely to prepare your dataset correctly for optimal testing and evaluation of our U-Net++ implementation.