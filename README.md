# unet-pp

A simple UNet++ implementation (with Python and Tensorflow) created as the final project for our postgraduate studies, "Artificial intelligence and automation of business processes in practical terms" at the Gdańsk University of Technology (Polish: "Sztuczna inteligencja i automatyzacja procesów biznesowych w ujęciu praktycznym"). This project provides an educational framework for demonstrating the capabilities and applications of UNet++ in image segmentation tasks.


## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
  - [Using the Cityscapes Dataset](#using-the-cityscapes-dataset)
  - [Preparing a Custom Dataset](#preparing-a-custom-dataset)
- [Dataset Analysis](#dataset-analysis)
- [Model Configuration](#model-configuration)
  - [Challenges](#challenges)
  - [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [References](#references)



## Getting Started

### Prerequisites
- A Linux / WSL2 machine with Tensorflow/Keras and CUDA installed
  - Follow: https://www.tensorflow.org/install?hl=en
  - **Caution**: TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows. Starting with TensorFlow 2.11, you will need to install TensorFlow in WSL2, or install tensorflow or tensorflow-cpu and, optionally, try the TensorFlow-DirectML-Plugin


### Installation
**TODO**: A step-by-step guide to get the project up and running on your local machine.

(Use the provided conda yml to install the necessary venv/env)


## Dataset Preparation

To test our implementation of UNet++ we decided to use the [Cityscapes](https://www.cityscapes-dataset.com/) dataset that focuses on semantic understanding of urban street scenes.

### Using the Cityscapes Dataset
  - Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainval.zip` from the Cityscapes dataset.
  - Use the provided CLI tool designed for preparing the Cityscapes dataset to meet the specified criteria


#### CLI Tool

Params:
  - `--dataset`: Specifies the dataset to process (currently supports "cityscapes" only).
  - `--input_zip_path`: The path to the ZIP file containing the input data.
  - `--mask_zip_path`: The path to the ZIP file containing the mask data.
  - `--output_path`: The path where the processed/prepared dataset will be saved.
  - `--class_to_keep`: (Optional) A string of classes to retain. If specified, the tool will exclude any inputs/masks where none of the specified classes are present, and it will create grayscale masks exclusively for the specified classes (setting all other classes to 0 or black).

Usage: 

  - Execute the following command to process the Cityscapes dataset with all classes:
    ```shell
    python preprocessing_data.py --dataset "cityscapes" --input_zip_path "./path/to/leftImg8bit_trainvaltest.zip" --mask_zip_path "./path/to/gtFine_trainval.zip" --output_path "/path/to/output"
    ```

  - Execute the following command to process the Cityscapes dataset while only retaining the car and person classes:
      ```shell
    python preprocessing_data.py --dataset "cityscapes" --input_zip_path "./path/to/leftImg8bit_trainvaltest.zip" --mask_zip_path "./path/to/gtFine_trainval.zip" --output_path "/path/to/output" car person
    ```

*Additional Notes*
- The `class_to_keep` parameter allows for selective data processing, which can be particularly useful for focused model training or testing.
* Skip the `test` folder from the Cityscapes dataset since it is used for testing on their evaluation server. The annotations are not public!


### Preparing a Custom Dataset
You may use any dataset of your choice, provided it includes a single folder containing both:
  - RGB input images
  - Grayscale image masks


## Dataset Analysis

To check the class distribution in mask images you can use the provided CLI tool
### CLI Tool

Params:
- `--input_path`: The path to the file containing the input data.
- `--target_mask_suffix`: (Optional) Specifies the suffix to be appended to the target mask file names if there are different files in the input path 

Usage:
  - Execute the following command to check the classes distribution if there are only the masks you are interested in in the input path
    ```shell
    python classes_distribution.py --input_path "./path/to/masks"
    ```
  - Execute the following command to check the classes distribution if your masks have a suffix in the input path
    ```shell
    python classes_distribution.py --input_path "./path/to/masks" --target_mask_suffix suffix_name
    ```




## Model Configuration
**Details**
- **Classes**: 10
- **Image Size**: 512x256 (downscaled from the original size of 2048x1024)
- **Batch Size**: 4 (606 batches per epoch)
- **Loss Function**: Categorical Cross Entropy
- **Hardware**: GeForce RTX 4080 Super
- **Training Time**: 78 minutes
- **Hyperparameter Tuning**: Conducted with Keras Tuner over ~7 days, considering various combinations including deep supervision, batch normalization, activation functions, optimizers, learning rates, and dropout (see script in examples folder)

**Best Model Configuration**
- **Deep Supervision**: False
- **Batch Normalization**: True
- **Activation**: ReLU
- **Optimizer**: Adam with a learning rate of 0.0001
- **Dropout**: False
- **Results**: val_loss: *0.0714*, test_loss: *0.0662*

See some example results in the provided **ipnyb** file.



### Challenges

- Encountered issues with GPU memory limits, addressed by reducing number of classes, image and batch size
- Experienced issues with the Dying ReLU problem

### Future Improvements
  - Implementing image augmentation e.g. https://github.com/albumentations-team/albumentations
  - Training on small image patches (patch-based image segmentation pipeline)
  - Experimenting with larger batch sizes (8 – 32)
  - Exploring additional loss functions and metrics, such as Focal loss
  - Adjusting the number of levels in the model (default 4) and the initial number of filters (default 64)

## Acknowledgments
Special thanks to our mentor, [Michał Maj](https://github.com/maju116), who guided us through the completion of this project. His expertise and insights were invaluable!


## References
- [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://www.cityscapes-dataset.com/wordpress/wp-content/papercite-data/pdf/cordts2016cityscapes.pdf)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf)












