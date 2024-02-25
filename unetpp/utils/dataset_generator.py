import os 
import shutil
import zipfile
import json
import cv2
import numpy as np
import random





class DatasetGenerator:
    def __init__(self, zip_dataset_path, input_zip_file, input_path, input_class_path, train_folder_path, val_folder_path, test_folder_path, class_to_kepp, train_size,val_size, test_size):
        self.zip_dataset_path = zip_dataset_path
        self.input_zip_file = input_zip_file
        self.input_path = input_path
        self.input_class_path = input_class_path
        self.train_folder_path = train_folder_path
        self.val_folder_path = val_folder_path
        self.test_folder_path = test_folder_path
        self.class_to_keep = class_to_kepp
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    @property
    def clsss_to_keep(self) -> list:
        return self.class_to_keep
    
    @clsss_to_keep.setter
    def clsss_to_keep(self, value: list) -> None:
        if not isinstance(value, list):
            raise ValueError("Classes to keep must be a list.")
        self._class_to_keep = value
    
    @property
    def train_size(self) -> float:
        return self.train_size
    
    @train_size.setter
    def train_size(self, value: float) -> None:
        if not isinstance(value, float) or value <= 0 or value > 1:
            raise ValueError("Train size must be a float between 0 and 1.")
        self._train_size = value
    
    @property
    def val_size(self) -> float:
        return self.val_size
    
    @val_size.setter
    def val_size(self, value: float) -> None:
        if not isinstance(value, float) or value <= 0 or value > 1:
            raise ValueError("Val size must be a float between 0 and 1.")
        self._val_size = value
    
    @property
    def test_size(self) -> float:
        return self.test_size
    
    @test_size.setter
    def test_size(self, value: float) -> None:
        if not isinstance(value, float) or value <= 0 or value > 1:
            raise ValueError("Test size must be a float between 0 and 1.")
        self._test_size = value


    def create_directories(self):
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.input_zip_file, exist_ok=True)
        os.makedirs(self.input_class_path, exist_ok=True)
        os.makedirs(self.train_folder_path, exist_ok=True)
        os.makedirs(self.val_folder_path, exist_ok=True)
        os.makedirs(self.test_folder_path, exist_ok=True)

    def open_zip_file(self):
        # Open the zip file
        with zipfile.ZipFile(self.zip_dataset_path, 'r') as zip_ref:
            # Extract all contents to the specified directory
            zip_ref.extractall(self.input_zip_file)


    def copy_images(self):
        for root, dirs, files in os.walk(self.input_zip_file):
            #skip test folder because it does not have corect mask
            if 'test' in root:
                continue
            else:
                for img in files:
                    #choose images
                    if img.endswith('leftImg8bit.png'):
                        source_file_path = os.path.join(root, img)
                        parts = img.split('_')
                        new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_input.png"
                        destination_file_path = os.path.join(self.input_path, new_filename)
                        shutil.copy(source_file_path, destination_file_path)
                    #choose masks
                    elif img.endswith('gtFine_labelIds.png'):
                        source_file_path = os.path.join(root, img)
                        parts = img.split('_')
                        new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_mask.png"
                        destination_file_path = os.path.join(self.input_path, new_filename)
                        shutil.copy(source_file_path, destination_file_path)
                    #choose json files
                    elif img.endswith('gtFine_polygons.json'):
                        source_file_path = os.path.join(root, img)
                        parts = img.split('_')
                        new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}.json"
                        destination_file_path = os.path.join(self.input_path, new_filename)
                        shutil.copy(source_file_path, destination_file_path)

    def convert_mask_with_class(self):
        # Iterate through JSON files
        for json_file in os.listdir(self.input_path):
            if json_file.endswith('.json'):
                with open(os.path.join(self.input_path, json_file), 'r') as f:
                    data = json.load(f)
                
                # Check if the class we want exists in the JSON data
                if any(obj['label'] in self.class_to_keep for obj in data['objects']):
                    mask_file_name = json_file.replace('.json', '_mask.png')
                    input_file_name = json_file.replace('.json', '_input.png')

                    mask_file = os.path.join(self.input_path, mask_file_name)
                    input_file = os.path.join(self.input_path, input_file_name)
                    
                    # Load mask image
                    mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    output_mask = np.zeros_like(mask_image)
                    
                    # Iterate through polygons
                    for obj in data['objects']:
                        # Check if the polygon label is the class we want to keep
                        if obj['label'] in self.class_to_keep:
                            polygon = np.array(obj['polygon'])
                            cv2.fillPoly(output_mask, [polygon], 255) 
                    
                    # Clean the original mask image by keeping only regions of interest
                    cleaned_mask = cv2.bitwise_and(mask_image, output_mask)

                    # Save processed mask image
                    output_file = os.path.join(self.input_class_path, json_file.replace('.json', '_mask.png'))
                    cv2.imwrite(output_file, cleaned_mask)

                    # Copy _input.png file if exists
                    if os.path.exists(input_file):
                        shutil.copy(input_file, self.input_class_path)
        
    def split_train_val_test(self):
        #TODO choose all clases folder or only keeped classes folder
        files = os.listdir(self.input_class_path)
        random.shuffle(files)
        # Initialize an empty list to store the pairs
        pairs = []
        # Loop through the list of files
        for file in files:
            # Check if the file ends with '_input.png'
            if file.endswith('_input.png'):
                # Construct the corresponding '_mask.png' filename
                mask_file = file.replace('_input.png', '_mask.png')
                # Check if this '_mask.png' file exists in the directory
                if mask_file in files:
                    # If it does, add the pair to the list
                    pairs.append((file, mask_file))

        train_size = int(self.train_size * len(pairs))
        val_size = int(self.val_size * len(pairs))
        test_size = len(pairs) - train_size - val_size

        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]


        train_path_input = os.path.join(self.train_folder_path, 'input')
        train_path_masks = os.path.join(self.train_folder_path, 'masks')
        os.makedirs(train_path_input, exist_ok=True)
        os.makedirs(train_path_masks, exist_ok=True)

        val_path_input = os.path.join(self.val_folder_path, 'input')
        val_path_masks = os.path.join(self.val_folder_path, 'masks')
        os.makedirs(val_path_input, exist_ok=True)
        os.makedirs(val_path_masks, exist_ok=True)

        test_path_input = os.path.join(self.test_folder_path, 'input')
        test_path_masks = os.path.join(self.test_folder_path, 'masks')
        os.makedirs(test_path_input, exist_ok=True)
        os.makedirs(test_path_masks, exist_ok=True)

        # Copy the files to the appropriate directories
        for pair in train_pairs:
            shutil.copy(os.path.join(self.input_class_path, pair[0]), train_path_input)
            shutil.copy(os.path.join(self.input_class_path, pair[1]), train_path_masks)

        for pair in val_pairs:
            shutil.copy(os.path.join(self.input_class_path, pair[0]), val_path_input)
            shutil.copy(os.path.join(self.input_class_path, pair[1]), val_path_masks)

        for pair in test_pairs:
            shutil.copy(os.path.join(self.input_class_path, pair[0]), test_path_input)
            shutil.copy(os.path.join(self.input_class_path, pair[1]), test_path_masks)

