from model.unetpp import UNetPlusPlus
from utils.dataset_generator import DatasetGenerator

# Build the model and print the summary
input_shape = (256, 256, 3)
num_classes = 1
model = UNetPlusPlus(input_shape, num_classes).model
model.summary()




zip_dataset_path_masks = r'unetpp\data\gtFine_trainvaltest.zip'
zip_dataset_path_img = r'unetpp\data\leftImg8bit_trainvaltest.zip'
input_zip_file =r'unetpp\data\data_zip'
input_path = r'unetpp\data\data_input'
input_class_path = r'unetpp\data\data_input_class'
class_to_kepp = ['person']
train_folder_path = r'unetpp\data\train'
val_folder_path = r'unetpp\data\val'
test_folder_path = r'unetpp\data\test'
train_size = 0.7
val_size = 0.15
test_size = 0.15

#create directories if not exist
DatasetGenerator(zip_dataset_path_masks, input_zip_file, input_path, input_class_path, train_folder_path, val_folder_path, test_folder_path, class_to_kepp, train_size,val_size, test_size).create_directories()
#open zip file
DatasetGenerator(zip_dataset_path_masks, input_zip_file, input_path, input_class_path, train_folder_path, val_folder_path, test_folder_path, class_to_kepp, train_size,val_size, test_size).open_zip_file()
DatasetGenerator(zip_dataset_path_masks, input_zip_file, input_path, input_class_path, train_folder_path, val_folder_path, test_folder_path, class_to_kepp, train_size,val_size, test_size).open_zip_file()
#copy images to the input folder (mask, json, img)
DatasetGenerator(zip_dataset_path_masks, input_zip_file, input_path, input_class_path, train_folder_path, val_folder_path, test_folder_path, class_to_kepp, train_size,val_size, test_size).copy_images()
#create mask with class 
DatasetGenerator(zip_dataset_path_masks, input_zip_file, input_path, input_class_path, train_folder_path, val_folder_path, test_folder_path, class_to_kepp, train_size,val_size, test_size).convert_mask_with_class()
#split data to train, val, test
DatasetGenerator(zip_dataset_path_masks, input_zip_file, input_path, input_class_path, train_folder_path, val_folder_path, test_folder_path, class_to_kepp, train_size,val_size, test_size).split_train_val_test()
