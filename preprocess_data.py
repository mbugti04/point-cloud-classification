import os
import zipfile
import shutil


def unzip_files(folder_path):
    # Iterate over each zip file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.zip'):
            zip_path = os.path.join(folder_path, file_name)
            
            # Extract the zip file to a temporary folder
            temp_folder = os.path.splitext(zip_path)[0]
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_folder)

def return_obj_file_paths(folder_path):
    # Iterate over each folder in the directory
    for folder_name in os.listdir(folder_path):
        folder = os.path.join(folder_path, folder_name)
        
        # Iterate over each file in the folder
        for file_name in os.listdir(folder):
            if file_name.endswith('.obj'):
                file_path = os.path.join(folder, file_name)
                yield file_path

# Path to the folder containing the zip files
folder_path = r'data\Boston\BOS_H_4_BldgModels_OBJ\objz'

combined_obj = ''