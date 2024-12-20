import random
import shutil
import urllib.request
import os
import requests
import zipfile
from bs4 import BeautifulSoup
import process_3dm_file
import memory_profiler
import gc


# @memory_profiler.profile
def download_data():
    url = "https://www.nyc.gov/site/planning/data-maps/open-data/dwn-nyc-3d-model-download.page"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    download_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".zip"):
            download_links.append("https://www.nyc.gov" + href)

    category_count = {}

    # Print the download links
    for link in download_links:
        file_name = link.split("/")[-1]
        category = file_name[12:14]
        file_path = os.path.join("data", "NYC", file_name)
        folder_path = os.path.splitext(file_path)[0]
        
        # Initialize category count if not already present
        if category not in category_count:
            category_count[category] = 0
        
        # Check if the file or folder already exists and if the category count is less than 3
        if category_count[category] < 100 and not os.path.exists(file_path) and not os.path.exists(folder_path):
            urllib.request.urlretrieve(link, file_path)
            print(f"Downloaded {file_name}")
            category_count[category] += 1
            break
        else:
            print(f"File {file_name} already exists or category limit reached, skipping download")

# @memory_profiler.profile
def unzip_files():
    folder_path = "data/NYC"
    
    # Iterate over each zip file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.zip'):
            zip_path = os.path.join(folder_path, file_name)
            
            # Extract the zip file to a temporary folder
            temp_folder = os.path.splitext(zip_path)[0]
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_folder)
            print(f"Extracted {file_name}")
            
            # Delete the zip file after extraction
            os.remove(zip_path)
            print(f"Deleted {file_name}")

            # reorganize_files()
            # print("yay it tried to reorganize")

            break

def reorganize_files():
    folder_path = "data/NYC"
    
    # Iterate over each folder in the directory
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        
        # Check if the item is a folder
        if os.path.isdir(folder_dir):
            # Get the r3d file inside the folder
            for file_name in os.listdir(folder_dir):
                if file_name.endswith('.3dm'):
                    r3d_file_path = os.path.join(folder_dir, file_name)
                    
                    # Create the destination folder based on its category
                    destination_folder = os.path.join(folder_path, file_name[12:14])
                    os.makedirs(destination_folder, exist_ok=True)
                    
                    # Randomly choose between 'test' and 'training'
                    subfolder = random.choice(['test', 'train'])
                    subfolder_path = os.path.join(destination_folder, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)
                    
                    # Move the r3d file to the subfolder
                    destination_file_path = os.path.join(subfolder_path, file_name)
                    shutil.move(r3d_file_path, destination_file_path)
                    
                    # Delete the existing folder
                    shutil.rmtree(folder_dir)
                    print(f"Moved {file_name} to {subfolder_path} and deleted {folder_name}")
                    break  # Only process one .3dm file per folder



if __name__ == "__main__":
    for i in range(5):
        download_data()
        # Unzip the file after downloading
        unzip_files()
        process_3dm_file.process_all_models()
# unzip_files()

# reorganize_files()