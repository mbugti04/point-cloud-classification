# This file downloads and processes the 3D model data from the NYC official website.
# It is not recommended to use this file, as the library to read the .3dm file seems to have a memory leak
# Also, the dataset takes up around 50gb.
# Thus, it is recommended to use the given, already converted data

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


# Download data from official website
# Step 1 in processing.
def download_from_link():
    url = "https://www.nyc.gov/site/planning/data-maps/open-data/dwn-nyc-3d-model-download.page"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    download_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".zip"):
            download_links.append("https://www.nyc.gov" + href)

    # Checks for all links
    for link in download_links:
        file_name = link.split("/")[-1]
        file_path = os.path.join("data", "NYC", file_name)
        folder_path = os.path.splitext(file_path)[0]
        
        # Check if the file or folder already exists
        if not os.path.exists(file_path) and not os.path.exists(folder_path):
            urllib.request.urlretrieve(link, file_path)
            print(f"Downloaded {file_name}")

            # Downloads one at a time so that the file may be processed first.
            # This is to save space.
            return True
        # else:
        #     print(f"File {file_name} already exists, skipping download")

    # Everything is already downloaded
    print("All files downloaded.")
    return False

# Step 2 in processing.
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

            # Extracts one at a time so that the file may be processed first.
            break

@memory_profiler.profile
def download_data():
    for i in range(2):
        all_files_processed = download_from_link()
        unzip_files()
        process_3dm_file.process_all_models()

        if all_files_processed:
            print("Done!")
            break

@memory_profiler.profile
def debug_read():
    model = process_3dm_file.load_model(r"C:\Users\musta\Downloads\nyc_3dmodel_bx02\NYC_3DModel_BX02.3dm")
    for i in range(5):
        vertices = process_3dm_file.process_3dm(model)

        # Explicitly delete attributes of model if it has any
        # if hasattr(model, '__dict__'):
        #     for attr in list(model.__dict__.keys()):
        #         delattr(model, attr)

        # del model
        # gc.collect()

        # Explicitly delete attributes of vertices if it has any
        if hasattr(vertices, '__dict__'):
            for attr in list(vertices.__dict__.keys()):
                delattr(vertices, attr)

        del vertices
        gc.collect()

if __name__ == "__main__":
    # download_data()
    debug_read()