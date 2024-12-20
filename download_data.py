import urllib.request
import os
import requests
import zipfile
from bs4 import BeautifulSoup

def download_data():
    url = "https://www.nyc.gov/site/planning/data-maps/open-data/dwn-nyc-3d-model-download.page"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    download_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".zip"):
            download_links.append("https://www.nyc.gov" + href)

    # Print the download links
    for link in download_links:
        file_name = link.split("/")[-1]
        file_path = os.path.join("data", "NYC", file_name)
        
        # Check if the file already exists
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(link, file_path)
            print(f"Downloaded {file_name}")
        else:
            print(f"File {file_name} already exists, skipping download")

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

unzip_files()