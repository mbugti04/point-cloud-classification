import process_3dm_file
import download_data

def process_data():
    download_data.unzip_files()
    process_3dm_file.process_models()

if __name__ == "__main__":
    process_data()