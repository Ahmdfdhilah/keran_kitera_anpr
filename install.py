import os
import subprocess
import gdown
from config import (
    current_directory,
    model_file_id,
    model_file_name,
    weights_file_name,
    weights_file_id
)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

def install_requirements():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to install requirements.")
        print(e)

def download_file_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

if __name__ == "__main__":
    os.chdir(current_directory)

    # 1. Membuat direktori result
    create_directory('result')

    # 2. Membuat direktori model
    create_directory('model')

    # 3. Install requirements.txt
    install_requirements()

    # 4. Download file model dari Google Drive
    model_dest_path = os.path.join('model', model_file_name)
    download_file_from_gdrive(model_file_id, model_dest_path)

    # 5. Download file .weights dari Google Drive
    weights_dest_path = os.path.join('model', weights_file_name)
    download_file_from_gdrive(weights_file_id, weights_dest_path)

    print("Setup completed successfully.")
