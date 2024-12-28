import os
import shutil
import subprocess
# import gdown
from config import (
    current_directory,

)

def setup_model():
    create_directory('models')
    model_path = 'models/train_with_nms_timelimit_custom_best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: YOLOv11 model not found at {model_path}")
        print("Please ensure your trained YOLOv11 model is in the models directory")
        exit(1)
    else:
        print(f"YOLOv11 model found at {model_path}")

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
        print("Failed to install requirements:")
        print(e)
        exit(1)

def setup_environment():
    """Setup the complete environment for ANPR system"""
    print("Starting ANPR environment setup...")
    
    # Set current directory
    os.chdir(current_directory)
    
    # Create required directories
    create_directory('result')
    create_directory('models')
    
    # Setup YOLOv11 model
    setup_model()
    
    # Install Python requirements
    install_requirements()
    
    print("Setup completed successfully.")

if __name__ == "__main__":
    setup_environment()
