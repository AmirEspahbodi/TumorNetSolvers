# ./scripts/setup_dataset.py

import os
import requests
import zipfile
import logging
from tqdm import tqdm
import sys

# --- Configuration ---
# Use '/resolve/main/' for direct raw file downloads from Hugging Face
ZIP_URL = "https://huggingface.co/datasets/zeinebH/TumorSimulations/resolve/main/Dataset800_Brain.zip"
PTH_URL = "https://huggingface.co/datasets/zeinebH/TumorSimulations/resolve/main/param_dict.pth"

BASE_DATA_DIR_REL = "data_and_outputs"  # Relative to repo root
ZIP_EXTRACT_SUBDIR = "raw_data"
PTH_SAVE_SUBDIR = "preprocessed_data"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for visibility
)

def calculate_paths(script_path):
    """Calculates absolute paths based on the script's location."""
    script_dir = os.path.dirname(os.path.abspath(script_path))
    root_dir = os.path.dirname(script_dir) # Assumes script is in ./scripts
    base_data_path = os.path.join(root_dir, BASE_DATA_DIR_REL)

    paths = {
        "base_data": base_data_path,
        "zip_download": os.path.join(base_data_path, os.path.basename(ZIP_URL)),
        "zip_extract": os.path.join(base_data_path, ZIP_EXTRACT_SUBDIR),
        "pth_save_dir": os.path.join(base_data_path, PTH_SAVE_SUBDIR),
        "pth_save_path": os.path.join(base_data_path, PTH_SAVE_SUBDIR, os.path.basename(PTH_URL))
    }
    return paths

def download_file(url: str, save_path: str):
    """Downloads a file from a URL to a save path with a progress bar."""
    logging.info(f"Attempting to download: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        logging.info(f"Downloading to: {save_path}")
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

        # Final check to ensure progress bar completes if total_size was 0
        if total_size == 0 and bar.n > 0:
             bar.total = bar.n # Update total to actual downloaded size
             bar.refresh() # Refresh display

        logging.info(f"Successfully downloaded {os.path.basename(save_path)}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed for {url}: {e}")
        # Clean up potentially incomplete file
        if os.path.exists(save_path):
            os.remove(save_path)
            logging.info(f"Removed incomplete file: {save_path}")
        return False
    except IOError as e:
        logging.error(f"Failed to write file {save_path}: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str):
    """Extracts a zip file to a specified directory."""
    logging.info(f"Attempting to extract: {zip_path}")
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            logging.info(f"Extracting contents to: {extract_to}")
            zip_ref.extractall(extract_to)
        logging.info(f"Successfully extracted {os.path.basename(zip_path)}")
        return True
    except zipfile.BadZipFile:
        logging.error(f"Error: {zip_path} is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        logging.error(f"Failed to extract {zip_path}: {e}")
        return False

def setup_data():
    """Main function to download and set up dataset components."""
    paths = calculate_paths(__file__)

    # --- 1. Download and Extract ZIP ---
    logging.info("--- Starting Dataset ZIP Download and Extraction ---")
    if os.path.exists(paths["zip_extract"]) and os.listdir(paths["zip_extract"]):
         logging.warning(f"Extraction directory '{paths['zip_extract']}' already exists and is not empty. Skipping ZIP download and extraction.")
         zip_success = True # Assume success if already exists
    else:
        if download_file(ZIP_URL, paths["zip_download"]):
            if extract_zip(paths["zip_download"], paths["zip_extract"]):
                # Clean up downloaded zip file after successful extraction
                try:
                    os.remove(paths["zip_download"])
                    logging.info(f"Removed temporary zip file: {paths['zip_download']}")
                    zip_success = True
                except OSError as e:
                    logging.warning(f"Could not remove temporary zip file {paths['zip_download']}: {e}")
                    zip_success = True # Extraction succeeded, so treat overall as success
            else:
                 zip_success = False # Extraction failed
                 logging.error("Zip extraction failed. Aborting further steps dependent on it.")
        else:
             zip_success = False # Download failed
             logging.error("Zip download failed. Aborting further steps dependent on it.")

    if not zip_success:
        logging.error("Setup incomplete due to ZIP file issues.")
        return # Stop if zip part failed

    # --- 2. Download PTH file ---
    logging.info("--- Starting PTH File Download ---")
    # Ensure the specific subdirectory for the PTH file exists
    os.makedirs(paths["pth_save_dir"], exist_ok=True)

    if os.path.exists(paths["pth_save_path"]):
        logging.warning(f"File '{paths['pth_save_path']}' already exists. Skipping PTH download.")
        pth_success = True
    else:
        if download_file(PTH_URL, paths["pth_save_path"]):
            pth_success = True
        else:
            pth_success = False
            logging.error("PTH download failed.")

    # --- Final Summary ---
    if zip_success and pth_success:
        logging.info("--- Dataset setup completed successfully! ---")
    else:
        logging.error("--- Dataset setup failed or was incomplete. Please check logs. ---")


if __name__ == "__main__":
    logging.info("Starting dataset setup script...")
    setup_data()