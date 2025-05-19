import os
import requests
import tarfile
import logging
import pathlib
from tqdm import tqdm
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for visibility
)

FILES_TO_DOWNLOAD = [
    {
        "URL": "https://huggingface.co/zeinebH/TumorNetSolvers/resolve/main/checkpoint_TumorSurrogate_best_ema_dice.tar.gz",
        "NAME": "ts_dice.tar.gz"
    },
    {
        "URL": "https://huggingface.co/zeinebH/TumorNetSolvers/resolve/main/checkpoint_TumorSurrogate_best_ema_loss.tar.gz",
        "NAME": "ts_loss.tar.gz"
    },
    {
        "URL": "https://huggingface.co/zeinebH/TumorNetSolvers/resolve/main/checkpoint_ViT_best_ema_dice.tar.gz",
        "NAME": "vit_dice.tar.gz"
    },
    {
        "URL": "https://huggingface.co/zeinebH/TumorNetSolvers/resolve/main/checkpoint_ViT_best_ema_loss.tar.tar.gz.gz",
        "NAME": "vit_loss.tar.gz"
    },
    {
        "URL": "https://huggingface.co/zeinebH/TumorNetSolvers/resolve/main/checkpoint_nnUnet_best_ema_dice.pth.tar.gz",
        "NAME": "nnUnet_dice.tar.gz"
    },
    {
        "URL": "https://huggingface.co/zeinebH/TumorNetSolvers/resolve/main/checkpoint_ViT_best_ema_loss.tar.gz",
        "NAME": "nnUnet_loss.tar.gz"
    }
]

DOWNLOAD_DIR = "./final_models"



def download_file(url: str, file_name: str):
    """Downloads a file from a URL to a save path with a progress bar."""
    logging.info(f"Attempting to download: {url}")
    save_path = os.path.join(DOWNLOAD_DIR, file_name)
    try:
        response = requests.get(url, stream=True, timeout=60) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        logging.info(f"Downloading to: {save_path}")
        with open(save_path, 'wb') as file, tqdm(
            desc=f"{file_name}",
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

        logging.info(f"Successfully downloaded {file_name}")
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

def extract_tar_gz(tar_file_path: str):
    try:
        tar_path = pathlib.Path(tar_file_path)
        if not tar_path.is_file():
            raise FileNotFoundError(f"Error: File not found at '{tar_file_path}'")
        extract_directory = tar_path.parent
        extract_directory.mkdir(parents=True, exist_ok=True)
        print(f"Attempting to extract '{tar_path.name}' to '{extract_directory}'...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_directory)
            print(f"Successfully extracted '{tar_path.name}' to '{extract_directory}'")
        return True
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return False
    except tarfile.ReadError as read_error:
        print(f"Error: Failed to read tar file '{tar_file_path}'. It might not be a valid .tar.gz file or is corrupted.")
        print(f"Details: {read_error}")
        return False
    except PermissionError as perm_error:
        print(f"Error: Permission denied while trying to extract to '{extract_directory}'.")
        print(f"Details: {perm_error}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return False

def get_models():
    """Main function to download and set up dataset components."""
    # --- 1. Download and Extract ZIP ---
    if not os.path.exists(DOWNLOAD_DIR):
        os.mkdir(DOWNLOAD_DIR)

    logging.info("--- Starting Dataset ZIP Download and Extraction ---")
    for model in FILES_TO_DOWNLOAD:
        if download_file(model["URL"], model["NAME"]):
            file_path = os.path.join(DOWNLOAD_DIR, model["NAME"])
            if extract_tar_gz(file_path):
                try:
                    os.remove(file_path)
                    logging.info(f"Removed temporary zip file: {file_path}")
                    tar_success = True
                except OSError as e:
                    logging.warning(f"Could not remove temporary zip file {file_path}: {e}")
                    tar_success = True
            else:
                 tar_success = False
                 logging.error("Zip extraction failed. Aborting further steps dependent on it.")

    if not tar_success:
        logging.error("Setup incomplete due to ZIP file issues.")
        return


if __name__ == "__main__":
    logging.info("Starting dataset setup script...")
    get_models()
