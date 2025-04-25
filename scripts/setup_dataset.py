# ./scripts/setup_dataset.py

import os
import argparse
from datasets import load_dataset, DownloadConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_hf_dataset(dataset_name: str, config_name: str = None, save_path: str = None):
    """
    Downloads a dataset from Hugging Face Hub and saves it to a specified directory.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub (e.g., 'glue', 'squad').
        config_name (str, optional): The specific configuration or subset of the dataset
                                     (e.g., 'mrpc' for 'glue'). Defaults to None.
        save_path (str, optional): The directory path where the dataset should be saved.
                                   If None, defaults to './data_and_outputs/raw_data' relative
                                   to the script's parent directory (repo root).
    """
    # --- Determine Save Path ---
    if save_path is None:
        # Assume the script is in './scripts', calculate root and then target path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir) # Go up one level from ./scripts
        save_path = os.path.join(root_dir, 'data_and_outputs', 'raw_data')
        logging.info(f"Save path not specified, defaulting to: {save_path}")

    # --- Ensure Save Directory Exists ---
    try:
        os.makedirs(save_path, exist_ok=True)
        logging.info(f"Ensured directory exists: {save_path}")
    except OSError as e:
        logging.error(f"Error creating directory {save_path}: {e}")
        raise # Re-raise the exception after logging

    # --- Construct full dataset identifier ---
    full_dataset_id = dataset_name
    if config_name:
        full_dataset_id += f" ({config_name})"

    logging.info(f"Attempting to download dataset: '{full_dataset_id}'")

    # --- Download and Save Dataset ---
    try:
        # Note: load_dataset downloads to a cache first by default.
        # We then save it explicitly to our target directory.
        # Using cache_dir in load_dataset points the *cache*, not the final save location directly.
        # For saving directly in the desired format (Arrow files, etc.), use save_to_disk.

        logging.info("Loading dataset from Hugging Face Hub...")
        dataset = load_dataset(path=dataset_name, name=config_name) # Downloads all available splits by default

        # Define the final save path *within* the target directory
        # Usually good practice to save it in a subdirectory named after the dataset
        dataset_save_dir = os.path.join(save_path, dataset_name.replace('/', '_')) # Replace slashes for valid dir names
        if config_name:
            dataset_save_dir += f"_{config_name}"

        os.makedirs(dataset_save_dir, exist_ok=True) # Ensure subdirectory exists

        logging.info(f"Saving dataset to disk at: {dataset_save_dir}")
        dataset.save_to_disk(dataset_save_dir)

        logging.info(f"Dataset '{full_dataset_id}' successfully downloaded and saved to {dataset_save_dir}")

    except Exception as e:
        logging.error(f"Failed to download or save dataset '{full_dataset_id}': {e}")
        # Consider more specific exception handling (e.g., FileNotFoundError for dataset, network errors)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face and save it locally.")

    parser.add_argument("dataset_name",
                        type=str,
                        help="Name of the dataset on Hugging Face Hub (e.g., 'glue', 'wikitext', 'imdb').")
    parser.add_argument("-c", "--config_name",
                        type=str,
                        default=None,
                        help="Specific configuration or subset of the dataset (e.g., 'mrpc' for 'glue', 'wikitext-2-raw-v1' for 'wikitext'). Optional.")
    parser.add_argument("-o", "--output_dir",
                        type=str,
                        default=None,
                        help="Directory to save the dataset. Defaults to './data_and_outputs/raw_data' relative to the repository root.")

    args = parser.parse_args()

    # Use the provided output directory if specified, otherwise let the function handle the default
    target_save_path = args.output_dir if args.output_dir else None

    download_hf_dataset(dataset_name=args.dataset_name,
                        config_name=args.config_name,
                        save_path=target_save_path)