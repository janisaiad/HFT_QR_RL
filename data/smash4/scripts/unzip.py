import os
import zstandard as zstd
from tqdm import tqdm

def extract_zst_files(directory: str) -> None:
    """
    Function to extract all .zst files in a specified directory.

    Args:
    directory (str): Directory where the .zst files are located.

    Returns:
    None
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    # Define the output directory
    output_directory = "/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/dbn/"
    
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get the list of .zst files in the directory
    zst_files = [file_name for file_name in os.listdir(directory) if file_name.endswith(".zst")]
    
    # Iterate over all .zst files in the directory with progress bar
    for file_name in tqdm(zst_files, desc="Extracting .zst files"):
        file_path = os.path.join(directory, file_name)
        output_file_path = os.path.join(output_directory, os.path.splitext(file_name)[0])  # Remove .zst extension
        
        # Open the .zst file and decompress it
        with open(file_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with open(output_file_path, 'wb') as output_file:
                dctx.copy_stream(compressed_file, output_file)

# Example usage
extract_zst_files("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/download/DBEQ-20241022-MYAHGPS79Q")
