import zipfile
import os

def unzip_file(zip_file_path: str, extract_to: str) -> None:
    """
    Function to unzip a zip file to a specified directory.

    Args:
    zip_file_path (str): Path to the zip file.
    extract_to (str): Directory where the contents will be extracted.

    Returns:
    None
    """
    # Check if the zip file exists
    if not os.path.isfile(zip_file_path):
        raise FileNotFoundError(f"The zip file {zip_file_path} does not exist.")
    
    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Example usage
unzip_file("/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/download.zip", "/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/")