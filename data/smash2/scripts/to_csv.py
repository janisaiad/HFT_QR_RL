import os
import pandas as pd
import databento as db
from tqdm import tqdm

def convert_dbn_to_csv(directory: str) -> None:
    """
    Function to convert all .dbn files in a specified directory to CSV files,
    and organize them into folders based on the 'symbol' value in the dataframe.

    Args:
    directory (str): Directory where the .dbn files are located.

    Returns:
    None
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    # Define the output directory
    output_directory = "/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get the list of .dbn files in the directory
    dbn_files = [file_name for file_name in os.listdir(directory) if file_name.endswith(".dbn")]
    
    # Iterate over all .dbn files in the directory with progress bar
    for file_name in tqdm(dbn_files, desc="Converting .dbn files to CSV"):
        file_path = os.path.join(directory, file_name)
        # Read the .dbn file using databento
        store = db.DBNStore.from_file(file_path)
        df = store.to_df()
        data = df.to_dict(orient='records')
        
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Group the data by 'symbol' and save each group to a separate CSV file
        for symbol, group in df.groupby("symbol"):
            # Create a folder for the symbol if it doesn't exist
            symbol_folder = os.path.join(output_directory, symbol)
            if not os.path.exists(symbol_folder):
                os.makedirs(symbol_folder)
            
            # Extract the date from the original file name
            date_str = file_name.split("basic-")[1].split(".")[0]
            
            # Define the output CSV file path
            output_file_path = os.path.join(symbol_folder, f"{date_str}.csv")
            
            # Save the group to a CSV file
            group.to_csv(output_file_path, index=False)

# Example usage
convert_dbn_to_csv("/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/")


# important : error  33/64 afer that disk usage is 100%