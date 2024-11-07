import os
import polars as pl
import databento as db
from tqdm import tqdm

def convert_dbn_to_csv(directory: str) -> None:
    """
    Function to convert all .dbn files in a specified directory to CSV files,
    and organize them into folders based on the 'symbol' value in the dataframe.
    Uses polars for improved performance and memory efficiency.

    Args:
    directory (str): Directory where the .dbn files are located.

    Returns:
    None
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    # Define the output directory
    output_directory = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ"
    
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
        
        # Convert pandas DataFrame to polars DataFrame
        df = pl.from_pandas(df)
        
        # Extract the date from the original file name
        date_str = file_name.split("itch-")[1].split(".")[0]
        
        # Group and write by symbol using polars
        for symbol in df["symbol"].unique():
            # Create a folder for the symbol if it doesn't exist
            symbol_folder = os.path.join(output_directory, symbol)
            if not os.path.exists(symbol_folder):
                os.makedirs(symbol_folder)
            
            # Define the output CSV file path
            output_file_path = os.path.join(symbol_folder, f"{date_str}.csv")
            
            # Filter data for this symbol and write to CSV
            df.filter(pl.col("symbol") == symbol).write_csv(output_file_path)

# Example usage
convert_dbn_to_csv("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/dbn")

# important : error  33/64 afer that disk usage is 100%