import os
import polars as pl
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='divide_data.log'
)

def process_symbol_folder(base_path: str) -> None:
    """
    Process each symbol folder and filter parquet files to keep only matching symbol data.
    
    Args:
        base_path (str): Base path containing symbol folders
    """
    try:
        # Get all symbol folders
        symbol_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        
        print(f"Processing {len(symbol_folders)} symbol folders...")
        
        for symbol in tqdm(symbol_folders, desc="Processing symbols"):
            symbol_path = os.path.join(base_path, symbol)
            
            # Get all parquet files in the symbol folder
            parquet_files = [f for f in os.listdir(symbol_path) if f.endswith('.parquet')]
            
            for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
                file_path = os.path.join(symbol_path, parquet_file)
                
                try:
                    # Read parquet file
                    df = pl.read_parquet(file_path)
                    
                    # Filter for matching symbol
                    df_filtered = df.filter(pl.col("symbol") == symbol)
                    
                    # Save filtered data back to parquet
                    df_filtered.write_parquet(file_path)
                    
                    logging.info(f"Successfully processed {file_path}")
                    
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
                    
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    base_path = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ"
    
    try:
        print("Starting symbol data filtering process...")
        process_symbol_folder(base_path)
        print("\nFiltering completed successfully!")
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        print(f"Script failed: {str(e)}")
