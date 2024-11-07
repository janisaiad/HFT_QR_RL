import os
import polars as pl
import databento as db
from tqdm import tqdm
import psutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='debug_conversion.log'
)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    logging.info(f"Memory usage: {mem:.2f} MB")

def convert_dbn_to_csv(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.")
        
        output_directory = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        dbn_files = [f for f in os.listdir(directory) if f.endswith(".dbn")]
        
        for file_name in tqdm(dbn_files, desc="Converting .dbn files to CSV"):
            try:
                logging.info(f"Processing file: {file_name}")
                log_memory_usage()
                
                file_path = os.path.join(directory, file_name)
                
                # Log file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                logging.info(f"File size: {file_size:.2f} MB")
                
                # Read the file in chunks if possible
                store = db.DBNStore.from_file(file_path)
                logging.info("DBNStore loaded")
                log_memory_usage()
                
                df = store.to_df()
                logging.info("Converted to pandas DataFrame")
                log_memory_usage()
                
                df = pl.from_pandas(df)
                logging.info("Converted to polars DataFrame")
                log_memory_usage()
                
                date_str = file_name.split("itch-")[1].split(".")[0]
                unique_symbols = df["symbol"].unique()
                logging.info(f"Number of unique symbols: {len(unique_symbols)}")
                
                for symbol in unique_symbols:
                    symbol_folder = os.path.join(output_directory, symbol)
                    if not os.path.exists(symbol_folder):
                        os.makedirs(symbol_folder)
                    
                    output_file_path = os.path.join(symbol_folder, f"{date_str}.csv")
                    
                    # Write symbol data
                    symbol_df = df.filter(pl.col("symbol") == symbol)
                    symbol_df.write_csv(output_file_path)
                    
                # Explicitly clean up
                del df
                del store
                
            except Exception as e:
                logging.error(f"Error processing file {file_name}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        convert_dbn_to_csv("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/dbn/NASDAQ")
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")