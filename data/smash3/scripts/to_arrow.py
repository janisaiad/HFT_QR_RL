import os
import polars as pl
import databento as db
from tqdm import tqdm
import psutil
import logging
import sys
from datetime import datetime

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
    print(f"Current memory usage: {mem:.2f} MB")

def convert_dbn_to_csv(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.")
        
        output_directory = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Get dbn files and sort by size
        dbn_files = [f for f in os.listdir(directory) if f.endswith(".dbn")]
        dbn_files.sort(key=lambda x: os.path.getsize(os.path.join(directory, x)))
        
        print(f"\nStarting conversion of {len(dbn_files)} .dbn files at {datetime.now().strftime('%H:%M:%S')}")
        
        for file_idx, file_name in enumerate(dbn_files, 1):
            try:
                print(f"\nProcessing file {file_idx}/{len(dbn_files)}: {file_name}")
                logging.info(f"Processing file: {file_name}")
                log_memory_usage()
                
                file_path = os.path.join(directory, file_name)
                
                # Log file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                logging.info(f"File size: {file_size:.2f} MB")
                print(f"File size: {file_size:.2f} MB")
                
                print("Loading DBNStore...", end='', flush=True)
                store = db.DBNStore.from_file(file_path)
                print(" Done")
                logging.info("DBNStore loaded")
                log_memory_usage()

                date_str = file_name.split("itch-")[1].split(".")[0]
                
                # Get unique symbols from metadata
                symbols = store.metadata.symbols
                n_symbols = len(symbols)
                logging.info(f"Number of unique symbols: {n_symbols}")
                print(f"Processing {n_symbols} unique symbols...")
                
                with tqdm(total=n_symbols, desc="Saving symbol data") as pbar:
                    for symbol in symbols:
                        symbol_folder = os.path.join(output_directory, symbol)
                        if not os.path.exists(symbol_folder):
                            os.makedirs(symbol_folder)
                        
                        output_file_path = os.path.join(symbol_folder, f"{date_str}.csv")
                        
                        # Filter and write data for this symbol using to_csv
                        store.to_csv(
                            path=output_file_path,
                            pretty_ts=True,
                            pretty_px=True,
                            map_symbols=True
                        )
                        pbar.update(1)
                
                # Explicitly clean up
                del store
                print(f"Completed processing {file_name}")
                
            except Exception as e:
                logging.error(f"Error processing file {file_name}: {str(e)}")
                print(f"Error processing file {file_name}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting DBN to CSV conversion process...")
        convert_dbn_to_csv("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/dbn/NASDAQ")
        print("\nConversion completed successfully!")
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        print(f"Script failed: {str(e)}")