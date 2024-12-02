import os
import databento as db
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='debug_info.log'
)

def display_dbn_info(directory: str) -> None:
    """Display information about DBN files in the given directory"""
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.")
        
        # Get dbn files and sort by size
        dbn_files = [f for f in os.listdir(directory) if f.endswith(".dbn")]
        dbn_files.sort(key=lambda x: os.path.getsize(os.path.join(directory, x)))
        
        print(f"\nAnalyzing {len(dbn_files)} .dbn files at {datetime.now().strftime('%H:%M:%S')}")
        
        for file_idx, file_name in enumerate(dbn_files, 1):
            try:
                print(f"\nFile {file_idx}/{len(dbn_files)}: {file_name}")
                file_path = os.path.join(directory, file_name)
                
                # Log file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"File size: {file_size:.2f} MB")
                
                print("Loading DBN metadata...", end='', flush=True)
                store = db.DBNStore.from_file(file_path)
                print(" Done")

                # Display file information
                print(f"\nDate: {file_name.split('itch-')[1].split('.')[0]}")
                print(f"Number of unique symbols: {len(store.metadata.symbols)}")
                print(f"Number of total records: {store.metadata.record_count:,}")
                print(f"Schema: {store.metadata.schema}")
                print(f"Start timestamp: {store.metadata.start}")
                print(f"End timestamp: {store.metadata.end}")
                print(f"Dataset: {store.metadata.dataset}")
                print(f"Symbols: {', '.join(list(store.metadata.symbols)[:5])}...")
                
                # Clean up
                del store
                
            except Exception as e:
                logging.error(f"Error analyzing file {file_name}: {str(e)}")
                print(f"Error analyzing file {file_name}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting DBN file analysis...")
        display_dbn_info("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/dbn/NASDAQ")
        print("\nAnalysis completed successfully!")
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        print(f"Script failed: {str(e)}")