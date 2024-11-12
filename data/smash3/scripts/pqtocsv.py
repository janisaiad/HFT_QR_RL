import polars as pl
import os
import glob
from tqdm import tqdm
# Define base directory and stock symbols
base_dirs = [
    "/home/janis/3A/EA/done ed/CSV_GOOGL_NASDAQ_SL",
    "/home/janis/3A/EA/done ed/CSV_KHC_NASDAQ_SL", 
    "/home/janis/3A/EA/done ed/CSV_LCID_NASDAQ_SL",
    "/home/janis/3A/EA/done ed/CSV_GOOGL_NASDAQ_PL",
    "/home/janis/3A/EA/done ed/CSV_KHC_NASDAQ_PL",
    "/home/janis/3A/EA/done ed/CSV_LCID_NASDAQ_PL"
]
symbols = ["GOOGL", "KHC", "LCID"]
sl_types = ["SL", "PL"]

# Process each CSV file
for base_dir in tqdm(base_dirs):
    # Create search pattern for CSV files
    search_pattern = os.path.join(base_dir, f"xnas-itch-*.mbp-10.csv")
    
    # Get list of all matching CSV files
    csv_files = glob.glob(search_pattern)
    
    for csv_file in tqdm(csv_files):
        # Extract date from filename
        date = csv_file.split("-")[2][:8]  # Gets YYYYMMDD from filename
        
        # Read CSV file using polars
        df = pl.read_csv(csv_file)
        
        # Determine SL/PL type from base directory
        sl = "SL" if "SL" in base_dir else "PL"
        
        # Determine symbol from base directory
        symbol = next(s for s in symbols if s in base_dir)
        
        # Create output directory path
        output_dir = f"/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/{symbol}_filtered"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename with date and filtered tag
        output_filename = os.path.join(output_dir, f"{date}_filtered_{sl}.parquet")
        
        # Convert to parquet and save using polars
        df.write_parquet(output_filename, compression="snappy")
        
        print(f"Converted {csv_file} to {output_filename}")
