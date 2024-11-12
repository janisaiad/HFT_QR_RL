import polars as pl
import numpy as np
from tqdm import tqdm
import glob
import os
from dataframe_transform import transform_dataframe

def process_parquet_files(folder_path: str, alpha: float = 0.95):
    """
    Process parquet files from a folder, transform dataframes and identify outliers based on time delta quantiles
    
    Args:
        folder_path: Path to folder containing parquet files
        alpha: Quantile threshold for identifying outliers (default 0.95)
    """
    # Get list of parquet files
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    
    for file in tqdm(files):
        # Read parquet file
        df = pl.read_parquet(file)
        
        # Apply transformations from dataframe_transform.py
        df = transform_dataframe(df)
        
        # Calculate time delta quantile threshold
        threshold = df['time_diff'].quantile(alpha)
        
        # Add column indicating if row is an outlier
        df = df.with_columns([
            pl.when(pl.col('time_diff') > threshold)
            .then(True)
            .otherwise(False)
            .alias('is_outlier')
        ])
        
        # Save transformed dataframe
        output_path = os.path.join(os.path.dirname(file), 
                                 f"{os.path.basename(file).split('.')[0]}_transformed.parquet")
        df.write_parquet(output_path)

if __name__ == "__main__":
    data_folder = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/CHICAGO/LCID"
    process_parquet_files(data_folder)
