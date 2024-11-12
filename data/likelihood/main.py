import polars as pl
import numpy as np
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt
from dataframe_transform import transform_dataframe

def process_parquet_files(folder_path: str, alpha: float = 0.95):
    """
    Process parquet files from a folder, transform dataframes and identify outliers based on time delta quantiles.
    Creates plots showing rare trade events.
    
    Args:
        folder_path: Path to folder containing parquet files
        alpha: Quantile threshold for identifying outliers (default 0.95)
        
    Note: Time is in nanoseconds using Databento MBP10 format
    """
    # Get list of parquet files
    files = glob.glob(os.path.join(folder_path, "*PL.parquet"))
    
    for file in tqdm(files):
        # Read parquet file
        df = pl.read_parquet(file)
        
        # Apply transformations from dataframe_transform.py
        df = transform_dataframe(df)
        
        # Convert nanosecond timestamps to seconds for analysis
        df = df.with_columns([
            (pl.col('time_diff') / 1e9).alias('time_diff_sec')
        ])
        
        # Calculate time delta quantile threshold (in seconds)
        threshold = df['time_diff_sec'].quantile(alpha)
        
        # Add column indicating if row is an outlier
        df = df.with_columns([
            pl.when(pl.col('time_diff_sec') > threshold)
            .then(True)
            .otherwise(False)
            .alias('is_outlier')
        ])
        
        # Convert to pandas for plotting
        pdf = df.to_pandas()
        
        # Create plot
        plt.figure(figsize=(15,5))
        plt.plot(pdf.index, pdf['time_diff_sec'], color='blue', alpha=0.5, label='Time between events')
        
        # Highlight rare trade events in red
        rare_trades = pdf[(pdf['action'] == 'T') & (pdf['is_outlier'])]
        plt.scatter(rare_trades.index, rare_trades['time_diff_sec'], 
                   color='red', alpha=0.8, label='Rare trade events')
        
        plt.axhline(y=threshold, color='green', linestyle='--', 
                   label=f'{alpha} quantile threshold')
        plt.ylabel('Time between events (seconds)')
        plt.xlabel('Event index')
        plt.title('Time Series with Rare Trade Events Highlighted')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(file),
                                f"{os.path.basename(file).split('.')[0]}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Save transformed dataframe
        output_path = os.path.join(os.path.dirname(file), 
                                 f"{os.path.basename(file).split('.')[0]}_transformed.parquet")
        df.write_parquet(output_path)

if __name__ == "__main__":
    data_folder = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID"
    process_parquet_files(data_folder)
