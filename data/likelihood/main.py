import polars as pl
import numpy as np
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataframe_transform import transform_dataframe
from datetime import datetime
import logging

def process_parquet_files(folder_path: str, alpha_add: float = 0.998, alpha_cancel: float = 0.995, alpha_trade: float = 0.98):
    """
    Process parquet files from a folder, transform dataframes and identify outliers based on time delta quantiles.
    Creates plots showing price movements and rare events using both matplotlib and plotly.
    Logs processing details and summary statistics.
    
    Args:
        folder_path: Path to folder containing parquet files
        alpha: Quantile threshold for identifying outliers (default 0.95 if not specified in log)
        
    Note: Time is in nanoseconds using Databento MBP10 format
    """
    # Get stock symbol from folder path
    stock = os.path.basename(folder_path).split('_')[0].upper()
    
    # Get current date
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Setup logging - one log file for all processing
    log_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"processing_{stock}_{current_date}_{datetime.now().strftime('%H%M%S')}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    files = glob.glob(os.path.join(folder_path, "*PL.parquet"))
    plot_output_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/png"
    txt_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/txt"
    os.makedirs(plot_output_dir, exist_ok=True)
    
    schema = {
        'ts_event': pl.String,
        'action': pl.Utf8,
        'side': pl.Utf8, 
        'size': pl.Int64,
        'price': pl.Float64,
        'bid_px_00': pl.Float64,
        'ask_px_00': pl.Float64,
        'bid_sz_00': pl.Int64,
        'ask_sz_00': pl.Int64,
        'bid_ct_00': pl.Int64,
        'ask_ct_00': pl.Int64,
        'bid_px_01': pl.Float64,
        'ask_px_01': pl.Float64,
        'bid_sz_01': pl.Int64,
        'ask_sz_01': pl.Int64,
        'bid_ct_01': pl.Int64,
        'ask_ct_01': pl.Int64,
        'time_diff': pl.Float64,
        'price_same': pl.Float64,
        'price_opposite': pl.Float64,
        'size_same': pl.Int64,
        'size_opposite': pl.Int64,
        'nb_ppl_same': pl.Int64,
        'nb_ppl_opposite': pl.Int64,
        'diff_price': pl.Float64,
        'Mean_price_diff': pl.Float64,
        'imbalance': pl.Float64,
        'indice': pl.Int64,
        'bid_sz_00_diff': pl.Int64,
        'ask_sz_00_diff': pl.Int64,
        'status': pl.Utf8,
        'new_limite': pl.Utf8,
        'price_middle': pl.Float64
    }
    for file in tqdm(files):
        logging.info(f"Processing file: {file}")
        logging.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        df = pl.read_parquet(file, schema=schema)
        df = transform_dataframe(df)
        
        # Calculate summary statistics
        stats = {
            "total_rows": len(df),
            "num_trades": len(df.filter(pl.col("action") == "T")),
            "num_bid_updates": len(df.filter(pl.col("side") == "B")),
            "num_ask_updates": len(df.filter(pl.col("side") == "A")),
            "avg_trade_size": df.filter(pl.col("action") == "T")["size"].mean(),
            "max_price": df["price"].max(),
            "min_price": df["price"].min(),
            "avg_price": df["price"].mean(),
            "price_volatility": df["price"].std(),
            "avg_bid_ask_spread": (df["ask_px_00"] - df["bid_px_00"]).mean()
        }
        
        # Log statistics
        logging.info("Summary Statistics:")
        for stat, value in stats.items():
            logging.info(f"{stat}: {value}")
            print(f"{stat}: {value}")
        logging.info(f"Alpha_add: {alpha_add}, Alpha_cancel: {alpha_cancel}, Alpha_trade: {alpha_trade}")

        # Convert deltas to seconds
        df = df.with_columns([
            pl.col('add_deltas').truediv(1e9).alias('add_deltas_sec'),
            pl.col('cancel_deltas').truediv(1e9).alias('cancel_deltas_sec'),
            pl.col('trade_deltas').truediv(1e9).alias('trade_deltas_sec')
        ])

        # Create imbalance buckets
        imbalance_bins = np.linspace(-1, 1, 9)  # 8 buckets
        
        # Create figure with 24 subplots (3 event types x 8 imbalance buckets)
        fig, axes = plt.subplots(8, 3, figsize=(20, 40))
        event_types = [('A', 'add_deltas_sec', 'Add'), 
                      ('C', 'cancel_deltas_sec', 'Cancel'),
                      ('T', 'trade_deltas_sec', 'Trade')]
        
        # Create text file for outliers
        file_date = os.path.basename(file).split('_')[0]
        outliers_file = os.path.join(txt_dir, f"{stock}_{file_date}_outliers.txt")
        
        dic_alpha = {"A": alpha_add, "C": alpha_cancel, "T": alpha_trade}
        with open(outliers_file, 'w') as f:
            for col, (action, delta_col, title) in enumerate(event_types):
                # Filter dataframe for this action
                action_df = df.filter(pl.col('action') == action)
                
                for row in range(8):
                    ax = axes[row, col]
                    
                    # Plot price and bid-ask for all data points
                    ax.plot(action_df['ts_event'], action_df['price'], color='blue', alpha=0.7)
                    ax.plot(action_df['ts_event'], action_df['bid_px_00'], color='green', alpha=0.3)
                    ax.plot(action_df['ts_event'], action_df['ask_px_00'], color='red', alpha=0.3)
                    
                    # Filter for current imbalance bucket
                    bucket_df = action_df.filter(
                        (pl.col('imbalance') >= imbalance_bins[row]) & 
                        (pl.col('imbalance') < imbalance_bins[row+1])
                    )
                    
                    # Calculate threshold for this bucket
                    if len(bucket_df) > 0:
                        bucket_threshold = bucket_df.select(
                            pl.col(delta_col).quantile(1-dic_alpha[action])
                        ).item()
                        
                        # Get outliers for this bucket
                        outliers = bucket_df.filter(pl.col(delta_col) < bucket_threshold)
                        
                        # Plot outliers
                        ax.scatter(outliers['ts_event'], outliers['price'], 
                                 color='red', alpha=0.8, s=20)
                        
                        # Write outliers to text file
                        outlier_times = outliers['ts_event'].to_list()
                        f.write(f"Action: {title}, Bucket: [{imbalance_bins[row]:.2f}, {imbalance_bins[row+1]:.2f}]\n")
                        f.write(','.join(map(str, outlier_times)) + '\n\n')
                    
                    # Set title and labels
                    ax.set_title(f'{title} Events - Imbalance [{imbalance_bins[row]:.2f}, {imbalance_bins[row+1]:.2f}]')
                    if row == 7:  # Bottom row
                        ax.set_xlabel('Time')
                    if col == 0:  # First column
                        ax.set_ylabel('Price')
                    
                    # Rotate x-axis labels
                    ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        
        plot_path = os.path.join(plot_output_dir,
                                f"{stock}_{file_date}_imbalance_buckets.png")
        plt.savefig(plot_path)
        plt.close()
        
        output_path = os.path.join(os.path.dirname(file), 
                                 f"{stock}_{file_date}_transformed.parquet")
        # df.write_parquet(output_path)
        logging.info(f"Completed processing file: {file}\n")

if __name__ == "__main__":
    data_folder = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/GOOGL_filtered"
    process_parquet_files(data_folder)
