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

def process_parquet_files(folder_path: str, alpha_add: float = 0.98, alpha_cancel: float = 0.98, alpha_trade: float = 0.985):
    """
    Process parquet files from a folder, transform dataframes and identify outliers based on time delta quantiles.
    Creates plots showing price movements and rare events using both matplotlib and plotly.
    Logs processing details and summary statistics.
    Separates analysis by zero vs non-zero price differences.
    
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
    
    for file in tqdm(files):
        logging.info(f"Processing file: {file}")
        logging.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        df = pl.read_parquet(file)
        df = transform_dataframe(df)
        
        # Shift imbalance down by one row
        df = df.with_columns([
            pl.col('imbalance').shift(1).alias('imbalance')
        ])
        
        # Split into zero and non-zero price difference dataframes
        df_zero = df.filter(pl.col("diff_price") == 0)
        df_nonzero = df.filter(pl.col("diff_price") != 0)
        
        # Calculate percentage split
        total_rows = len(df)
        zero_pct = len(df_zero) / total_rows * 100
        nonzero_pct = len(df_nonzero) / total_rows * 100
        
        logging.info(f"Data split:")
        logging.info(f"Zero price difference: {zero_pct:.2f}%")
        logging.info(f"Non-zero price difference: {nonzero_pct:.2f}%")
        print(f"Zero price difference: {zero_pct:.2f}%")
        print(f"Non-zero price difference: {nonzero_pct:.2f}%")

        # Process each dataset separately
        for df_type, df_subset in [("zero_spread", df_zero), ("nonzero_spread", df_nonzero)]:
            # Calculate summary statistics
            stats = {
                "total_rows": len(df_subset),
                "num_trades": len(df_subset.filter(pl.col("action") == "T")),
                "num_bid_updates": len(df_subset.filter(pl.col("side") == "B")),
                "num_ask_updates": len(df_subset.filter(pl.col("side") == "A")),
                "avg_trade_size": df_subset.filter(pl.col("action") == "T")["size"].mean(),
                "max_price": df_subset["price"].max(),
                "min_price": df_subset["price"].min(),
                "avg_price": df_subset["price"].mean(),
                "price_volatility": df_subset["price"].std(),
                "avg_bid_ask_spread": (df_subset["ask_px_00"] - df_subset["bid_px_00"]).mean()
            }
            
            # Log statistics
            logging.info(f"\nSummary Statistics for {df_type}:")
            for stat, value in stats.items():
                logging.info(f"{stat}: {value}")
                print(f"{df_type} - {stat}: {value}")
            logging.info(f"Alpha_add: {alpha_add}, Alpha_cancel: {alpha_cancel}, Alpha_trade: {alpha_trade}")

            # Convert deltas to seconds
            df_subset = df_subset.with_columns([
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
            outliers_file = os.path.join(txt_dir, f"{stock}_{file_date}_{df_type}_outliers.txt")
            
            dic_alpha = {"A": alpha_add, "C": alpha_cancel, "T": alpha_trade}
            with open(outliers_file, 'w') as f:
                for col, (action, delta_col, title) in enumerate(event_types):
                    # Filter dataframe for this action
                    action_df = df_subset.filter(pl.col('action') == action)
                    
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
                        
                        # Calculate threshold and get points with quantiles
                        if len(bucket_df) > 0:
                            # Sort points by delta values
                            sorted_df = bucket_df.sort(delta_col)
                            n_points = len(sorted_df)
                            
                            # Calculate quantiles for each point
                            quantiles = [1 - (i+1)/n_points for i in range(n_points)]
                            points_with_quantiles = list(zip(
                                sorted_df['ts_event'].to_list(),
                                quantiles
                            ))
                            
                            # Get points with quantiles above alpha threshold
                            outliers = [(point, q) for point, q in points_with_quantiles 
                                      if q >= dic_alpha[action]]
                            
                            if outliers:
                                outlier_points = [point for point, _ in outliers]
                                outlier_df = bucket_df.filter(pl.col('ts_event').is_in(outlier_points))
                                
                                # Plot outliers
                                ax.scatter(outlier_df['ts_event'], outlier_df['price'], 
                                         color='red', alpha=0.8, s=20)
                                
                                # Write outliers with quantiles to text file
                                f.write(f"Action: {title}, Bucket: [{imbalance_bins[row]:.2f}, {imbalance_bins[row+1]:.2f}]\n")
                                for point, quantile in outliers:
                                    f.write(f"{point},{quantile:.6f}\n")
                                f.write("\n")
                        
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
                                    f"{stock}_{file_date}_{df_type}_imbalance_buckets.png")
            plt.savefig(plot_path)
            plt.close()

        logging.info(f"Completed processing file: {file}\n")

if __name__ == "__main__":
    data_folder = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/KHC_filtered"
    process_parquet_files(data_folder)
