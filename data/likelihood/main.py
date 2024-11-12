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

def process_parquet_files(folder_path: str, alpha: float = 0.975):
    """
    Process parquet files from a folder, transform dataframes and identify outliers based on time delta quantiles.
    Creates plots showing price movements and rare events using both matplotlib and plotly.
    Logs processing details and summary statistics.
    
    Args:
        folder_path: Path to folder containing parquet files
        alpha: Quantile threshold for identifying outliers (default 0.975)
        
    Note: Time is in nanoseconds using Databento MBP10 format
    """
    # Setup logging
    log_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    files = glob.glob(os.path.join(folder_path, "*PL.parquet"))
    plot_output_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood/png"
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
    
    for file in tqdm(files[0:1]):
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

        # Convert deltas to seconds
        df = df.with_columns([
            pl.col('bid_deltas').truediv(1e9).alias('bid_deltas_sec'),
            pl.col('ask_deltas').truediv(1e9).alias('ask_deltas_sec'),
            pl.col('trade_deltas').truediv(1e9).alias('trade_deltas_sec')
        ])

        # Calculate thresholds for each type
        bid_threshold = df.select(pl.col('bid_deltas_sec').quantile(1-alpha)).item()
        ask_threshold = df.select(pl.col('ask_deltas_sec').quantile(1-alpha)).item()
        trade_threshold = df.select(pl.col('trade_deltas_sec').quantile(1-alpha)).item()

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 21))

        # Plot for bid deltas
        bid_df = df.filter(pl.col('side') == 'B')
        ax1.plot(bid_df['ts_event'], bid_df['price'], color='blue', alpha=0.7)
        bid_outliers = bid_df.filter(pl.col('bid_deltas_sec') < bid_threshold)
        ax1.scatter(bid_outliers['ts_event'], bid_outliers['price'], color='red', alpha=0.8)
        ax1.set_title('Bid Events with Small Time Deltas')
        ax1.set_ylabel('Price')

        # Plot for ask deltas  
        ask_df = df.filter(pl.col('side') == 'A')
        ax2.plot(ask_df['ts_event'], ask_df['price'], color='blue', alpha=0.7)
        ask_outliers = ask_df.filter(pl.col('ask_deltas_sec') < ask_threshold)
        ax2.scatter(ask_outliers['ts_event'], ask_outliers['price'], color='red', alpha=0.8)
        ax2.set_title('Ask Events with Small Time Deltas')
        ax2.set_ylabel('Price')

        # Plot for trade deltas
        trade_df = df.filter(pl.col('action') == 'T')
        ax3.plot(trade_df['ts_event'], trade_df['price'], color='blue', alpha=0.7)
        trade_outliers = trade_df.filter(pl.col('trade_deltas_sec') < trade_threshold)
        ax3.scatter(trade_outliers['ts_event'], trade_outliers['price'], color='red', alpha=0.8)
        ax3.set_title('Trade Events with Small Time Deltas')
        ax3.set_ylabel('Price')
        ax3.set_xlabel('Time')

        plt.tight_layout()
        plot_path = os.path.join(plot_output_dir,
                                f"{os.path.basename(file).split('.')[0]}_deltas_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Create interactive plotly plot
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=('Bid Events', 'Ask Events', 'Trade Events'))

        # Bid subplot
        fig.add_trace(
            go.Scatter(x=bid_df['ts_event'], y=bid_df['price'],
                      mode='lines', name='Bid Price', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=bid_outliers['ts_event'], y=bid_outliers['price'],
                      mode='markers', name='Bid Outliers',
                      marker=dict(color='red', size=8)),
            row=1, col=1
        )

        # Ask subplot
        fig.add_trace(
            go.Scatter(x=ask_df['ts_event'], y=ask_df['price'],
                      mode='lines', name='Ask Price', line=dict(color='blue', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=ask_outliers['ts_event'], y=ask_outliers['price'],
                      mode='markers', name='Ask Outliers',
                      marker=dict(color='red', size=8)),
            row=2, col=1
        )

        # Trade subplot
        fig.add_trace(
            go.Scatter(x=trade_df['ts_event'], y=trade_df['price'],
                      mode='lines', name='Trade Price', line=dict(color='blue', width=1)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=trade_outliers['ts_event'], y=trade_outliers['price'],
                      mode='markers', name='Trade Outliers',
                      marker=dict(color='red', size=8)),
            row=3, col=1
        )

        fig.update_layout(height=1200, title_text="Events with Small Time Deltas (Interactive)",
                         showlegend=True)
        
        plotly_path = os.path.join(plot_output_dir,
                                  f"{os.path.basename(file).split('.')[0]}_deltas_plotly.html")
        fig.write_html(plotly_path)
        
        output_path = os.path.join(os.path.dirname(file), 
                                 f"{os.path.basename(file).split('.')[0]}_transformed.parquet")
        df.write_parquet(output_path)
        logging.info(f"Completed processing file: {file}\n")

if __name__ == "__main__":
    data_folder = "/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID_filtered"
    process_parquet_files(data_folder)
