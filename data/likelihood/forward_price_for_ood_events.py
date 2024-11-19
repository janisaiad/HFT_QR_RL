import polars as pl
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import os
import glob
import json
from datetime import datetime
import logging
from tqdm import tqdm

def analyze_forward_price_differences(points_file: str, spread_type: str) -> dict:
    """
    Analyze and plot the distribution of forward price differences for points from outlier file.
    
    Args:
        points_file: Path to txt file containing points grouped by action and bucket
        spread_type: Either 'zero_spread' or 'nonzero_spread'
        
    Returns:
        Dictionary containing various distribution metrics
    """
    # Parse points from outlier file
    all_timestamps = []
    try:
        with open(points_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('Action:') and ',' in line:
                    try:
                        timestamp, _ = line.strip().split(',')
                        all_timestamps.append(float(timestamp))
                    except ValueError as e:
                        logging.warning(f"Could not parse line: {line}. Error: {e}")
                        
    except FileNotFoundError:
        logging.error(f"File not found: {points_file}")
        return {}
    except Exception as e:
        logging.error(f"Error reading file {points_file}: {e}")
        return {}

    # Get stock and date from filename
    filename = os.path.basename(points_file)
    stock = filename.split('_')[0]
    file_date = filename.split('_')[1]
    logging.info(f"Processing stock {stock} for date {file_date}")

    # Create output directories with spread type
    plot_dir = f"/home/janis/3A/EA/HFT_QR_RL/data/likelihood/forward_price_plots/{stock}/{file_date}/{spread_type}"
    os.makedirs(plot_dir, exist_ok=True)
    logging.info(f"Created plot directory: {plot_dir}")

    # Read parquet and calculate true price differences
    parquet_path = f"/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/{stock}_filtered/{file_date}_filtered_PL.parquet"
    logging.info(f"Reading parquet file: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    
    # Calculate mid price and true price differences (50 events ahead)
    df = df.with_columns([
        ((pl.col("bid_px_00") + pl.col("ask_px_00"))/2).alias("mid_price"),
        ((pl.col("bid_sz_00") - pl.col("ask_sz_00"))/(pl.col("bid_sz_00") + pl.col("ask_sz_00"))).alias("imbalance")
    ])
    
    df = df.with_columns([
        (pl.col("mid_price").shift(-50) - pl.col("mid_price")).alias("true_price_diff_50")
    ])
    logging.info("Calculated forward price differences")
    
    # Save new dataframe with true price differences
    output_parquet = f"/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/{stock}_filtered/{file_date}_filtered_PL_with_true_diff_price.parquet"
    df.write_parquet(output_parquet)
    logging.info(f"Saved processed dataframe to: {output_parquet}")

    # Filter df for timestamps from outlier file and get imbalances and price differences
    df_filtered = df.filter(pl.col('ts_event').is_in(all_timestamps))
    imb_sample = df_filtered.get_column("imbalance").to_numpy()
    price_sample = df_filtered.get_column("true_price_diff_50").to_numpy()
    logging.info(f"Filtered data points: {len(df_filtered)} rows")

    # Create 2D histogram plot with KDE
    plt.figure(figsize=(12,8))
    logging.info("Created figure for plotting")
    
    # Plot 2D histogram
    plt.hist2d(imb_sample, price_sample, bins=50, cmap='viridis', density=True)
    logging.info("Created 2D histogram")
    
    # Calculate and plot KDE
    try:
        xy = np.vstack([imb_sample, price_sample])
        kde = gaussian_kde(xy)
        logging.info("Calculated KDE")
        
        # Create mesh grid for KDE
        xmin, xmax = imb_sample.min(), imb_sample.max()
        ymin, ymax = price_sample.min(), price_sample.max()
        
        x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([x.ravel(), y.ravel()])
        
        z = kde(positions)
        z = z.reshape(x.shape)
        
        # Plot KDE contours
        plt.contour(x, y, z, colors='r', alpha=0.5)
        logging.info("Added KDE contours to plot")
    except Exception as e:
        logging.warning(f"KDE failed for {stock} {file_date}: {e}")
    
    plt.colorbar(label='Density')
    plt.title(f'True Price Difference vs Imbalance Distribution\nAll Points ({spread_type})')
    plt.xlabel('Imbalance')
    plt.ylabel('True Forward Price Difference')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(plot_dir, f'{stock}_{file_date}_{spread_type}_distribution.png')
    try:
        plt.savefig(plot_path)
        logging.info(f"Successfully saved plot to: {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plot to {plot_path}: {e}")
    plt.close()
    logging.info("Closed plot")

    return {}

if __name__ == "__main__":
    # Base directories
    base_dir = "/home/janis/3A/EA/HFT_QR_RL/data/likelihood"
    txt_dir = os.path.join(base_dir, "txt")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=os.path.join(log_dir, f"forward_price_analysis_{current_date}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logging.info("Starting forward price analysis")
    
    # Process all txt files for both spread types
    for spread_type in ['zero_spread', 'nonzero_spread']:
        txt_files = glob.glob(os.path.join(txt_dir, f"*_{spread_type}_outliers.txt"))
        logging.info(f"Found {len(txt_files)} files for {spread_type}")
        
        for txt_file in tqdm(txt_files):
            logging.info(f"Processing {spread_type} file: {txt_file}")
            
            filename = os.path.basename(txt_file)
            stock_date = '_'.join(filename.split('_')[:2])  # Get stock and date
            
            try:
                metrics = analyze_forward_price_differences(txt_file, spread_type)
                
                # Include spread type in output filename
                output_file = os.path.join(results_dir, f"{stock_date}_{spread_type}_forward_price_metrics.json")
                with open(output_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                    
                logging.info(f"Successfully processed {filename}")
                logging.info(f"Metrics saved to {output_file}")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                continue
