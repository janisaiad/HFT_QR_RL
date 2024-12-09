import os
import json
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

def load_metrics_data(json_files, stock, type_, bucket, n_values, metrics):
    """Load metrics data from JSON files for given parameters."""
    stock_files = [f for f in json_files if f.stem.split("_")[0] == stock]
    metrics_data = {m: {n: [] for n in n_values} for m in metrics}
    days = []
    
    # Convert bucket to string with proper formatting, omitting + for positive values
    bucket_str = f"{bucket:.2f}"
    
    for file in tqdm(stock_files, desc=f"Loading metrics for {stock} {type_} {bucket}"):
        try:
            with open(file) as f:
                data = json.load(f)
                
            day = file.stem.split("_")[1]
            days.append(day)
                
            if type_ not in data or bucket_str not in data[type_]:
                logging.warning(f"Missing data for {type_} or bucket {bucket_str} in {file} for day {day}")
                # Add None for all metrics when data is missing
                for n in n_values:
                    for metric in metrics:
                        metrics_data[metric][n].append(None)
                continue
                
            for n in tqdm(n_values, desc=f"Processing n_values for {stock} {type_} {bucket}"):
                n_key = f"n_{n}"
                if n_key in data[type_][bucket_str]:
                    for metric in metrics:
                        if metric in data[type_][bucket_str][n_key]:
                            metrics_data[metric][n].append(
                                data[type_][bucket_str][n_key][metric]
                            )
                        else:
                            logging.warning(f"Missing metric {metric} for {n_key} in {file} for day {day}")
                            metrics_data[metric][n].append(None)
                else:
                    logging.warning(f"Missing n_value {n_key} in {file} for day {day}")
                    for metric in metrics:
                        metrics_data[metric][n].append(None)
                        
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")
            # Add None for all metrics when file processing fails
            for n in n_values:
                for metric in metrics:
                    metrics_data[metric][n].append(None)
                    
    return metrics_data, days

def create_distribution_plots(metrics_data, n_values, metrics, bucket, stock, type_):
    """Create distribution plots for each metric and n_value combination."""
    try:
        n_metrics = len(metrics)
        n_n_values = len(n_values)
        
        # Create a grid of subplots - one row per metric, one column per n_value
        fig, axes = plt.subplots(n_metrics, n_n_values, figsize=(5*n_n_values, 4*n_metrics))
        fig.suptitle(f'{stock} - {type_} - Bucket {bucket:.2f}', fontsize=16)
        
        for i, metric in enumerate(metrics):
            for j, n in enumerate(n_values):
                ax = axes[i,j]
                data = metrics_data[metric][n]
                
                if data and len(data) > 0:
                    # Convert to numpy array and filter invalid values
                    data_array = np.array(data)
                    valid_data = data_array[np.isfinite(data_array)]
                    
                    if len(valid_data) > 0:
                        try:
                            # Use auto bins
                            ax.hist(valid_data, bins='auto', alpha=0.7, density=True)
                            
                            # Adjust axis limits
                            ax.set_xlim(np.percentile(valid_data, [1, 99]))
                            
                        except Exception as e:
                            logging.warning(f"Couldn't create histogram for {metric} (n={n}): {str(e)}")
                            ax.text(0.5, 0.5, 'Error plotting', 
                                  horizontalalignment='center',
                                  verticalalignment='center')
                    else:
                        ax.text(0.5, 0.5, 'No Valid Data', 
                               horizontalalignment='center',
                               verticalalignment='center')
                else:
                    ax.text(0.5, 0.5, 'No Data', 
                           horizontalalignment='center',
                           verticalalignment='center')
                
                ax.set_title(f'{metric} (n={n})')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logging.error(f"Error creating distribution plots for bucket {bucket}: {str(e)}")
        plt.close('all')  # Clean up figures in case of error
        raise

def create_metrics_df(metrics_data, days, n_value):
    """Create a polars DataFrame for metrics data for a specific n_value."""
    try:
        data_dict = {"day": days}
        
        # Add metrics data for all days, using None for missing values
        for metric in metrics_data:
            data_dict[metric] = metrics_data[metric][n_value]
                
        return pl.DataFrame(data_dict)
            
    except Exception as e:
        logging.error(f"Error creating DataFrame for n_value {n_value}: {str(e)}")
        raise

def analyze_metrics_distributions(results_dir="data/likelihood/results", 
                                output_dir="data/likelihood/distribution_plots",
                                parquet_dir="data/likelihood/distribution_parquets"):
    """Analyze and visualize metrics distributions across stocks and types."""
    
    # Setup logging
    log_dir = "data/likelihood/logs"
    os.makedirs(log_dir, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=os.path.join(log_dir, f"metrics_analysis_{current_date}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting metrics distribution analysis")
    
    try:
        results_dir = Path(results_dir)
        output_dir = Path(output_dir)
        parquet_dir = Path(parquet_dir)
        
        for dir_path in [output_dir, parquet_dir]:
            dir_path.mkdir(exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
        
        json_files = list(results_dir.glob("*_metrics.json"))
        stocks = list(set([f.stem.split("_")[0] for f in json_files]))
        
        logging.info(f"Found {len(json_files)} JSON files for {len(stocks)} stocks")
        
        types = ["Add", "Trade", "Cancel"]
        buckets = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75]
        n_values = [10, 20]
        
        metrics = ["n_points", "mean", "std", "min", "max", "ks_statistic", "ks_pvalue",
                  "n_clusters", "noise_points", "entropy", "mean_nearest_neighbor", 
                  "max_nearest_neighbor", "max_gap", "mean_gap", "gap_std", "discrepancy"]

        for stock in tqdm(stocks, desc="Processing stocks"):
            logging.info(f"Processing stock: {stock}")
            
            for type_ in tqdm(types, desc="Processing types"):
                logging.info(f"Processing type: {type_}")
                
                for bucket in tqdm(buckets, desc="Processing buckets"):
                    logging.info(f"Processing bucket: {bucket}")
                    
                    try:
                        metrics_data, days = load_metrics_data(json_files, stock, type_, bucket, n_values, metrics)
                        
                        if days:  # Only proceed if we have data
                            fig = create_distribution_plots(metrics_data, n_values, metrics, bucket, stock, type_)
                            
                            for n in n_values:
                                df = create_metrics_df(metrics_data, days, n)
                                output_name = f"{stock}_{type_}_bucket_{bucket:.2f}_n_{n}"
                                df.write_parquet(parquet_dir / f"{output_name}.parquet")
                                logging.info(f"Saved parquet file: {output_name}.parquet")
                            
                            plot_file = output_dir / f"{stock}_{type_}_bucket_{bucket:.2f}_distributions.png"
                            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                            logging.info(f"Saved plot: {plot_file}")
                            plt.close(fig)
                        else:
                            logging.warning(f"No data found for {stock} {type_} bucket {bucket}")
                            
                    except Exception as e:
                        logging.error(f"Error processing bucket {bucket}: {str(e)}")
                
    except Exception as e:
        logging.error(f"Fatal error in analyze_metrics_distributions: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_metrics_distributions()
