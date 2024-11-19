import numpy as np
from scipy.stats import kstest, entropy
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import os
import glob
import json
from datetime import datetime
import logging
from tqdm import tqdm

def analyze_point_distribution(points_file: str, spread_type: str, eps: float = 0.1, min_samples: int = 5) -> dict:
    """
    Analyze a 1D point distribution to measure how it deviates from uniform.
    
    Args:
        points_file: Path to txt file containing points grouped by action and bucket
        spread_type: Either 'zero_spread' or 'nonzero_spread'
        eps: Distance threshold for DBSCAN clustering
        min_samples: Minimum samples per cluster for DBSCAN
        
    Returns:
        Dictionary containing various distribution metrics
    """
    # Parse points by action and bucket
    points_by_category = {}
    current_action = None
    current_bucket = None
    try:
        with open(points_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Action:'):
                    parts = line.strip().split(',')
                    action = parts[0].split(':')[1].strip()
                    bucket = parts[1].split('[')[1].split(']')[0].strip()
                    current_action = action
                    current_bucket = bucket
                    
                    if current_action not in points_by_category:
                        points_by_category[current_action] = {}
                    if current_bucket not in points_by_category[current_action]:
                        points_by_category[current_action][current_bucket] = []
                        
                elif line.strip() and current_action and current_bucket:
                    try:
                        if ',' in line:
                            timestamp, quantile = line.strip().split(',')
                            points_by_category[current_action][current_bucket].append(
                                (float(timestamp), float(quantile))
                            )
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
    
    # Create output directories with spread type
    plot_dir = f"/home/janis/3A/EA/HFT_QR_RL/data/likelihood/metrics_plots/{stock}/{file_date}/{spread_type}"
    os.makedirs(plot_dir, exist_ok=True)

    # Sample sizes to analyze
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000]

    # Calculate metrics for each action/bucket combination and sample size
    metrics = {}
    for action in tqdm(points_by_category):
        metrics[action] = {}
        
        for bucket in points_by_category[action]:
            points_data = points_by_category[action][bucket]
            metrics[action][bucket] = {}
            
            # Sort by quantile (already sorted but just to be sure)
            points_data.sort(key=lambda x: x[1], reverse=True)
            
            for n_points in sample_sizes:
                if n_points > len(points_data):
                    break
                    
                # Extract timestamps for this sample size
                points = np.array([p[0] for p in points_data[:n_points]])
                
                metrics[action][bucket][f'n_{n_points}'] = {}
                sample_metrics = metrics[action][bucket][f'n_{n_points}']
                
                # Normalize points for analysis
                points_norm = (points - np.min(points)) / (np.max(points) - np.min(points))
                
                # Basic statistics
                sample_metrics['n_points'] = n_points
                sample_metrics['mean'] = float(np.mean(points))
                sample_metrics['std'] = float(np.std(points))
                sample_metrics['min'] = float(np.min(points))
                sample_metrics['max'] = float(np.max(points))
                
                # KS test against uniform
                ks_stat, ks_pval = kstest(points_norm, 'uniform')
                sample_metrics['ks_statistic'] = float(ks_stat)
                sample_metrics['ks_pvalue'] = float(ks_pval)
                
                # Clustering analysis
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_norm.reshape(-1,1))
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                sample_metrics['n_clusters'] = int(n_clusters)
                sample_metrics['noise_points'] = int(np.sum(clustering.labels_ == -1))
                
                # Distribution entropy
                hist, _ = np.histogram(points_norm, bins=50, density=True)
                sample_metrics['entropy'] = float(entropy(hist + 1e-10))
                
                # Nearest neighbor stats
                dists = pdist(points_norm.reshape(-1,1))
                sample_metrics['mean_nearest_neighbor'] = float(np.mean(dists))
                sample_metrics['max_nearest_neighbor'] = float(np.max(dists))
                
                # Gap analysis
                gaps = np.diff(np.sort(points_norm))
                sample_metrics['max_gap'] = float(np.max(gaps))
                sample_metrics['mean_gap'] = float(np.mean(gaps))
                sample_metrics['gap_std'] = float(np.std(gaps))
                
                # CDF discrepancy
                sorted_points = np.sort(points_norm)
                uniform_cdf = np.linspace(0, 1, n_points)
                empirical_cdf = np.arange(1, n_points + 1) / n_points
                sample_metrics['discrepancy'] = float(np.sqrt(np.mean((empirical_cdf - uniform_cdf) ** 2)))
                
                # Generate plots with spread type in filename
                plt.figure(figsize=(10,6))
                plt.step(sorted_points, empirical_cdf)
                plt.title(f'Empirical CDF - {action} - Bucket {bucket} - N={n_points} ({spread_type})')
                plt.xlabel('Value')
                plt.ylabel('Cumulative Probability')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plot_dir, f'{stock}_{file_date}_{spread_type}_{action}_{bucket}_n{n_points}_distribution.png'))
                plt.close()
                
                plt.figure(figsize=(10,6))
                plt.scatter(points_norm, np.zeros_like(points_norm), c=clustering.labels_)
                plt.title(f'DBSCAN Clustering - {action} - Bucket {bucket} - N={n_points} ({spread_type})')
                plt.xlabel('Normalized Value')
                plt.savefig(os.path.join(plot_dir, f'{stock}_{file_date}_{spread_type}_{action}_{bucket}_n{n_points}_clusters.png'))
                plt.close()
    
    return metrics

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
        filename=os.path.join(log_dir, f"points_analysis_{current_date}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Process all txt files for both spread types
    for spread_type in ['zero_spread', 'nonzero_spread']:
        txt_files = glob.glob(os.path.join(txt_dir, f"GOOGL_*_{spread_type}_outliers.txt"))
        
        for txt_file in tqdm(txt_files):
            logging.info(f"Processing {spread_type} file: {txt_file}")
            
            filename = os.path.basename(txt_file)
            stock_date = '_'.join(filename.split('_')[:2])  # Get stock and date
            
            try:
                metrics = analyze_point_distribution(txt_file, spread_type)
                
                # Include spread type in output filename
                output_file = os.path.join(results_dir, f"{stock_date}_{spread_type}_metrics.json")
                with open(output_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                    
                logging.info(f"Successfully processed {filename}")
                logging.info(f"Metrics saved to {output_file}")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                continue