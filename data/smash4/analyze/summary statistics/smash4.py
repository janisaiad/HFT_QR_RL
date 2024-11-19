import os
import polars as pl
import glob
from pathlib import Path

def analyze_parquet_files(base_path):
    """
    Analyze parquet files in given folders and generate summary statistics using Polars with GPU acceleration
    
    Args:
        base_path (str): Base path containing the folders to analyze
    """
    # Enable GPU acceleration
    pl.Config.set_streaming_chunk_size(50_000)
    pl.Config.set_gpu_acceleration(True)
    
    # List of folders to analyze
    folders = ['CSX', 'INTC', 'WBD']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder} not found. Skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"Analysis for {folder}")
        print(f"{'='*50}\n")
        
        # Find all parquet files in the folder
        parquet_files = glob.glob(os.path.join(folder_path, f"{folder}_*.parquet"))
        
        if not parquet_files:
            print(f"No parquet files found in {folder}")
            continue
            
        # Read and concatenate all parquet files using Polars
        dfs = []
        for file in parquet_files:
            df = pl.read_parquet(file)
            dfs.append(df)
        
        combined_df = pl.concat(dfs)
        
        # Basic dataset information
        print(f"Total number of records: {combined_df.height}")
        print(f"Number of files analyzed: {len(parquet_files)}")
        print("\nColumns in the dataset:")
        print(combined_df.columns)
        
        # Analyze each column
        for column in combined_df.columns:
            print(f"\nAnalysis for column: {column}")
            print("-" * 30)
            
            # Get data type
            dtype = combined_df[column].dtype
            print(f"Data type: {dtype}")
            
            # For numeric columns
            if pl.datatypes.is_numeric(dtype):
                print("\nNumeric Statistics:")
                stats = combined_df.select([
                    pl.col(column).count().alias("count"),
                    pl.col(column).mean().alias("mean"),
                    pl.col(column).std().alias("std"),
                    pl.col(column).min().alias("min"),
                    pl.col(column).quantile(0.25).alias("25%"),
                    pl.col(column).median().alias("50%"),
                    pl.col(column).quantile(0.75).alias("75%"),
                    pl.col(column).max().alias("max")
                ]).collect()
                print(stats)
            
            # Value counts and percentages
            value_counts = (combined_df
                .select(pl.col(column))
                .groupby(column)
                .count()
                .sort("count", descending=True)
                .with_columns([
                    (pl.col("count") / combined_df.height * 100).alias("percentage")
                ])
                .limit(10)
                .collect()
            )
            
            print("\nTop 10 most common values:")
            for row in value_counts.iter_rows():
                value, count, percentage = row
                print(f"{value}: {count} occurrences ({percentage:.2f}%)")
            
            # Missing values
            missing_count = combined_df.select(pl.col(column).null_count()).collect()[0,0]
            missing_percentage = (missing_count / combined_df.height) * 100
            print(f"\nMissing values: {missing_count} ({missing_percentage:.2f}%)")
            
        # Additional analysis
        print("\nCorrelation analysis (for numeric columns):")
        numeric_cols = [col for col in combined_df.columns if pl.datatypes.is_numeric(combined_df[col].dtype)]
        if len(numeric_cols) > 1:
            correlation_matrix = combined_df.select(numeric_cols).corr().collect()
            print(correlation_matrix)
            
        # Time-based analysis if datetime columns exist
        datetime_cols = [col for col in combined_df.columns if pl.datatypes.is_temporal(combined_df[col].dtype)]
        if len(datetime_cols) > 0:
            print("\nTemporal analysis:")
            for dt_col in datetime_cols:
                stats = combined_df.select([
                    pl.col(dt_col).min().alias("earliest_date"),
                    pl.col(dt_col).max().alias("latest_date"),
                    (pl.col(dt_col).max() - pl.col(dt_col).min()).alias("date_range")
                ]).collect()
                
                print(f"\nTimeline analysis for {dt_col}:")
                print(f"Earliest date: {stats[0,'earliest_date']}")
                print(f"Latest date: {stats[0,'latest_date']}")
                print(f"Date range: {stats[0,'date_range']}")

if __name__ == "__main__":
    # Assuming the parquet files are in the same directory as this script
    current_dir = Path(__file__).parent
    analyze_parquet_files(current_dir)
