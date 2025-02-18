import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def scan_database():
    """
    Scan the MBP_10 database folder to get list of stocks and dates by looking at parquet files.
    Creates a JSON file with results and generates a chronological plot of stocks.
    Returns:
        stocks (list): List of stock symbols found
        dates (list): List of unique dates found in parquet files
    """
    # Base path
    base_path = Path('/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10')
    results_path = Path('/home/janis/3A/EA/HFT_QR_RL/data/smash4/analyze/results')
    
    # Create results directory if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of stock folders
    stocks = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    # Get list of parquet files and extract dates
    dates = set()
    stock_dates = {stock: [] for stock in stocks}
    
    for stock in stocks:
        stock_path = base_path / stock
        parquet_files = list(stock_path.glob('*.parquet'))
        for pq_file in parquet_files:
            # Extract date from filename (format: SYMBOL_YYYY-MM-DD.parquet)
            date = pq_file.stem.split('_')[1]
            dates.add(date)
            stock_dates[stock].append(date)
    
    dates = sorted(list(dates))
    
    # Save results to JSON
    results = {
        'stocks': stocks,
        'dates': dates,
        'stock_dates': stock_dates,
        'summary': {
            'total_stocks': len(stocks),
            'total_dates': len(dates)
        }
    }
    
    with open(results_path / 'database_scan_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create chronological plot
    plt.figure(figsize=(15, 8))
    
    # Convert dates to pandas datetime for better plotting
    date_range = pd.to_datetime(dates)
    
    for i, stock in enumerate(stocks):
        stock_date_points = pd.to_datetime(stock_dates[stock])
        plt.scatter([d for d in stock_date_points], 
                   [i] * len(stock_date_points),
                   marker='|', 
                   label=stock)

    plt.yticks(range(len(stocks)), stocks)
    plt.xlabel('Date')
    plt.ylabel('Stocks')
    plt.title('Chronological Presence of Stocks in Database')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(results_path / 'stocks_chronological_plot.png')
    plt.close()
    
    print(f"Found {len(stocks)} stocks")
    print(f"Found {len(dates)} dates")
    print(f"Results saved to {results_path}")
    
    return stocks, dates

if __name__ == '__main__':
    stocks, dates = scan_database()
    print("\nStocks:", stocks)
    print("\nDates:", dates)
