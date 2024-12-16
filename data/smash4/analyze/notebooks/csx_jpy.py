# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import polars as pl
import pygwalker as pyg
import glob
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Get all parquet files in the specified directory
parquet_files = glob.glob('/mnt/beegfs/project/lobib/repos/HFT_QR_RL/data/smash4/parquet/CSX/*')

# Check if the list of parquet files is not empty
if not parquet_files:
    raise ValueError("No parquet files found in the specified directory")

# Read and concatenate all parquet files using polars with GPU support
df = pl.concat([pl.read_parquet(f, use_pyarrow=True) for f in parquet_files])


# %%
print(len(parquet_files))


# %%
# Get schema information from the dataframe
print("DataFrame Schema:")
print(df.schema)

# Get basic statistics about the data
print("\nDataFrame Info:")
print(df.describe())

# Show first few rows to understand the data structure
print("\nFirst few rows:")
print(df.head())


# %%
# Create subplots for each day
fig, axes = plt.subplots(74, 1, figsize=(12, 30))

# Get unique dates
dates = df['ts_event'].dt.date().unique()

for day_idx, date in enumerate(dates):
    ax = axes[day_idx]
    
    # Filter data for current date and time range 14:00 to 20:00
    df_filtered = df.filter(
        (pl.col('ts_event').dt.date() == date) &
        (pl.col('ts_event').dt.hour() >= 14) &
        (pl.col('ts_event').dt.hour() <= 20)
    )

    # Plot bid and ask prices for each level
    for i in range(10):
        bid_col = f"bid_px_{i:02d}"
        ask_col = f"ask_px_{i:02d}"
        bid_size = f"bid_sz_{i:02d}"
        ask_size = f"ask_sz_{i:02d}"
        
        # Plot with alpha decreasing for deeper levels
        alpha = 0.7 * (1 - i/10)
        ax.plot(df_filtered['ts_event'], df_filtered[bid_col], 
                label=f'Bid L{i}' if i==0 else None,
                color='blue', alpha=alpha, linewidth=1)
        ax.plot(df_filtered['ts_event'], df_filtered[ask_col],
                label=f'Ask L{i}' if i==0 else None, 
                color='red', alpha=alpha, linewidth=1)

    # Filter for trades (where side is populated)
    trades = df_filtered.filter(pl.col('side').is_not_null())

    # Calculate weighted size based on order book state
    trades = trades.with_columns([
        (pl.col('bid_sz_00') * pl.col('ask_sz_00') / 
         (pl.col('bid_sz_00') + pl.col('ask_sz_00'))).alias('weighted_size')
    ])

    # Plot trades
    ax.scatter(trades['ts_event'], trades['price'],
              s=trades['weighted_size']/trades['weighted_size'].mean()*100,
              alpha=0.5,
              color='purple',
              label='Trades')

    # Customize plot
    ax.set_title(f'Order Book Depth and Trades - {date}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_ylim(min(df_filtered['bid_px_00']), max(df_filtered['ask_px_00']))  # Set y-axis limits from 30 to 40

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig(f'../CSX/png/orderbook_depth_trades.png')
plt.close()


# %%
