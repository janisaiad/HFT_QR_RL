import os
import json
import polars as pl
from tqdm import tqdm
import altair as alt

# Function to load parquet data
def load_parquet(stock, date):
    file_path = f'/home/janis/3A/EA/HFT_QR_RL/data/smash4/parquet/{stock}/{stock}_{date}.parquet'
    return pl.read_parquet(file_path)

# Specify dates and stocks
dates = ["2024-08-12"]
stocks = ["CSX"]

# Load data for each stock and date
data_dict = {}
for stock in stocks:
    data_dict[stock] = {}
    for date in dates:
        data_dict[stock][date] = load_parquet(stock, date).sample(fraction=1, seed=1)

# Calculate statistics
stats_dict = {}
for stock in stocks:
    stats_dict[stock] = {}
    for date in dates:
        df = data_dict[stock][date]
        
        mid_price = (df['bid_px_00'] + df['ask_px_00']) / 2
        spread = df['ask_px_00'] - df['bid_px_00']
        
        stats = {
            'mean_mid_price': float(mid_price.mean()),
            'std_mid_price': float(mid_price.std()),
            'min_mid_price': float(mid_price.min()),
            'max_mid_price': float(mid_price.max()),
            'mean_spread': float(spread.mean()),
            'std_spread': float(spread.std()),
            'min_spread': float(spread.min()),
            'max_spread': float(spread.max()),
            'total_volume_bid': int(df['bid_ct_00'].sum()),
            'total_volume_ask': int(df['ask_ct_00'].sum()),
            'num_quotes': len(df),
        }
        
        stats_dict[stock][date] = stats

# Print statistics
for stock in stats_dict:
    print(f"\nStatistics for {stock}:")
    for date in stats_dict[stock]:
        print(f"\nDate: {date}")
        for metric, value in stats_dict[stock][date].items():
            if 'price' in metric:
                print(f"{metric}: ${value:.4f}")
            elif 'spread' in metric:
                print(f"{metric}: ${value:.6f}")
            else:
                print(f"{metric}: {value:,}")

# Create visualization for each stock and date
for date in dates:
    for stock in tqdm(stocks, desc="Processing stocks"):
        # Get data and calculate mid price
        df = data_dict[stock][date].sort('ts_event')
        df = df.with_columns([
            ((pl.col('bid_px_00') + pl.col('ask_px_00')) / 2).alias('mid_price')
        ])
        
        # Convert to pandas for Altair
        pdf = df.to_pandas()
        
        # Base chart
        base = alt.Chart(pdf).encode(x='ts_event:T')
        
        # Mid price line
        mid_line = base.mark_line(color='black', size=1).encode(
            y='mid_price:Q',
            tooltip=['ts_event', 'mid_price']
        )
        
        # Best bid with size markers
        bid_points = base.mark_point(color='green', opacity=0.3).encode(
            y='bid_px_00:Q',
            size=alt.Size('bid_sz_00:Q', scale=alt.Scale(range=[20, 200])),
            tooltip=['ts_event', 'bid_px_00', 'bid_sz_00']
        )
        
        # Best ask with size markers  
        ask_points = base.mark_point(color='red', opacity=0.3).encode(
            y='ask_px_00:Q', 
            size=alt.Size('ask_sz_00:Q', scale=alt.Scale(range=[20, 200])),
            tooltip=['ts_event', 'ask_px_00', 'ask_sz_00']
        )
        
        # Second best bid/ask lines
        bid2_line = base.mark_line(color='lightgreen', size=1).encode(
            y='bid_px_01:Q',
            tooltip=['ts_event', 'bid_px_01']
        )
        
        ask2_line = base.mark_line(color='pink', size=1).encode(
            y='ask_px_01:Q',
            tooltip=['ts_event', 'ask_px_01']
        )
        
        # Trades as points
        trades = pdf[pdf['rtype'] == 2]
        trade_points = alt.Chart(trades).mark_point(color='black', size=50).encode(
            x='ts_event:T',
            y='mid_price:Q',
            tooltip=['ts_event', 'mid_price']
        )
        
        # Combine all layers
        chart = alt.layer(
            mid_line, bid_points, ask_points, 
            bid2_line, ask2_line, trade_points
        ).properties(
            width=1200,
            height=800,
            title=f"Order Book Visualization for {stock} on {date}"
        ).configure_axis(
            grid=True
        )
        
        # Create directory and save
        save_dir = f"/home/janis/3A/EA/HFT_QR_RL/data/smash4/altair/{stock}"
        os.makedirs(save_dir, exist_ok=True)
        
        chart.save(f"{save_dir}/orderbook_visualization_{stock}_{date}.html")
