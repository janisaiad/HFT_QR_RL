# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import os
import json
import polars as pl
from tqdm import tqdm


import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.express as px


# +
# Fonction pour charger les données parquet
def load_parquet(stock, date):
    file_path = f'/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10/{stock}/{stock}_{date}.parquet'
    return pl.read_parquet(file_path)
# Spécifier les dates
dates = ["2024-08-12"]

stocks = ["CSX"]

# Charger les données pour chaque stock et chaque date
data_dict = {}
for stock in stocks:
    data_dict[stock] = {}
    for date in dates:
        data_dict[stock][date] = load_parquet(stock, date).sample(fraction=1, seed=1)

# Concaténer toutes les données
data = pl.concat([data_dict[stock][date] for stock in stocks for date in dates])

# Trier par ts_event
data = data.sort("ts_event")


# +
# Calculate basic statistics for each stock and date
stats_dict = {}
for stock in stocks:
    stats_dict[stock] = {}
    for date in dates:
        df = data_dict[stock][date]
        
        # Calculate mid price
        mid_price = (df['bid_px_00'] + df['ask_px_00']) / 2
        
        # Calculate spread
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

# -

# Create an interactive plotly figure
for date in dates:
    for stock in tqdm(stocks, desc="Processing stocks"):
        fig = go.Figure()
        
        # Get data for this stock and date and sort by timestamp
        df = data_dict[stock][date].sort('ts_event')
        
        # Calculate mid price using polars
        df = df.with_columns([
            ((pl.col('bid_px_00') + pl.col('ask_px_00')) / 2).alias('mid_price')
        ])

        # Add mid price line
        fig.add_trace(go.Scatter(
            x=df.get_column('ts_event'),
            y=df.get_column('mid_price'),
            mode='lines',
            name='Mid Price',
            line=dict(color='black', width=1)
        ))

        # Add best bid price with size-proportional markers
        fig.add_trace(go.Scatter(
            x=df.get_column('ts_event'),
            y=df.get_column('bid_px_00'),
            mode='lines+markers',
            name='Best Bid',
            line=dict(color='green', width=1),
            marker=dict(
                size=df.get_column('bid_sz_00'),
                sizeref=2.*max(df.get_column('bid_sz_00'))/17**2,
                sizemode='area',
                color='green',
                opacity=0.3
            )
        ))

        # Add best ask price with size-proportional markers
    
        fig.add_trace(go.Scatter(
            x=df.get_column('ts_event'),
            y=df.get_column('ask_px_00'),
            mode='lines+markers',
            name='Best Ask', 
            line=dict(color='red', width=1),
            marker=dict(
                size=df.get_column('ask_sz_00'),
                sizeref=2.*max(df.get_column('ask_sz_00'))/17**2,
                sizemode='area',
                color='red',
                opacity=0.3
            )
        ))

        # Add second best bid price
        fig.add_trace(go.Scatter(
            x=df.get_column('ts_event'),
            y=df.get_column('bid_px_01'),
            mode='lines',
            name='Second Best Bid',
            line=dict(color='rgba(0,255,0,0.3)', width=1)
        ))

        # Add second best ask price
        fig.add_trace(go.Scatter(
            x=df.get_column('ts_event'),
            y=df.get_column('ask_px_01'),
            mode='lines',
            name='Second Best Ask',
            line=dict(color='rgba(255,0,0,0.3)', width=1)
        ))

        # Add trades as black dots
        trades = df.filter(pl.col('rtype') == 2)  # Assuming rtype 2 indicates trades
        fig.add_trace(go.Scatter(
            x=trades.get_column('ts_event'),
            y=trades.get_column('mid_price'),  # Using mid price for trade points
            mode='markers',
            name='Trades',
            marker=dict(color='black', size=8)
        ))

        # Update layout with fixed x and y ranges
        fig.update_layout(
            title=f"Order Book Visualization for {stock} on {date}",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            width=1200,
            height=800,
            xaxis=dict(
                range=['13:00', '20:00']  # Set x-axis range from 13h to 20h
            ),
            yaxis=dict(
                range=[32, 36]  # Set y-axis range from 32 to 36
            )
        )

        # Create directory if it doesn't exist
        save_dir = f"/home/janis/3A/EA/HFT_QR_RL/data/smash4/plotly/{stock}"
        os.makedirs(save_dir, exist_ok=True)

        # Save the plot as HTML
        fig.write_html(f"{save_dir}/orderbook_visualization_{stock}_{date}.html")
        
        fig.show(renderer="browser")