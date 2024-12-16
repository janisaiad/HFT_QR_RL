# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import os
import json
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl


# %%

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load JSON files
condition = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/dbn/condition.json')
manifest = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/dbn/manifest.json')
metadata = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/csv/metadata.json')
# Function to load CSV data
def load_csv(stock):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/csv/{stock}/20240722.csv'
    return pl.read_csv(file_path)

# Load data for each stock (1/10th of the dataset)
stocks = ['LCID']
data = {stock: load_csv(stock).sample(n=load_csv(stock).height // 2, seed=1) for stock in stocks}

# Sort data by 'ts_event' for each stock
for stock in stocks:
    data[stock] = data[stock].sort('ts_event')


# %%
data['LCID'].head()

# %%
# Create an interactive plotly figure
fig = go.Figure()

for stock in tqdm(stocks, desc="Processing stocks"):
    df = data[stock]
    
    # Convert ts_event to datetime and calculate mid price using polars
    df = df.with_columns([
        pl.col('ts_event').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f%z").alias('ts_event'),
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

    # Add best bid price
    fig.add_trace(go.Scatter(
        x=df.get_column('ts_event'),
        y=df.get_column('bid_px_00'),
        mode='lines',
        name='Best Bid',
        line=dict(color='green', width=1)
    ))

    # Add best ask price
    fig.add_trace(go.Scatter(
        x=df.get_column('ts_event'),
        y=df.get_column('ask_px_00'),
        mode='lines',
        name='Best Ask',
        line=dict(color='red', width=1)
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

# Update layout
fig.update_layout(
    title=f"Order Book Visualization for {stock}",
    xaxis_title="Time",
    yaxis_title="Price",
    showlegend=True,
    width=1200,
    height=800
)

# Show the plot
fig.show()


# %%
import datetime

# Function to load CSV data for a given date
def load_csv_for_date(stock, date):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash3/data/csv/{stock}/{date}.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    return pl.read_csv(file_path)

# Generate a list of dates for June, July, and specific days in August
june_dates = [(datetime.datetime.strptime("20240722", "%Y%m%d") + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(7)]
july_dates = [(datetime.datetime.strptime("20240801", "%Y%m%d") + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(31)]
august_dates = ["20240901", "20240902", "20240903", "20240904", "20240905", "20240906"]

# Combine all dates into one list
date_list = june_dates + july_dates + august_dates
# Load data for each stock and each date
data = {}
for stock in stocks:
    stock_data = []
    for date in date_list:
        try:
            # Use sample_n instead of sample(frac) for polars DataFrame
            df = load_csv_for_date(stock, date)
            sample_size = int(len(df) * 0.1)
            stock_data.append(df.sample(n=sample_size, seed=1))
        except FileNotFoundError as e:
            print(e)
    data[stock] = pl.concat(stock_data)

# Sort data by 'ts_event' for each stock
for stock in stocks:
    data[stock] = data[stock].sort('ts_event')


# %%
# Create an interactive plotly figure
fig = go.Figure()

for stock in tqdm(stocks, desc="Processing stocks"):
    df = data[stock]
    
    # Convert ts_event to datetime and calculate mid price using polars
    df = df.with_columns([
        pl.col('ts_event').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f%z").alias('ts_event'),
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

    # Add best bid price
    fig.add_trace(go.Scatter(
        x=df.get_column('ts_event'),
        y=df.get_column('bid_px_00'),
        mode='lines',
        name='Best Bid',
        line=dict(color='green', width=1)
    ))

    # Add best ask price
    fig.add_trace(go.Scatter(
        x=df.get_column('ts_event'),
        y=df.get_column('ask_px_00'),
        mode='lines',
        name='Best Ask',
        line=dict(color='red', width=1)
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

# Update layout
fig.update_layout(
    title=f"Order Book Visualization for {stock}",
    xaxis_title="Time",
    yaxis_title="Price",
    showlegend=True,
    width=1200,
    height=800
)

# Show the plot
fig.show()

# Plot each day separately
for date in date_list:
    fig = go.Figure()
    
    for stock in stocks:
        df = load_csv_for_date(stock, date)
        
        # Convert ts_event to datetime and calculate mid price using polars
        df = pl.DataFrame(df).with_columns([
            pl.col('ts_event').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f%z").alias('ts_event'),
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

        # Add best bid price
        fig.add_trace(go.Scatter(
            x=df.get_column('ts_event'),
            y=df.get_column('bid_px_00'),
            mode='lines',
            name='Best Bid',
            line=dict(color='green', width=1)
        ))

        # Add best ask price
        fig.add_trace(go.Scatter(
            x=df.get_column('ts_event'),
            y=df.get_column('ask_px_00'),
            mode='lines',
            name='Best Ask',
            line=dict(color='red', width=1)
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

    # Update layout
    fig.update_layout(
        title=f"Order Book Visualization for {date}",
        xaxis_title="Time",
        yaxis_title="Price", 
        showlegend=True,
        width=1200,
        height=800
    )

    # Show the plot
    fig.show()


# %%

# %%
import plotly.graph_objects as go

# Create a figure with subplots for each stock
fig = make_subplots(rows=2, cols=2, subplot_titles=stocks)

for i, stock in enumerate(stocks):
    df = data[stock]
    row = i // 2 + 1
    col = i % 2 + 1

    # Convert ts_event to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')

    # Calculate the middle price
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2

    # Plot the middle price with a very thin line
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['mid_price'], mode='lines', name='Mid Price', line=dict(color='black', width=0.5)), row=row, col=col)

    # Plot the best bid price with a scatter plot
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['bid_px_00'], mode='markers', name='Best Bid Price', marker=dict(color='blue', size=5)), row=row, col=col)

    # Plot the best ask price with a scatter plot
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['ask_px_00'], mode='markers', name='Best Ask Price', marker=dict(color='red', size=5)), row=row, col=col)

    # Plot the second best bid size
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['bid_sz_01'], mode='markers', name='Second Best Bid Size', marker=dict(color='blue', size=5, opacity=0.3)), row=row, col=col)

    # Plot the second best ask size
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['ask_sz_01'], mode='markers', name='Second Best Ask Size', marker=dict(color='red', size=5, opacity=0.3)), row=row, col=col)

    # Plot the third best bid price
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['bid_px_02'], mode='markers', name='Third Best Bid Price', marker=dict(color='blue', size=5, opacity=0.5)), row=row, col=col)

    # Plot the third best ask price
    fig.add_trace(go.Scatter(x=df['ts_event'], y=df['ask_px_02'], mode='markers', name='Third Best Ask Price', marker=dict(color='red', size=5, opacity=0.5)), row=row, col=col)

    # Update axes
    fig.update_xaxes(title_text="Time", row=row, col=col)
    fig.update_yaxes(title_text="Price", row=row, col=col)

# Update layout
fig.update_layout(title_text="Stock Prices", height=800, width=1200, showlegend=True)

# Show the plot
fig.show()


# %%

# %%

# %%
