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
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%



# %%

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load JSON files
condition = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/condition.json')
manifest = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/dbn/manifest.json')
metadata = load_json('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/metadata.json')

# Function to load CSV data
def load_csv(stock):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/{stock}/20240624.csv'
    return pd.read_csv(file_path)

# Load data for each stock (1/10th of the dataset)
stocks = ['ASAI', 'CGAU', 'HL', 'RIOT']
data = {stock: load_csv(stock).sample(frac=0.5, random_state=1) for stock in stocks}

# Sort data by 'ts_event' for each stock
for stock in stocks:
    data[stock].sort_values(by='ts_event', inplace=True)


# %%
# Create a plotly figure for RIOT
import plotly.graph_objects as go

# Get RIOT data and filter for time range
df = data['RIOT'].copy()

# Convert ts_event to datetime
df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')

# Filter for 13:00-20:30 time range
df = df[(df['ts_event'].dt.hour >= 13) & 
        ((df['ts_event'].dt.hour < 20) | 
         ((df['ts_event'].dt.hour == 20) & (df['ts_event'].dt.minute <= 30)))]

# Sample 1/10th of the data
df = df.sample(frac=0.1, random_state=42)

# Sort by timestamp
df = df.sort_values('ts_event')

# Calculate mid price
df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2

# Create figure
fig = go.Figure()

# Add mid price line
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['mid_price'],
    mode='lines',
    name='Mid Price',
    line=dict(color='black', width=0.5)
))

# Add best bid price
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['bid_px_00'],
    mode='markers',
    name='Best Bid Price',
    marker=dict(color='blue', size=5)
))

# Add best ask price
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['ask_px_00'],
    mode='markers', 
    name='Best Ask Price',
    marker=dict(color='red', size=5)
))

# Add second best bid size
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['bid_sz_01'],
    mode='markers',
    name='Second Best Bid Size',
    marker=dict(
        color='blue',
        size=df['bid_sz_01'].apply(lambda x: 0.3*x**0.5),
        opacity=0.3
    )
))

# Add second best ask size  
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['ask_sz_01'],
    mode='markers',
    name='Second Best Ask Size', 
    marker=dict(
        color='red',
        size=df['ask_sz_01'].apply(lambda x: 0.3*x**0.5),
        opacity=0.3
    )
))

# Add third best bid price
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['bid_px_02'],
    mode='markers',
    name='Third Best Bid Price',
    marker=dict(color='blue', size=5, opacity=0.5)
))

# Add third best ask price
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['ask_px_02'],
    mode='markers',
    name='Third Best Ask Price',
    marker=dict(color='red', size=5, opacity=0.5)
))

# Update layout
fig.update_layout(
    title='RIOT Price Data (13:00-20:30)',
    xaxis_title='Time',
    yaxis_title='Price',
    xaxis_range=[df['ts_event'].iloc[0], df['ts_event'].iloc[-1]],
    yaxis_range=[df['mid_price'].min(), df['mid_price'].max()]
)

# Show plot
fig.show()


# %%
import datetime

# Function to load CSV data for a given date
def load_csv_for_date(stock, date):
    file_path = f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/data/csv/{stock}/{date}.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    return pd.read_csv(file_path)

# Generate a list of dates for June, July, and specific days in August
june_dates = [(datetime.datetime.strptime("20240624", "%Y%m%d") + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(5)]
july_dates = [(datetime.datetime.strptime("20240701", "%Y%m%d") + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(31)]
august_dates = ["20240801", "20240802", "20240805", "20240806", "20240807", "20240808"]

# Combine all dates into one list
date_list = june_dates + july_dates + august_dates

# Load data for each stock and each date
data = {}
for stock in stocks:
    stock_data = []
    for date in date_list:
        try:
            stock_data.append(load_csv_for_date(stock, date).sample(frac=0.1, random_state=1))
        except FileNotFoundError as e:
            print(e)
    data[stock] = pd.concat(stock_data)

# Sort data by 'ts_event' for each stock
for stock in stocks:
    data[stock].sort_values(by='ts_event', inplace=True)


# %%

# Create a figure with subplots for each stock
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for i, stock in enumerate(tqdm(stocks, desc="Processing stocks")):
    df = data[stock]
    row = i // 2
    col = i % 2

    # Convert ts_event to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')

    # Calculate the middle price
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2

    # Plot the middle price with a very thin line
    axs[row, col].plot(df['ts_event'], df['mid_price'], label='Mid Price', color='black', linewidth=0.5)

    # Plot the best bid price with a scatter plot
    axs[row, col].scatter(df['ts_event'], df['bid_px_00'], label='Best Bid Price', color='blue', s=10)

    # Plot the best ask price with a scatter plot
    axs[row, col].scatter(df['ts_event'], df['ask_px_00'], label='Best Ask Price', color='red', s=10)

    # Plot the second best bid size
    axs[row, col].scatter(df['ts_event'], df['bid_sz_01'], s=0.3*df['bid_sz_01']**0.5, color='blue', alpha=0.3, label='Second Best Bid Size')

    # Plot the second best ask size
    axs[row, col].scatter(df['ts_event'], df['ask_sz_01'], s=0.3*df['ask_sz_01']**0.5, color='red', alpha=0.3, label='Second Best Ask Size')

    # Plot the third best bid price (red transparent line)
    axs[row, col].scatter(df['ts_event'], df['bid_px_02'], label='Third Best Bid Price', color='blue', alpha=0.5, s=10)

    # Plot the third best ask price (red transparent line)
    axs[row, col].scatter(df['ts_event'], df['ask_px_02'], label='Third Best Ask Price', color='red', alpha=0.5, s=10)

    # Update axes
    axs[row, col].set_title(stock)
    axs[row, col].set_xlabel("Time")
    axs[row, col].set_ylabel("Price")
    axs[row, col].set_xlim(df['ts_event'].iloc[0], df['ts_event'].iloc[-1])
    axs[row, col].set_ylim(df['mid_price'].min(), df['mid_price'].max())
    axs[row, col].legend()

# Update layout
fig.tight_layout()

# Save the final plot as PNG
fig.savefig('/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/analyze/vizualize/plots/final_plot.png')

# Show the plot
plt.show()

# Plot each day separately and save as PNG
for date in date_list:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for i, stock in enumerate(stocks):
        df = load_csv_for_date(stock, date)
        row = i // 2
        col = i % 2

        # Convert ts_event to datetime
        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')

        # Calculate the middle price
        df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2

        # Plot the middle price with a very thin line
        axs[row, col].plot(df['ts_event'], df['mid_price'], label='Mid Price', color='black', linewidth=0.5)

        # Plot the best bid price with a scatter plot
        axs[row, col].scatter(df['ts_event'], df['bid_px_00'], label='Best Bid Price', color='blue', s=10)

        # Plot the best ask price with a scatter plot
        axs[row, col].scatter(df['ts_event'], df['ask_px_00'], label='Best Ask Price', color='red', s=10)

        # Plot the second best bid size
        axs[row, col].scatter(df['ts_event'], df['bid_sz_01'], s=0.3*df['bid_sz_01']**0.5, color='blue', alpha=0.3, label='Second Best Bid Size')

        # Plot the second best ask size
        axs[row, col].scatter(df['ts_event'], df['ask_sz_01'], s=0.3*df['ask_sz_01']**0.5, color='red', alpha=0.3, label='Second Best Ask Size')

        # Plot the third best bid price (red transparent line)
        axs[row, col].scatter(df['ts_event'], df['bid_px_02'], label='Third Best Bid Price', color='blue', alpha=0.5, s=10)

        # Plot the third best ask price (red transparent line)
        axs[row, col].scatter(df['ts_event'], df['ask_px_02'], label='Third Best Ask Price', color='red', alpha=0.5, s=10)

        # Update axes
        axs[row, col].set_title(stock)
        axs[row, col].set_xlabel("Time")
        axs[row, col].set_ylabel("Price")
        axs[row, col].set_xlim(df['ts_event'].iloc[0], df['ts_event'].iloc[-1])
        axs[row, col].set_ylim(df['mid_price'].min(), df['mid_price'].max())
        axs[row, col].legend()

    # Update layout
    fig.tight_layout()


    # Save the daily plot as PNG
    fig.savefig(f'/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/data/smash2/analyze/vizualize/plots/{date}_plot.png')

    # Show the plot
    plt.show()


# %%

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
