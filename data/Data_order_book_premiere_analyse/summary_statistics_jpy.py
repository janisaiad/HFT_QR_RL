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
import pandas as pd


# %% [markdown]
# ## Data Fields Description
#
# | Field Name       | Data Type  | Description                                                                                       |
# |------------------|------------|---------------------------------------------------------------------------------------------------|
# | publisher_id     | uint16_t   | The publisher ID assigned by Databento, which denotes the dataset and venue.                      |
# | instrument_id    | uint32_t   | The numeric instrument ID.                                                                        |
# | ts_event         | uint64_t   | The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch.|
# | price            | int64_t    | The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001.       |
# | size             | uint32_t   | The order quantity.                                                                               |
# | action           | char       | The event action. Always Trade in the TBBO schema. See Action.                                    |
# | side             | char       | The side that initiates the event. Can be Ask for a sell aggressor, Bid for a buy aggressor, or None where no side is specified by the original trade.|
# | flags            | uint8_t    | A bit field indicating event end, message characteristics, and data quality. See Flags.           |
# | depth            | uint8_t    | The book level where the update event occurred.                                                   |
# | ts_recv          | uint64_t   | The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch.|
# | ts_in_delta      | int32_t    | The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv.      |
# | sequence         | uint32_t   | The message sequence number assigned at the venue.                                                |
# | bid_px_00        | int64_t    | The bid price at the top level.                                                                   |
# | ask_px_00        | int64_t    | The ask price at the top level.                                                                   |
# | bid_sz_00        | uint32_t   | The bid size at the top level.                                                                    |
# | ask_sz_00        | uint32_t   | The ask size at the top level.                                                                    |
# | bid_ct_00        | uint32_t   | The number of bid orders at the top level.                                                        |
# | ask_ct_00        | uint32_t   | The number of ask orders at the top level.                                                        |
#

# %%
import matplotlib.pyplot as plt

# Load the TBBO.csv file
tbbo_data = pd.read_csv("Data_csv/TBBO.csv")

# Filter data for publisher_id 39 and 40
filtered_data = tbbo_data[tbbo_data["publisher_id"].isin([39, 40])]

# Group the data by symbol
grouped_data = filtered_data.groupby("symbol")

# Function to calculate imbalance and plot mean mid-price for each bucket
def analyze_symbol(data):
    # Remove points where bid or ask sizes are 0
    data = data[(data["bid_sz_00"] != 0) | (data["ask_sz_00"] != 0)]
    
    # Calculate the bid-ask imbalance
    data["imbalance"] = (data["bid_sz_00"] - data["ask_sz_00"]) / (data["bid_sz_00"] + data["ask_sz_00"])

    # Calculate the mid-price
    data["mid_price"] = (data["bid_px_00"] + data["ask_px_00"]) / 2

    # Create buckets for imbalance
    data["imbalance_bucket"] = (data["imbalance"] * 10).round() / 10

    # Calculate mean mid-price for each bucket
    bucketed_data = data.groupby("imbalance_bucket")["mid_price"].mean().reset_index()

    # Plot the mean mid-price for each bucket
    plt.plot(bucketed_data["imbalance_bucket"], bucketed_data["mid_price"], marker='o')
    plt.xlabel("Imbalance Bucket")
    plt.ylabel("Mean Mid Price")
    plt.title(f"Imbalance vs Mean Mid Price for {data['symbol'].iloc[0]}")
    plt.show()

# Apply the analysis function to each symbol
for symbol, data in grouped_data:
    analyze_symbol(data)    



# %%
# Function to plot imbalance curves for a small part of the dataset
def plot_imbalance_curves_small_sample(grouped_data, sample_size=100):
    for symbol, data in grouped_data:
        plt.figure(figsize=(12, 8))
        
        # Take a small sample of the data
        data_sample = data.sample(n=sample_size, random_state=1)
        
        # Calculate the bid-ask imbalance
        data_sample["imbalance"] = (data_sample["bid_sz_00"] - data_sample["ask_sz_00"]) / (data_sample["bid_sz_00"] + data_sample["ask_sz_00"])
        
        # Check if 'ts_recv' column exists
        if "ts_recv" in data_sample.columns:
            # Plot the imbalance as a function of time
            plt.plot(data_sample["ts_recv"], data_sample["imbalance"], label=symbol)
        else:
            print(f"Warning: 'ts_recv' column not found for symbol {symbol}")
        
        plt.xlabel("Time")
        plt.ylabel("Imbalance")
        plt.title(f"Imbalance Curve for {symbol} Over Time")
        plt.legend()
        plt.show()

# Call the function to plot imbalance curves for a small sample
plot_imbalance_curves_small_sample(grouped_data)


# %%
import pandas as pd

def calculate_descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics and liquidity indicators for the given dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset containing order book data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    stats = {}
    
    # Calculate bid-ask spread
    data["bid_ask_spread"] = data["ask_px_00"] - data["bid_px_00"]
    
    # Calculate mid price
    data["mid_price"] = (data["ask_px_00"] + data["bid_px_00"]) / 2
    
    # Calculate mean and standard deviation of bid-ask spread
    stats["mean_bid_ask_spread"] = data["bid_ask_spread"].mean()
    stats["std_bid_ask_spread"] = data["bid_ask_spread"].std()
    
    # Calculate mean and standard deviation of mid price
    stats["mean_mid_price"] = data["mid_price"].mean()
    stats["std_mid_price"] = data["mid_price"].std()
    
    # Calculate mean and standard deviation of bid size and ask size
    stats["mean_bid_size"] = data["bid_sz_00"].mean()
    stats["std_bid_size"] = data["bid_sz_00"].std()
    stats["mean_ask_size"] = data["ask_sz_00"].mean()
    stats["std_ask_size"] = data["ask_sz_00"].std()
    
    # Calculate mean and standard deviation of bid count and ask count
    stats["mean_bid_count"] = data["bid_ct_00"].mean()
    stats["std_bid_count"] = data["bid_ct_00"].std()
    stats["mean_ask_count"] = data["ask_ct_00"].mean()
    stats["std_ask_count"] = data["ask_ct_00"].std()
    
    # Calculate volume
    stats["total_volume"] = data["size"].sum()
    
    return pd.DataFrame([stats])

# Apply the descriptive statistics function to each symbol
all_stats = []
for symbol, data in grouped_data:
    symbol_stats = calculate_descriptive_statistics(data)
    symbol_stats["symbol"] = symbol
    all_stats.append(symbol_stats)

# Concatenate all statistics into a single DataFrame
all_stats_df = pd.concat(all_stats, ignore_index=True)

# Display the descriptive statistics for each symbol
print(all_stats_df)

# Determine which stock has the highest and smallest stat for each metric
highest_stats = {}
smallest_stats = {}
for column in all_stats_df.columns:
    if column not in ["symbol"]:
        highest_stat = all_stats_df[column].idxmax()
        smallest_stat = all_stats_df[column].idxmin()
        highest_stats[column] = all_stats_df.loc[highest_stat, "symbol"]
        smallest_stats[column] = all_stats_df.loc[smallest_stat, "symbol"]

# Print the stock with the highest and smallest stat for each metric
for stat in highest_stats.keys():
    print(f"The stock with the highest {stat} is {highest_stats[stat]}")
for stat in smallest_stats.keys():
    print(f"The stock with the smallest {stat} is {smallest_stats[stat]}")


# %%
# Calculate tick size for each stock and each publisher_id
tick_sizes = {}
for symbol, data in grouped_data:
    tick_sizes[symbol] = {}
    for publisher_id in data["publisher_id"].unique():
        publisher_data = data[data["publisher_id"] == publisher_id]
        # Assuming tick size is the minimum price increment
        tick_size = publisher_data["price"].diff().min()
        tick_sizes[symbol][publisher_id] = tick_size

# Sort the tick sizes from highest to smallest for each stock and each publisher_id
sorted_tick_sizes = {}
for symbol, publisher_ticks in tick_sizes.items():
    sorted_tick_sizes[symbol] = dict(sorted(publisher_ticks.items(), key=lambda item: item[1], reverse=True))

# Print the tick size for each stock and each publisher_id
for symbol, publisher_ticks in sorted_tick_sizes.items():
    for publisher_id, tick_size in publisher_ticks.items():
        print(f"The tick size for {symbol} (Publisher {publisher_id}) is {tick_size}")


# %%

# Plot price for all stocks in function of time, a plot for each stock
for symbol, data in grouped_data:
    plt.figure(figsize=(10, 6))
    
    # Ensure the lengths of x and y are the same
    min_length = min(len(data["ts_event"]), len(data["price"]), len(data["bid_px_00"]), len(data["ask_px_00"]))
    
    plt.plot(data["ts_event"][:min_length:30], data["price"][:min_length:30], label=f"Price of {symbol}")
    plt.plot(data["ts_event"][:min_length:30], data["bid_px_00"][:min_length:30], label=f"Bid Price of {symbol}", color='green')
    plt.plot(data["ts_event"][:min_length:30], data["ask_px_00"][:min_length:30], label=f"Ask Price of {symbol}", color='red')
    
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Price of {symbol} over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
import plotly.graph_objects as go

# Plot price for all stocks in function of time, a plot for each stock and each publisher id using Plotly
for symbol, data in grouped_data:
    for publisher_id in data["publisher_id"].unique():
        publisher_data = data[data["publisher_id"] == publisher_id]
        
        fig = go.Figure()
        
        # Ensure the lengths of x and y are the same
        min_length = min(len(publisher_data["ts_event"]), len(publisher_data["price"]), len(publisher_data["bid_px_00"]), len(publisher_data["ask_px_00"]))
        
        fig.add_trace(go.Scatter(x=publisher_data["ts_event"][:min_length:10], y=publisher_data["price"][:min_length:10], mode='lines', name=f"Price of {symbol} (Publisher {publisher_id})"))
        fig.add_trace(go.Scatter(x=publisher_data["ts_event"][:min_length:10], y=publisher_data["bid_px_00"][:min_length:10], mode='lines', name=f"Bid Price of {symbol} (Publisher {publisher_id})", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=publisher_data["ts_event"][:min_length:10], y=publisher_data["ask_px_00"][:min_length:10], mode='lines', name=f"Ask Price of {symbol} (Publisher {publisher_id})", line=dict(color='red')))
        
        fig.update_layout(
            title=f"Price of {symbol} over Time (Publisher {publisher_id})",
            xaxis_title="Time",
            yaxis_title="Price",
            legend_title="Legend",
            template="plotly_white"
        )
        
        fig.show()


# %%
# Calculate the number of trades for each stock
trade_counts = {}
for symbol, data in grouped_data:
    # Assuming each row represents a trade
    trade_count = len(data)
    trade_counts[symbol] = trade_count

# Sort the trade counts from highest to smallest
sorted_trade_counts = dict(sorted(trade_counts.items(), key=lambda item: item[1], reverse=True))

# Print the number of trades for each stock
for symbol, trade_count in sorted_trade_counts.items():
    print(f"The number of trades for {symbol} is {trade_count}")


# %%
