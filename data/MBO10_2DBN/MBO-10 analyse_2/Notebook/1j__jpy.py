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
import pandas as pd # ca on va devoir bcp l'uiliser
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
import polars as pl

# %%
df = pl.read_parquet('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/KHC/20240927.parquet')
df.head()

# %%
# Filter for KHC symbol and depth 0
df = df.filter(pl.col('symbol') == 'KHC')
df = df.filter(pl.col('depth') == 0)

# Convert columns to numpy arrays
size_bid = df['bid_sz_00'].to_numpy()
size_ask = df['ask_sz_00'].to_numpy()
time = df['ts_event'].dt.time()

# Plot queue sizes
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='Bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# Get trade prices and times
trades = df.filter(pl.col('action') == 'T')
price = trades['price'].to_numpy()
time_price = trades['ts_event'].dt.time()

# Get bid/ask prices
bid_px_00 = df['bid_px_00'].to_numpy()
ask_px_00 = df['ask_px_00'].to_numpy()
bid_px_01 = df['bid_px_01'].to_numpy()
ask_px_01 = df['ask_px_01'].to_numpy()
bid_px_02 = df['bid_px_02'].to_numpy()
ask_px_02 = df['ask_px_02'].to_numpy()
bid_px_03 = df['bid_px_03'].to_numpy()
ask_px_03 = df['ask_px_03'].to_numpy()
time = df['ts_event'].dt.time()

print(f"Number of trades: {len(price)}")

# Plot prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_price, y=price, mode='lines', name='Prix', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_00, mode='lines', name='Bid 0', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_00, mode='lines', name='Ask 0', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_01, mode='lines', name='Bid 1', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_01, mode='lines', name='Ask 1', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_02, mode='lines', name='Bid 2', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_02, mode='lines', name='Ask 2', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_03, mode='lines', name='Bid 3', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_03, mode='lines', name='Ask 3', showlegend=True))
fig.update_layout(title='Bid-ask', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# Calculate diffs for GOOGL symbol
depth = 0
df = df.filter(pl.col('symbol') == 'GOOGL')

# Replace 614 with 0 in specific columns rather than all columns
df = df.with_columns([
    pl.col(f'bid_sz_0{depth}').cast(pl.Int32).replace(614, 0).alias(f'bid_sz_0{depth}'),
    pl.col(f'ask_sz_0{depth}').cast(pl.Int32).replace(614, 0).alias(f'ask_sz_0{depth}')
])

# Calculate diffs
df = df.with_columns([
    pl.col(f'bid_sz_0{depth}').diff().alias(f'bid_sz_0{depth}_diff'),
    pl.col(f'ask_sz_0{depth}').diff().alias(f'ask_sz_0{depth}_diff')
])

# %%
df = pl.read_parquet('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/KHC/20240927.parquet')
df.head()

# %%
# Filter for LCID symbol and depth 0
df = df.filter(pl.col('symbol') == 'LCID')
df = df.filter(pl.col('depth') == 0)

# Convert columns to numpy arrays
size_bid = df['bid_sz_00'].to_numpy()
size_ask = df['ask_sz_00'].to_numpy()
time = df['ts_event'].dt.time()

# Plot queue sizes
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='Bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# Get trade prices and times
trades = df.filter(pl.col('action') == 'T')
price = trades['price'].to_numpy()
time_price = trades['ts_event'].dt.time()

# Get bid/ask prices
bid_px_00 = df['bid_px_00'].to_numpy()
ask_px_00 = df['ask_px_00'].to_numpy()
bid_px_01 = df['bid_px_01'].to_numpy()
ask_px_01 = df['ask_px_01'].to_numpy()
bid_px_02 = df['bid_px_02'].to_numpy()
ask_px_02 = df['ask_px_02'].to_numpy()
bid_px_03 = df['bid_px_03'].to_numpy()
ask_px_03 = df['ask_px_03'].to_numpy()
time = df['ts_event'].dt.time()

print(f"Number of trades: {len(price)}")

# Plot prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_price, y=price, mode='lines', name='Prix', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_00, mode='lines', name='Bid 0', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_00, mode='lines', name='Ask 0', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_01, mode='lines', name='Bid 1', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_01, mode='lines', name='Ask 1', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_02, mode='lines', name='Bid 2', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_02, mode='lines', name='Ask 2', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=bid_px_03, mode='lines', name='Bid 3', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=ask_px_03, mode='lines', name='Ask 3', showlegend=True))
fig.update_layout(title='Bid-ask', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# Calculate diffs for GOOGL symbol
depth = 0
df = df.filter(pl.col('symbol') == 'GOOGL')

# Replace 614 with 0 in specific columns rather than all columns
df = df.with_columns([
    pl.col(f'bid_sz_0{depth}').cast(pl.Int32).replace(614, 0).alias(f'bid_sz_0{depth}'),
    pl.col(f'ask_sz_0{depth}').cast(pl.Int32).replace(614, 0).alias(f'ask_sz_0{depth}')
])

# Calculate diffs
df = df.with_columns([
    pl.col(f'bid_sz_0{depth}').diff().alias(f'bid_sz_0{depth}_diff'),
    pl.col(f'ask_sz_0{depth}').diff().alias(f'ask_sz_0{depth}_diff')
])

df = df.filter(pl.col('depth') == depth)
