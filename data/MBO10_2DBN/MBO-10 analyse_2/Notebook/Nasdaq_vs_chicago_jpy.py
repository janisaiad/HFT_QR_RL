# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: IA_m1
#     language: python
#     name: python3
# ---

# %%
import polars as pl
import pandas as pd # ca on va devoir bcp l'uiliser
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

# %%
# Load parquet file using polars
df = pl.read_parquet('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/CHICAGO/LCID/20240927.parquet')

# Filter data
df = df.filter(pl.col('publisher_id') == 39)\
       .filter(pl.col('symbol') == 'LCID')\
       .filter(pl.col('depth') == 0)

# Convert columns to numpy arrays for plotting
size_bid = df.select('bid_sz_00').to_numpy().flatten()
size_ask = df.select('ask_sz_00').to_numpy().flatten()
time = df.select('ts_event').to_numpy().flatten()
time = pd.to_datetime(time)

# Plot queue sizes
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='Bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# Get trade prices and times
trades = df.filter(pl.col('action') == 'T')
price_1 = trades.select('price').to_numpy().flatten()
time_price_1 = pd.to_datetime(trades.select('ts_event').to_numpy().flatten())

# Get bid/ask prices
bid_px_00 = df.select('bid_px_00').to_numpy().flatten()
ask_px_00 = df.select('ask_px_00').to_numpy().flatten()
bid_px_01 = df.select('bid_px_01').to_numpy().flatten()
ask_px_01 = df.select('ask_px_01').to_numpy().flatten()
bid_px_02 = df.select('bid_px_02').to_numpy().flatten()
ask_px_02 = df.select('ask_px_02').to_numpy().flatten()
bid_px_03 = df.select('bid_px_03').to_numpy().flatten()
ask_px_03 = df.select('ask_px_03').to_numpy().flatten()
time = pd.to_datetime(df.select('ts_event').to_numpy().flatten())

print(len(price_1))

# Plot bid-ask prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_price_1, y=price_1, mode='lines', name='Prix', showlegend=True))
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


# %%
df.head()

# %%
# Replace 614 with 0 and calculate diffs
df = df.with_columns([
    pl.when(pl.col('bid_sz_00') == 614).then(0).otherwise(pl.col('bid_sz_00')).alias('bid_sz_00'),
    pl.when(pl.col('ask_sz_00') == 614).then(0).otherwise(pl.col('ask_sz_00')).alias('ask_sz_00'),
    pl.when(pl.col('bid_sz_01') == 614).then(0).otherwise(pl.col('bid_sz_01')).alias('bid_sz_01'),
    pl.when(pl.col('ask_sz_01') == 614).then(0).otherwise(pl.col('ask_sz_01')).alias('ask_sz_01'),
    pl.when(pl.col('bid_sz_02') == 614).then(0).otherwise(pl.col('bid_sz_02')).alias('bid_sz_02'),
    pl.when(pl.col('ask_sz_02') == 614).then(0).otherwise(pl.col('ask_sz_02')).alias('ask_sz_02'),
    pl.when(pl.col('bid_sz_03') == 614).then(0).otherwise(pl.col('bid_sz_03')).alias('bid_sz_03'),
    pl.when(pl.col('ask_sz_03') == 614).then(0).otherwise(pl.col('ask_sz_03')).alias('ask_sz_03')
])

depth = 0
df = df.filter(pl.col('symbol') == 'LCID')

# Calculate diffs using polars
df = df.with_columns([
    pl.col(f'bid_sz_0{depth}').diff().alias(f'bid_sz_0{depth}_diff'),
    pl.col(f'ask_sz_0{depth}').diff().alias(f'ask_sz_0{depth}_diff')
])

df = df.filter(pl.col('depth') == depth)

# %% [markdown]
# # We just removed 614 and calculated the diffs
#

# %%
# Load parquet file using polars
df = pl.read_parquet('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID/20240927.parquet')


# %%

# Filter data
df = df.filter(pl.col('publisher_id') == 2)\
       .filter(pl.col('symbol') == 'LCID')\
       .filter(pl.col('depth') == 0)

# Extract bid/ask sizes
size_bid = df.select('bid_sz_00').to_series()
size_ask = df.select('ask_sz_00').to_series() 
time = df.select('ts_event').to_series()

# Plot queue sizes
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='Bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Taille de queues', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# Extract price data
price2 = df.filter(pl.col('action') == 'T').select('price').to_series()
time_price2 = df.filter(pl.col('action') == 'T').select('ts_event').to_series()

# Extract bid/ask prices
bid_px_00 = df.select('bid_px_00').to_series()
ask_px_00 = df.select('ask_px_00').to_series()
bid_px_01 = df.select('bid_px_01').to_series()
ask_px_01 = df.select('ask_px_01').to_series()
bid_px_02 = df.select('bid_px_02').to_series()
ask_px_02 = df.select('ask_px_02').to_series()
bid_px_03 = df.select('bid_px_03').to_series()
ask_px_03 = df.select('ask_px_03').to_series()
time = df.select('ts_event').to_series()


# %%

# Plot bid-ask prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_price2, y=price2, mode='lines', name='Prix', showlegend=True))
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




# %%
df.head()

# %%
print(f"Number of unique actions: {df['action'].n_unique()}")
print("\nCount of each action type:")
print(df['action'].value_counts())


# %%
print(f"Number of unique actions: {df['depth'].n_unique()}")
print("\nCount of each action type:")
print(df['depth'].value_counts())


# %%
# Replace 614 with 0 and calculate diffs
df = df.with_columns([
    pl.when(pl.col('bid_sz_00') == 614).then(0).otherwise(pl.col('bid_sz_00')).alias('bid_sz_00'),
    pl.when(pl.col('ask_sz_00') == 614).then(0).otherwise(pl.col('ask_sz_00')).alias('ask_sz_00'),
    pl.when(pl.col('bid_sz_01') == 614).then(0).otherwise(pl.col('bid_sz_01')).alias('bid_sz_01'),
    pl.when(pl.col('ask_sz_01') == 614).then(0).otherwise(pl.col('ask_sz_01')).alias('ask_sz_01'),
    pl.when(pl.col('bid_sz_02') == 614).then(0).otherwise(pl.col('bid_sz_02')).alias('bid_sz_02'),
    pl.when(pl.col('ask_sz_02') == 614).then(0).otherwise(pl.col('ask_sz_02')).alias('ask_sz_02'),
    pl.when(pl.col('bid_sz_03') == 614).then(0).otherwise(pl.col('bid_sz_03')).alias('bid_sz_03'),
    pl.when(pl.col('ask_sz_03') == 614).then(0).otherwise(pl.col('ask_sz_03')).alias('ask_sz_03')
])

depth = 0
df = df.filter(pl.col('symbol') == 'LCID')

# Calculate diffs using polars
df = df.with_columns([
    pl.col(f'bid_sz_0{depth}').diff().alias(f'bid_sz_0{depth}_diff'),
    pl.col(f'ask_sz_0{depth}').diff().alias(f'ask_sz_0{depth}_diff')
])

df = df.filter(pl.col('depth') == depth)

# %%

fig = go.Figure()
fig.add_trace(go.Scatter(x = time_price_1, y = price_1, mode ='lines', name ='Prix_Nasdaq', showlegend = True))
fig.add_trace(go.Scatter(x = time_price2, y = price2, mode ='lines', name ='Prix_Chicago', showlegend = True))
fig.update_layout(title='Chicago vs Nasdaq', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
df = df.filter(pl.col('depth') == 0)\
       .filter(pl.col('action') == 'T')\
       .filter(pl.col('publisher_id') == 2)

# %%

df_bid = df.filter(pl.col('price') == pl.col('bid_px_00'))
df_ask = df.filter(pl.col('price') == pl.col('ask_px_00'))
time = df_bid.select('ts_event').to_series().dt.strftime('%Y-%m-%d %H:%M:%S')

size_bid = df_bid.select('bid_sz_00').to_series()
size_ask = df_bid.select('ask_sz_00').to_series()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Trade du bid queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

size_bid = df_ask.select('bid_sz_00').to_series()
size_ask = df_ask.select('ask_sz_00').to_series()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Trade du ask queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

# %%
df = pl.read_parquet('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/GOOGL/20240927.parquet')

df = df.filter(pl.col('depth') == 0)\
       .filter(pl.col('action') == 'T')\
       .filter(pl.col('publisher_id') == 2)
df.head()

# %%
df_bid = df.filter(pl.col('price') == pl.col('bid_px_00'))
df_ask = df.filter(pl.col('price') == pl.col('ask_px_00'))
time = df_bid.select('ts_event').to_series().dt.strftime('%Y-%m-%d %H:%M:%S')

size_bid = df_bid.select('bid_sz_00').to_series()
size_ask = df_bid.select('ask_sz_00').to_series()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Trade du bid queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

size_bid = df_ask.select('bid_sz_00').to_series()
size_ask = df_ask.select('ask_sz_00').to_series()


# %%

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Trade du ask queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

# %%
# Load DBN file
df = pl.read_parquet('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/KHC/20240927.parquet')

df = df.filter(pl.col('depth') == 0)\
       .filter(pl.col('action') == 'T')\
       .filter(pl.col('publisher_id') == 2)    
       
df.head()

# %%
df = df.filter(pl.col('depth') == 0)\
       .filter(pl.col('action') == 'T')\
       .filter(pl.col('publisher_id') == 2)
df.head()

df_bid = df.filter(pl.col('price') == pl.col('bid_px_00'))
df_ask = df.filter(pl.col('price') == pl.col('ask_px_00'))
time = df_bid.select('ts_event').to_series().dt.strftime('%Y-%m-%d %H:%M:%S')

size_bid = df_bid.select('bid_sz_00').to_series()
size_ask = df_bid.select('ask_sz_00').to_series()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Trade du bid queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

size_bid = df_ask.select('bid_sz_00').to_series()
size_ask = df_ask.select('ask_sz_00').to_series()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=size_bid, mode='lines', name='bid', showlegend=True))
fig.add_trace(go.Scatter(x=time, y=size_ask, mode='lines', name='Ask', showlegend=True))
fig.update_layout(title='Trade du ask queue ask et bid', xaxis_title='size', yaxis_title='taille de queue', showlegend=True)
fig.show()

# %%
