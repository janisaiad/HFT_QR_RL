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
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from glob import glob
import os

# Read all parquet files
files = glob(os.path.join("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID_filtered", "*filtered.parquet"))

# Initialize empty lists to store data
all_data = []

# Read and concatenate all files
for file in files:
    df = pl.read_parquet(file)
    all_data.append(df)

# Concatenate all dataframes
df_all = pl.concat(all_data)

# %%

# Create figure with secondary y-axis
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Scatter(
        x=df_all['ts_event'].to_list(),
        y=df_all['price_middle'].to_list(),
        name="Mid Price",
        line=dict(color='blue')
    )
)

fig.add_trace(
    go.Scatter(
        x=df_all['ts_event'].to_list(),
        y=df_all['imbalance'].to_list(),
        name="Order Imbalance",
        yaxis="y2",
        line=dict(color='red')
    )
)

# Set layout
fig.update_layout(
    title="Mid Price and Order Imbalance Over Time",
    xaxis=dict(title="Time"),
    yaxis=dict(
        title="Mid Price",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title="Order Imbalance",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        anchor="x",
        overlaying="y",
        side="right"
    ),
    showlegend=True
)

fig.show()


# %%
