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
from glob import glob
import os
import matplotlib.pyplot as plt
import pygwalker as pyg

# %%


# Read all parquet files
files = glob(os.path.join("/mnt/beegfs/project/lobib/repos/HFT_QR_RL/data/smash4/parquet/CSX/", "*.parquet"))

print(files)

# %%

# Initialize empty lists to store data
all_data = []

# Read and concatenate all files
for file in files:
    df = pl.read_parquet(file)
    all_data.append(df)

# Concatenate all dataframes
df_all = pl.concat(all_data)

# %%
# Create histogram of price distribution
plt.figure(figsize=(12, 6))
plt.hist(df_all.filter(pl.col('price') < 1000)['price'], bins=30, color='blue', alpha=0.7)
plt.yscale('log')
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# 5 s with cpu, 

# %%
