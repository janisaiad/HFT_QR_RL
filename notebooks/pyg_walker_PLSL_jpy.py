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
import plotly.graph_objects as go
import plotly.express as px
from glob import glob
import os
import pygwalker as pyg
import cudf


# %%

# Read all parquet files
files = glob(os.path.join("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/GOOGL_filtered", "*filtered_PL.parquet"))


# %%
print(files)

# %%

# Initialize empty list for polars dataframes
all_data_pl = []

# Read files with polars and cudf
for file in files:
    # Read with polars first
    df_pl = pl.read_parquet(file)
    
    all_data_pl.append(df_pl)

# Concatenate polars dataframes 
df_all_pl = pl.concat(all_data_pl)

# Convert final result to cudf
df_all_cudf = cudf.DataFrame.from_pandas(df_all_pl.to_pandas())

# %%
# Convert cudf dataframe to pandas for pygwalker compatibility
df_pandas = df_all_cudf.to_pandas()


# %%

# Initialize pygwalker with the dataframe
walker = pyg.walk(df_pandas)


# %%

# %%
