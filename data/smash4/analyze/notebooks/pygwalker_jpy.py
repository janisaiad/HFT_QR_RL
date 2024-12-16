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
import pygwalker as pyg
import glob

# %%
# Get all parquet files in the specified directory
parquet_files = glob.glob('/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/GOOGL_filtered/*_filtered_PL_with_true_diff_price.parquet')

# Check if the list of parquet files is not empty
if not parquet_files:
    raise ValueError("No parquet files found in the specified directory")

# Read and concatenate all parquet files using polars with GPU support
df = pl.concat([pl.read_parquet(f, use_pyarrow=True) for f in parquet_files])


# %%

# Convert to pandas for pygwalker compatibility
df_pandas = df.to_pandas()


# %%

# Initialize pygwalker with the dataframe
walker = pyg.walk(df_pandas)

