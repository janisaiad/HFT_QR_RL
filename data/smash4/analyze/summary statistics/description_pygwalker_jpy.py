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
# Read all parquet files in the directory and concatenate them
import glob


# %%

# Get list of all parquet files in the directory
parquet_files = glob.glob("/mnt/beegfs/project/lobib/repos/HFT_QR_RL/data/smash4/parquet/INTC/*.parquet")

# Read and concatenate all parquet files
df = pl.concat([
    pl.read_parquet(f).lazy() 
    for f in parquet_files
]).collect(engine="GPU")


# %%
print(df.columns)

# %%

# Convert polars DataFrame to pandas for pygwalker compatibility
df_pandas = df.to_pandas()


# %%

# Initialize pygwalker with the pandas DataFrame
walker = pyg.walk(df_pandas)


# %%
