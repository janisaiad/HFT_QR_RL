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
parquet_files = glob.glob('/mnt/beegfs/project/lobib/data_pq/DB_MBP_10/GEO/*.parquet')

if not parquet_files:
    raise ValueError("No parquet files found in the specified directory")

df = pl.concat([pl.read_parquet(f, use_pyarrow=True) for f in parquet_files[:1]])

# %%
df_pandas = df.to_pandas()

# %%
walker = pyg.walk(df_pandas)

# %%
# Create a line plot using matplotlib to show trade distribution over time
import matplotlib.pyplot as plt


# %%
# Filter for trades only
trades_df = df.filter(pl.col('action') == 'trade')

# Create histogram/count plot
plt.figure(figsize=(12, 6))
plt.hist(trades_df['ts_event'], bins=100, density=True)
plt.title('Trade Count Distribution Over Time')
plt.xlabel('Time')
plt.ylabel('Density')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
