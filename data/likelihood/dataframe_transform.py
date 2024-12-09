import polars as pl
import numpy as np
from tqdm import tqdm


def transform_dataframe(df): # polars df
    # Convert ts_event from datetime string to nanosecond timestamp
    df = df.with_columns([
        pl.col('ts_event').str.strptime(pl.Datetime).dt.timestamp().mul(1e9).alias('ts_event')
    ])
    
    # Sort by timestamp 
    df = df.sort('ts_event')
    
    # Calculate time differences between events
    df = df.with_columns([
        pl.col('ts_event').diff().fill_null(0).alias('time_diff')
    ])
    
    # Convert size columns to numeric
    df = df.with_columns([
        pl.col('ask_sz_00').cast(pl.Float64),
        pl.col('bid_sz_00').cast(pl.Float64),
        pl.col('size').cast(pl.Float64),
        pl.col('price').cast(pl.Float64),
        pl.col('price_same').cast(pl.Float64),
        pl.col('price_opposite').cast(pl.Float64),
        pl.col('size_same').cast(pl.Float64), 
        pl.col('size_opposite').cast(pl.Float64),
        pl.col('nb_ppl_same').cast(pl.Float64),
        pl.col('nb_ppl_opposite').cast(pl.Float64),
        pl.col('diff_price').cast(pl.Float64),
        pl.col('Mean_price_diff').cast(pl.Float64),
        pl.col('imbalance').cast(pl.Float64),
        pl.col('indice').cast(pl.Float64),
        pl.col('bid_sz_00_diff').cast(pl.Float64),
        pl.col('ask_sz_00_diff').cast(pl.Float64),
        pl.col('price_middle').cast(pl.Float64)
    ])
    # Calculate deltas for each event type and add as new columns
    df = df.with_columns([
        pl.when(pl.col('action') == 'T')
        .then(pl.col('time_diff'))
        .otherwise(None)
        .alias('trade_deltas'),
        
        pl.when(pl.col('action') == 'A')
        .then(pl.col('time_diff'))
        .otherwise(None)
        .alias('add_deltas'),
        
        pl.when(pl.col('action') == 'C')
        .then(pl.col('time_diff'))
        .otherwise(None)
        .alias('cancel_deltas')
    ])
    out_df = df.clone()
    
    return out_df