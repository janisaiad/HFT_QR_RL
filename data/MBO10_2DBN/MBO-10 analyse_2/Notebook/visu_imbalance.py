import polars as pl
import numpy as np
import plotly.graph_objects as go
import warnings
import os
import glob
from collections import Counter

def processing(file, output):
    actif = 'GOOGL'
    limite = 0
    
    # Read data with polars
    df = pl.read_csv(file)
    
    # Initial filtering
    df = df.filter(
        (pl.col('symbol') == actif) &
        (pl.col('depth') == limite) &
        (pl.col('side').is_in(['A','B']))
    )
    
    # Convert timestamp and filter time range
    df = df.with_columns([
        pl.col('ts_event').str.strptime(pl.Datetime, fmt=None).alias('ts_event')
    ]).filter(
        (pl.col('ts_event').dt.hour() >= 14) & 
        (pl.col('ts_event').dt.hour() < 19)
    )

    # Select columns and create derived columns
    df_ = df.select([
        'ts_event', 'action', 'side', 'size', 'price',
        f'bid_px_0{limite}', f'ask_px_0{limite}', 
        f'bid_sz_0{limite}', f'ask_sz_0{limite}',
        f'bid_ct_0{limite}', f'ask_ct_0{limite}',
        f'bid_px_0{limite+1}', f'ask_px_0{limite+1}', 
        f'bid_sz_0{limite+1}', f'ask_sz_0{limite+1}',
        f'bid_ct_0{limite+1}', f'ask_ct_0{limite+1}'
    ])

    # Calculate derived columns
    df_ = df_.with_columns([
        pl.col('ts_event').diff().dt.seconds().alias('time_diff'),
        pl.when(pl.col('side') == 'A')
          .then(pl.col(f'ask_px_0{limite}'))
          .otherwise(pl.col(f'bid_px_0{limite}'))
          .alias('price_same'),
        pl.when(pl.col('side') == 'A')
          .then(pl.col(f'bid_px_0{limite}'))
          .otherwise(pl.col(f'ask_px_0{limite}'))
          .alias('price_opposite'),
        pl.when(pl.col('side') == 'A')
          .then(pl.col(f'ask_sz_0{limite}'))
          .otherwise(pl.col(f'bid_sz_0{limite}'))
          .alias('size_same'),
        pl.when(pl.col('side') == 'A')
          .then(pl.col(f'bid_sz_0{limite}'))
          .otherwise(pl.col(f'ask_sz_0{limite}'))
          .alias('size_opposite'),
        pl.when(pl.col('side') == 'A')
          .then(pl.col(f'ask_ct_0{limite}'))
          .otherwise(pl.col(f'bid_ct_0{limite}'))
          .alias('nb_ppl_same'),
        pl.when(pl.col('side') == 'A')
          .then(pl.col(f'bid_ct_0{limite}'))
          .otherwise(pl.col(f'ask_ct_0{limite}'))
          .alias('nb_ppl_opposite'),
        pl.col('price').diff().alias('diff_price'),
        pl.col('diff_price').rolling_mean(10).shift(1).alias('Mean_price_diff'),
        ((pl.col(f'ask_sz_0{limite}') - pl.col(f'bid_sz_0{limite}')) / 
         (pl.col(f'ask_sz_0{limite}') + pl.col(f'bid_sz_0{limite}'))).alias('imbalance'),
        pl.arange(0, pl.count()).alias('indice'),
        pl.col(f'bid_sz_0{limite}').diff().alias(f'bid_sz_0{limite}_diff'),
        pl.col(f'ask_sz_0{limite}').diff().alias(f'ask_sz_0{limite}_diff')
    ])

    # Status conditions
    df_ = df_.with_columns([
        pl.when(
            ((pl.col('action') == 'T') & 
             ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{limite}_diff') == -pl.col('size')) |
              (pl.col('side') == 'A') & (pl.col(f'ask_sz_0{limite}_diff') == -pl.col('size')))) |
            ((pl.col('action') == 'A') &
             ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{limite}_diff') == pl.col('size')) |
              (pl.col('side') == 'A') & (pl.col(f'ask_sz_0{limite}_diff') == pl.col('size')))) |
            ((pl.col('action') == 'C') &
             ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{limite}_diff') == -pl.col('size')) |
              (pl.col('side') == 'A') & (pl.col(f'ask_sz_0{limite}_diff') == -pl.col('size'))))
        ).then('OK').otherwise('NOK').alias('status'),
        pl.when(
            (pl.col(f'bid_px_0{limite}').diff() > 0) | 
            (pl.col(f'ask_px_0{limite}').diff() > 0)
        ).then('new_limite').otherwise('n').alias('new_limite')
    ])

    # Group by timestamp and process trades
    df_ = df_.sort('ts_event')
    
    # Final calculations
    df_ = df_.with_columns([
        ((pl.col(f'ask_px_0{limite}') + pl.col(f'bid_px_0{limite}')) / 2).alias('price_middle'),
        pl.col('diff_price').shift(-50) - pl.col('diff_price').alias('Mean_price_diff'),
        (-(pl.col(f'ask_sz_0{limite}') - pl.col(f'bid_sz_0{limite}')) / 
         (pl.col(f'ask_sz_0{limite}') + pl.col(f'bid_sz_0{limite}'))).alias('imbalance')
    ])

    # Filter and save
    df_ = df_.filter(pl.col('time_diff') > 0)
    df_ = df_.filter(pl.col('status') != 'NOK')
    
    # Save as parquet
    df_.write_parquet(output + file[-29:-4] + '.parquet')
