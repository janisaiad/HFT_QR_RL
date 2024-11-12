import polars as pl
import numpy as np
from tqdm import tqdm


def transform_dataframe(df): # polars df
    # Convert to pandas for datetime operations
    pdf = df.to_pandas()
    
    # Calculate time differences between events
    pdf['ts_event'] = pd.to_datetime(pdf['ts_event'])
    pdf = pdf.sort_values('ts_event')
    
    # Get time differences for each type of event
    bid_times = pdf[pdf['side'] == 'B']['ts_event']
    ask_times = pdf[pdf['side'] == 'A']['ts_event'] 
    trade_times = pdf[pdf['action'] == 'T']['ts_event']
    
    # Calculate delta times between each event type and next event
    bid_deltas = []
    ask_deltas = []
    trade_deltas = []
    
    for i in range(len(pdf)-1):
        curr_time = pdf.iloc[i]['ts_event']
        next_time = pdf.iloc[i+1]['ts_event']
        delta = (next_time - curr_time).total_seconds()
        
        if pdf.iloc[i]['side'] == 'B':
            bid_deltas.append(delta)
        elif pdf.iloc[i]['side'] == 'A':
            ask_deltas.append(delta)
        elif pdf.iloc[i]['action'] == 'T':
            trade_deltas.append(delta)
            
    # Calculate imbalance
    pdf['imbalance'] = -(pdf['ask_sz_00'] - pdf['bid_sz_00'])/(pdf['ask_sz_00'] + pdf['bid_sz_00'])
    
    # Create output dataframe
    out_df = pl.DataFrame({
        'time': pdf['ts_event'],
        'bid_deltas': bid_deltas + [0], # Add 0 for last row
        'ask_deltas': ask_deltas + [0],
        'trade_deltas': trade_deltas + [0],
        'imbalance': pdf['imbalance']
    })
    
    return out_df