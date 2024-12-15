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
import pandas as pd # ca on va devoir bcp l'uiliser
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None) # pour 
from tqdm import tqdm
import os
import glob
import polars as pl


# %%
def dico_queue_size(sizes, dic):
    for i in range (len(sizes)):
        if sizes[i] not in dic:
            dic[sizes[i]] = [[], [], []]
    return dic

def compute_means(dico):
    sums = 0
    means = 0
    keys = np.array(list(dico.keys()))
    for i in range (len(keys)):
        means = means+keys[i]*len(dico[keys[i]][0])+keys[i]*len(dico[keys[i]][1])+keys[i]*len(dico[keys[i]][2])
        sums = sums+len(dico[keys[i]][0])+len(dico[keys[i]][1])+len(dico[keys[i]][2])
    return means/sums



# %%

def filtrage(dico, nombre_bins, threshold=100):
    dico_p = dict(reversed(list(dico.items())))
    keys = list(dico_p.keys())
    i = 0
    while len(dico_p[keys[i]][0])<threshold:
        i += 1
    values = np.linspace(0, keys[i], nombre_bins, endpoint=True)
    keys = np.array(list(dico.keys()))
    
    real_dic = {}
    for i in range (len(keys)):
        real_k_index = np.argmin(np.abs(values-keys[i]))
        real_k = values[real_k_index]
        
        if real_k not in real_dic:
            real_dic[real_k] = [
                np.array(dico[keys[i]][0]),
                np.array(dico[keys[i]][1]),
                np.array(dico[keys[i]][2])
            ]
        else:
            real_dic[real_k] = [
                np.concatenate([real_dic[real_k][0], dico[keys[i]][0]]),
                np.concatenate([real_dic[real_k][1], dico[keys[i]][1]]),
                np.concatenate([real_dic[real_k][2], dico[keys[i]][2]])
            ]
    return real_dic

def remove_nan_from_dico(dico):
    cleaned_dico = {}
    for key, value_lists in dico.items():
        cleaned_value_lists = []
        for value_list in value_lists:
            value_array = np.array(value_list)
            cleaned_array = value_array[~np.isnan(value_array)]
            cleaned_value_lists.append(cleaned_array.tolist())
        cleaned_dico[key] = cleaned_value_lists
    return cleaned_dico



# %%

# %%
def process_dataframe(store, level):
    # Initialize empty list to store filtered data
    filtered_data = []
    
    # Get unique symbols from metadata 
    symbols = store.metadata.symbols
    
    for symbol in symbols:
        # Get data for this symbol
        symbol_data = store.get_range(
            stype_in=['mbp-1'],  # Market by price level 1
            symbols=[symbol], 
            schema='mbo'  # Market by order
        )
        
        # Convert to polars DataFrame
        symbol_data = pl.from_pandas(symbol_data)
        
        # Filter conditions
        symbol_data = symbol_data.filter(
            (pl.col('publisher_id') == 2) &
            pl.col('side').is_in(['B','A'])
        )
        
        # Process each depth level
        depths = symbol_data.select('depth').unique().to_series().to_list()
        for depth in depths:
            if depth != level:
                continue
                
            depth_data = symbol_data.filter(pl.col('depth') == depth)
            
            # Calculate size differences
            depth_data = depth_data.with_columns([
                pl.col(f'bid_sz_0{depth}').diff().alias(f'bid_sz_0{depth}_diff'),
                pl.col(f'ask_sz_0{depth}').diff().alias(f'ask_sz_0{depth}_diff')
            ])
            
            # Apply filters
            depth_data = depth_data.filter(
                ~(
                    (pl.col('action') == 'T') &
                    (
                        ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') != -pl.col('size'))) |
                        ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') != -pl.col('size')))
                    )
                )
            )

            depth_data = depth_data.filter(
                ~(
                    (pl.col('action') == 'A') &
                    (
                        ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') != pl.col('size'))) |
                        ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') != pl.col('size')))
                    )
                )
            )

            depth_data = depth_data.filter(
                ~(
                    (pl.col('action') == 'C') &
                    (
                        ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') != -pl.col('size'))) |
                        ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') != -pl.col('size')))
                    )
                )
            )
            
            filtered_data.append(depth_data)
    
    # Combine all filtered data
    df_final = pl.concat(filtered_data)
    
    # Sort by timestamp
    df_final = df_final.sort('ts_event')
    
    # Add index columns
    df_final = df_final.with_row_count('reindex')
    df_final = df_final.with_columns(
        pl.col('reindex').diff().fill_null(0).cast(pl.Int64).alias('diff_reindex')
    )
    
    # Filter for GOOGL only
    df_result = df_final.filter(
        (pl.col('symbol') == 'GOOGL') &
        (pl.col('depth') == level)
    )
    
    return df_result

# Get list of DBN files
files_parquet = glob.glob(os.path.join("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID", "*.parquet"))

# %%
print(files_parquet)

# %%
dic = {}

for f in tqdm(files_parquet):
    # Read parquet file using polars
    MBO_ = pl.read_parquet(f)
    
    # Process timestamps and calculate elapsed time
    MBO_ = MBO_.with_columns([
        pl.col('ts_event').str.strptime(pl.Datetime).alias('ts_event'),
        pl.col('ts_event').str.strptime(pl.Datetime).diff().dt.seconds().alias('temps_ecoule_secondes')
    ])
    
    # Filter for time window between 14:00 and 20:00
    MBO_ = MBO_.filter(
        (pl.col('ts_event').dt.hour() >= 14) & 
        (pl.col('ts_event').dt.hour() < 20)
    )
    
    # Get unique sizes from bid and ask for depth 0
    bid_sizes = MBO_.select('bid_sz_00').unique().to_numpy().flatten()
    ask_sizes = MBO_.select('ask_sz_00').unique().to_numpy().flatten()
    sizes = np.unique(np.concatenate([bid_sizes, ask_sizes]))
    sizes.sort()
    
    # Drop null values
    MBO_ = MBO_.drop_nulls()
    
    # Initialize dictionary
    dic = dico_queue_size(sizes, dic)
    
    # Process each row
    for row in MBO_.iter_rows(named=True):
        taille = row['ask_sz_00'] if row['side'] == 'A' else row['bid_sz_00']
        
        if row['action'] == 'A':
            dic[taille][0].append(row['temps_ecoule_secondes'])
        elif row['action'] == 'C':
            dic[taille][1].append(row['temps_ecoule_secondes'])
        elif row['action'] == 'T':
            dic[taille][2].append(row['temps_ecoule_secondes'])

# %%
# visualisation
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 100

dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 30, threshold=50)

threshold_trade = 1000
threshold = 40000
for i in intensities:
    if len(intensities[i][0])!=0:
        if (len(intensities[i][0])>threshold):
            Add.append(np.mean(np.array(intensities[i][0])))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(np.mean(np.array(intensities[i][1])))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold_trade):
            Trade.append(np.mean(np.array(intensities[i][2])))
            sizes_trade.append(i)


# %%

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
#fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = 1/np.array(Trade), mode ='lines', name = f'Trade', showlegend = True))
fig.update_layout(title='Intensities GOOGL premiere limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
# visualisation
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 100

dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 30, threshold=500)

threshold_trade = 1000
threshold = 20000

for i in intensities:
    tab = np.concatenate((intensities[i][0], intensities[i][1], intensities[i][2]))
    if (len(intensities[i][0])>threshold):
            Add.append(1/np.mean(tab)*len(intensities[i][0])/len(tab))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(1/np.mean(tab)*len(intensities[i][1])/len(tab))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold_trade):
            Trade.append(1/np.mean(tab)*len(intensities[i][2])/len(tab))
            sizes_trade.append(i)
            
fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = np.array(Trade), mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities GOOGL premiere limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
dicu = {}

for f in tqdm(files_csv):
    MBO_ = pd.read_csv(f)
    MBO_filtered_depth_0_ = process_dataframe(MBO_,1)
    MBO_filtered_depth_0_['ts_event'] = MBO_filtered_depth_0_['ts_event'] = pd.to_datetime(MBO_filtered_depth_0_['ts_event'], errors='coerce')
    MBO_filtered_depth_0_['temps_ecoule'] = MBO_filtered_depth_0_['ts_event'].diff()
    MBO_filtered_depth_0_['temps_ecoule_secondes'] = MBO_filtered_depth_0_['temps_ecoule'].dt.total_seconds()
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[(MBO_filtered_depth_0_['ts_event'].dt.hour >= 14) & (MBO_filtered_depth_0_['ts_event'].dt.hour < 20)]
    MBO_filtered_depth__ = MBO_filtered_depth_0_.iloc[1:] # on enleve le premier NA
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0_['bid_sz_00'].to_numpy())).tolist()+(np.unique(MBO_filtered_depth_0_['ask_sz_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0_.dropna()
    sizes.sort()

    dicu = dico_queue_size(sizes, dicu) ## Add, Cancel, Trade
    for row in MBO_filtered_depth_0_.itertuples():
        if row.side == 'A':
            taille = row.ask_sz_00
        if row.side == 'B':
            taille = row.bid_sz_00
        if row.action =='A':
            dicu[taille][0].append(row.temps_ecoule_secondes)
        if row.action =='C':
            dicu[taille][1].append(row.temps_ecoule_secondes)
        if row.action =='T':
            dicu[taille][2].append(row.temps_ecoule_secondes)

# %%
# visualisation
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 100

dicu = remove_nan_from_dico(dicu)
average_sizes = compute_means(dicu)
intensities = dict(sorted(dicu.items()))
intensities = filtrage(intensities, 30, threshold=50)

threshold_trade = 1000
threshold = 40000
for i in intensities:
    if len(intensities[i][0])!=0:
        if (len(intensities[i][0])>threshold):
            Add.append(np.mean(np.array(intensities[i][0])))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(np.mean(np.array(intensities[i][1])))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold_trade):
            Trade.append(np.mean(np.array(intensities[i][2])))
            sizes_trade.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
#fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = 1/np.array(Trade), mode ='lines', name = f'Trade', showlegend = True))
fig.update_layout(title='Intensities GOOGL seconde limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
# visualisation
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 100

dicu = remove_nan_from_dico(dicu)
average_sizes = compute_means(dicu)
intensities = dict(sorted(dicu.items()))
intensities = filtrage(intensities, 30, threshold=50)

threshold_trade = 100
threshold = 4000
for i in intensities:
    tab = np.concatenate((intensities[i][0], intensities[i][1], intensities[i][2]))
    if (len(intensities[i][0])>threshold):
            Add.append(np.mean(tab)*len(intensities[i][0])/len(tab))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(np.mean(tab)*len(intensities[i][1])/len(tab))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold_trade):
            Trade.append(np.mean(tab)*len(intensities[i][2])/len(tab))
            sizes_trade.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = 1/np.array(Trade), mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities GOOGL seconde limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
dic = {}

for f in tqdm(files_csv):
    MBO_ = pd.read_csv(f)
    MBO = MBO_[MBO_["publisher_id"] == 2]
    MBO_filtered = MBO[MBO['symbol'] == "GOOGL"]
    MBO_filtered_depth_0_ = MBO_filtered[MBO_filtered['depth'] == 1]
    MBO_filtered_depth_0_['bid_sz_00_diff'] = MBO_filtered_depth_0_['bid_sz_00'].diff()
    MBO_filtered_depth_0_['ask_sz_00_diff'] = MBO_filtered_depth_0_['ask_sz_00'].diff()
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[
        ~(
            (MBO_filtered_depth_0_['action'] == 'C')&
            (
                ((MBO_filtered_depth_0_['side'] == 'B')&(MBO_filtered_depth_0_['bid_sz_00_diff'] == 0)) |
                ((MBO_filtered_depth_0_['side'] == 'A')&(MBO_filtered_depth_0_['ask_sz_00_diff'] == 0))
            )
        )
    ]
    MBO_filtered_depth_0 = MBO_filtered_depth_0_.iloc[1:] # on enleve le premier NA
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0['bid_sz_00'].to_numpy())).tolist()+(np.unique(MBO_filtered_depth_0['ask_sz_00'].to_numpy())).tolist()))

    MBO_filtered_depth_0['ts_event'] = pd.to_datetime(MBO_filtered_depth_0['ts_event'])
    MBO_filtered_depth_0['temps_ecoule'] = MBO_filtered_depth_0['ts_event'].diff()
    MBO_filtered_depth_0['temps_ecoule_secondes'] = MBO_filtered_depth_0['temps_ecoule'].dt.total_seconds()
    MBO_filtered_depth_0 = MBO_filtered_depth_0[(MBO_filtered_depth_0['ts_event'].dt.hour >= 14) & (MBO_filtered_depth_0['ts_event'].dt.hour < 20)]
    MBO_filtered_depth_0.dropna()
    sizes.sort()

    dic = dico_queue_size(sizes, dic) ## Add, Cancel, Trade
    for row in MBO_filtered_depth_0.itertuples():
        if row.side == 'A':
            taille = row.ask_sz_00
        if row.side == 'B':
            taille = row.bid_sz_00
        if row.action =='A':
            dic[taille][0].append(row.temps_ecoule_secondes)
        if row.action =='C':
            dic[taille][1].append(row.temps_ecoule_secondes)
        if row.action =='T':
            dic[taille][2].append(row.temps_ecoule_secondes)

# %%
# visualisation
Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 100

dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 100, threshold=1000)

threshold = 50000

for i in intensities:
    if len(intensities[i][0])!=0:
        if (len(intensities[i][0])>threshold):
            Add.append(np.mean(np.array(intensities[i][0])))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>threshold):
            Cancel.append(np.mean(np.array(intensities[i][1])))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>threshold):
            Trade.append(np.mean(np.array(intensities[i][2])))
            sizes_trade.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = 1/np.array(Trade), mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities ASAI', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()
