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
pd.set_option('display.max_rows', None)
from tqdm import tqdm
import os
import glob
import polars as pl


# %%
def process_dataframe(df: pl.DataFrame, depth: int) -> pl.DataFrame:
    # Replace 614 with 0 and filter for GOOGL symbol
    df = df.with_columns(pl.col('*').replace(614, 0)) \
           .filter(pl.col('symbol') == 'GOOGL')
    
    # Convert timestamp and calculate diffs
    df = df.with_columns([
        pl.col('ts_event').str.strptime(pl.Datetime, fmt='%Y-%m-%d %H:%M:%S%.f'),
        pl.col(f'bid_sz_0{depth}').diff().alias(f'bid_sz_0{depth}_diff'),
        pl.col(f'ask_sz_0{depth}').diff().alias(f'ask_sz_0{depth}_diff')
    ])
    
    # Filter for specific depth
    df = df.filter(pl.col('depth') == depth)
    
    # Calculate elapsed time
    df = df.with_columns([
        pl.col('ts_event').diff().alias('temps_ecoule'),
        pl.col('ts_event').diff().dt.seconds().alias('temps_ecoule_secondes')
    ])
    
    # Create status column based on conditions
    df = df.with_columns(
        pl.when(
            (pl.col('action') == 'T') & (
                ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') == -pl.col('size'))) |
                ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') == -pl.col('size')))
            ) |
            (pl.col('action') == 'A') & (
                ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') == pl.col('size'))) |
                ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') == pl.col('size')))
            ) |
            (pl.col('action') == 'C') & (
                ((pl.col('side') == 'B') & (pl.col(f'bid_sz_0{depth}_diff') == -pl.col('size'))) |
                ((pl.col('side') == 'A') & (pl.col(f'ask_sz_0{depth}_diff') == -pl.col('size')))
            )
        ).then('OK').otherwise('NOK').alias('status')
    )
    
    return df.filter(pl.col('status') == 'OK')

def dico_queue_size(sizes, dic):
    for size in sizes:
        if size not in dic:
            dic[size] = [[], [], []]
    return dic

def compute_means(dico):
    sums = 0
    means = 0
    keys = np.array(list(dico.keys()))
    for key in keys:
        means += key * sum(len(lst) for lst in dico[key])
        sums += sum(len(lst) for lst in dico[key])
    return means/sums if sums > 0 else 0

def filtrage(dico, nombre_bins, threshold=100):
    dico_p = dict(reversed(list(dico.items())))
    keys = list(dico_p.keys())
    i = 0
    while len(dico_p[keys[i]][0]) < threshold:
        i += 1
    values = np.linspace(0, keys[i], nombre_bins, endpoint=True)
    keys = np.array(list(dico.keys()))
    
    real_dic = {}
    for key in keys:
        real_k_index = np.argmin(np.abs(values-key))
        real_k = values[real_k_index]
        
        if real_k not in real_dic:
            real_dic[real_k] = [
                np.array(dico[key][0]),
                np.array(dico[key][1]), 
                np.array(dico[key][2])
            ]
        else:
            real_dic[real_k] = [
                np.concatenate([real_dic[real_k][0], dico[key][0]]),
                np.concatenate([real_dic[real_k][1], dico[key][1]]),
                np.concatenate([real_dic[real_k][2], dico[key][2]])
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

# Use glob to find all parquet files in the specified directory
files_parquet = glob.glob(os.path.join("/home/janis/3A/EA/HFT_QR_RL/data/smash3/data/csv/NASDAQ/LCID", "*.parquet"))

# %%
import polars as pl
from joblib import Parallel, delayed

def process_file(f, level, dic):
    # Read parquet file with polars
    MBO_ = pl.read_parquet(f)
    MBO_filtered_depth_0_ = process_dataframe(MBO_, level)
    
    # Filter by hour using polars expressions
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_.filter(
        (pl.col('ts_event').dt.hour() >= 14) & 
        (pl.col('ts_event').dt.hour() < 20)
    )
    
    # Get unique sizes using polars and convert to float to avoid integer overflow
    bid_sizes = MBO_filtered_depth_0_.select('bid_ct_00').unique().cast(pl.Float64).to_numpy().flatten()
    ask_sizes = MBO_filtered_depth_0_.select('ask_ct_00').unique().cast(pl.Float64).to_numpy().flatten()
    sizes = np.unique(np.concatenate([bid_sizes, ask_sizes]))
    
    # Drop nulls and sort
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_.drop_nulls()
    sizes.sort()
    
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    
    # Convert to pandas for itertuples since polars doesn't have direct equivalent
    df = MBO_filtered_depth_0_.to_pandas()
    for row in df.itertuples():
        if row.side == 'A':
            taille = float(row.ask_ct_00)  # Convert to float to avoid overflow
        elif row.side == 'B':
            taille = float(row.bid_ct_00)  # Convert to float to avoid overflow
        else:
            continue
        
        if row.action == 'A':
            dic[taille][0].append(row.temps_ecoule_secondes)
        elif row.action == 'C':
            dic[taille][1].append(row.temps_ecoule_secondes)
        elif row.action == 'T':
            dic[taille][2].append(row.temps_ecoule_secondes)
    
    return dic

level = 0

dic = {}
results = Parallel(n_jobs=4)(delayed(process_file)(f, level, dic) for f in tqdm(files_parquet))

for result in results:
    for size, actions in result.items():
        if size not in dic:
            dic[size] = actions
        else:
            for i in range(3):  # Add, Cancel, Trade
                dic[size][i].extend(actions[i])


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
# df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq/xnas-itch-20240723.mbp-10.csv')
# df = df[df['symbol'] == 'GOOGL']
# df = df[df['depth'] == 0]
# df['ts_event'] = pd.to_datetime(df['ts_event'])
# df = df[(df['ts_event'].dt.hour >= 14) & (df['ts_event'].dt.hour < 20)]
# average_sizes = np.mean(df['size'].to_numpy())
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 100, threshold=1000)

threshold_trade = 2000
threshold = 1000

#quarter = []

for i in intensities:
    tab = np.concatenate((intensities[i][0], intensities[i][1], intensities[i][2]))
    if (len(intensities[i][0])>threshold):
            Add.append(1/np.mean(tab)*len(intensities[i][0])/len(tab))
            #quarter.append(np.var(tab)*1/np.mean(tab)*len(intensities[i][0])/len(tab)*1/np.sqrt(len(tab)))
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
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = Add, mode ='lines', name ='Add', showlegend = True))
#fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = np.array(Add)+1.96*np.array(quarter), mode ='lines', name ='Add', showlegend = True))
#fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = np.array(Add)-1.96*np.array(quarter), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = Cancel, mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = Trade, mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities GOOGL premiere limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def process_file(f, level, dic):
    MBO_ = pd.read_csv(f)
    MBO_filtered_depth_0_ = process_dataframe(MBO_, level)
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[(MBO_filtered_depth_0_['ts_event'].dt.hour >= 14) & (MBO_filtered_depth_0_['ts_event'].dt.hour < 20)]
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0_['bid_ct_00'].to_numpy())).tolist() + (np.unique(MBO_filtered_depth_0_['ask_ct_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_.dropna()
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    for row in MBO_filtered_depth_0_.itertuples():
        if row.side == 'A':
            taille = row.ask_ct_00
        elif row.side == 'B':
            taille = row.bid_ct_00
        else:
            continue
        
        if row.action == 'A':
            dic[taille][0].append(row.temps_ecoule_secondes)
        elif row.action == 'C':
            dic[taille][1].append(row.temps_ecoule_secondes)
        elif row.action == 'T':
            dic[taille][2].append(row.temps_ecoule_secondes)
    
    return dic
level = 1

dic = {}
results = Parallel(n_jobs=4)(delayed(process_file)(f, level, dic) for f in tqdm(files_csv))

for result in results:
    for size, actions in result.items():
        if size not in dic:
            dic[size] = actions
        else:
            for i in range(3):  # Add, Cancel, Trade
                dic[size][i].extend(actions[i])


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
# df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq')
# df = df[df['symbol'] == 'GOOGL']
# df = df[df['depth'] == 1]
# df['ts_event'] = pd.to_datetime(df['ts_event'])
# df = df[(df['ts_event'].dt.hour >= 14) & (df['ts_event'].dt.hour < 20)]
# average_sizes = np.mean(df['size'].to_numpy())
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 50, threshold=50)



threshold_trade = 2000
threshold = 4000

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
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = Add, mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = Cancel, mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = Trade, mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities GOOGL seconde limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()


# %%
def process_dataframe(df, depth):
    df = df.replace(614, 0)
    df = df[df['symbol'] == 'KHC']
    df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
    df[f'bid_sz_0{depth}_diff'] = df[f'bid_sz_0{depth}'].diff()
    df[f'ask_sz_0{depth}_diff'] = df[f'ask_sz_0{depth}'].diff()
    df = df[df['depth'] == depth]
    # df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
    # try:
    #     df['ts_event'] = pd.to_datetime(df['ts_event'], format="%Y-%m-%d %H:%M:%S%z", errors='coerce')
    # except ValueError:
    #     df['ts_event'] = pd.to_datetime(df['ts_event'], format='mixed', errors='coerce')
    
    df['temps_ecoule'] = df['ts_event'].diff()
    df['temps_ecoule_secondes'] = df['temps_ecoule'].dt.total_seconds()
    condition_T = (
        (df['action'] == 'T') &
        (
            ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == -df['size'])) |
            ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == -df['size']))
        )
    )
    condition_A = (
        (df['action'] == 'A') &
        (
            ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == df['size'])) |
            ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == df['size']))
        )
    )
    condition_C = (
        (df['action'] == 'C') &
        (
            ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == -df['size'])) |
            ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == -df['size']))
        )
    )
    df['status'] = np.where(condition_T | condition_A | condition_C, 'OK', 'NOK')
    df = df[df['status'] == 'OK']
    return df

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

files_csv = glob.glob(os.path.join("/Volumes/T9/CSV_dezippe_nasdaq", "*.csv"))
#files_csv = glob.glob(os.path.join("/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq", "*.csv"))

# %%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def process_file(f, level, dic):
    MBO_ = pd.read_csv(f)
    MBO_filtered_depth_0_ = process_dataframe(MBO_, level)
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[(MBO_filtered_depth_0_['ts_event'].dt.hour >= 14) & (MBO_filtered_depth_0_['ts_event'].dt.hour < 20)]
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0_['bid_sz_00'].to_numpy())).tolist() + (np.unique(MBO_filtered_depth_0_['ask_sz_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_.dropna()
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    for row in MBO_filtered_depth_0_.itertuples():
        if row.side == 'A':
            taille = row.ask_sz_00
        elif row.side == 'B':
            taille = row.bid_sz_00
        else:
            continue
        
        if row.action == 'A':
            dic[taille][0].append(row.temps_ecoule_secondes)
        elif row.action == 'C':
            dic[taille][1].append(row.temps_ecoule_secondes)
        elif row.action == 'T':
            dic[taille][2].append(row.temps_ecoule_secondes)
    
    return dic
level = 0

dic = {}
results = Parallel(n_jobs=4)(delayed(process_file)(f, level, dic) for f in tqdm(files_csv))

for result in results:
    for size, actions in result.items():
        if size not in dic:
            dic[size] = actions
        else:
            for i in range(3):  # Add, Cancel, Trade
                dic[size][i].extend(actions[i])


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
# df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq')
# df = df[df['symbol'] == 'KHC']
# df = df[df['depth'] == 0]
# df['ts_event'] = pd.to_datetime(df['ts_event'])
# df = df[(df['ts_event'].dt.hour >= 14) & (df['ts_event'].dt.hour < 20)]
# average_sizes = np.mean(df['size'].to_numpy())
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 50, threshold=300)

threshold_trade = 2000
threshold = 4000

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
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = Add, mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = Cancel, mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = Trade, mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities KHC premiere limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def process_file(f, level, dic):
    MBO_ = pd.read_csv(f)
    MBO_filtered_depth_0_ = process_dataframe(MBO_, level)
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[(MBO_filtered_depth_0_['ts_event'].dt.hour >= 14) & (MBO_filtered_depth_0_['ts_event'].dt.hour < 20)]
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0_['bid_sz_00'].to_numpy())).tolist() + (np.unique(MBO_filtered_depth_0_['ask_sz_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_.dropna()
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    for row in MBO_filtered_depth_0_.itertuples():
        if row.side == 'A':
            taille = row.ask_sz_00
        elif row.side == 'B':
            taille = row.bid_sz_00
        else:
            continue
        
        if row.action == 'A':
            dic[taille][0].append(row.temps_ecoule_secondes)
        elif row.action == 'C':
            dic[taille][1].append(row.temps_ecoule_secondes)
        elif row.action == 'T':
            dic[taille][2].append(row.temps_ecoule_secondes)
    
    return dic
level = 1

dic = {}
results = Parallel(n_jobs=4)(delayed(process_file)(f, level, dic) for f in tqdm(files_csv))

for result in results:
    for size, actions in result.items():
        if size not in dic:
            dic[size] = actions
        else:
            for i in range(3):  # Add, Cancel, Trade
                dic[size][i].extend(actions[i])


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
# df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq')
# df = df[df['symbol'] == 'KHC']
# df = df[df['depth'] == 1]
# df['ts_event'] = pd.to_datetime(df['ts_event'])
# df = df[(df['ts_event'].dt.hour >= 14) & (df['ts_event'].dt.hour < 20)]
# average_sizes = np.mean(df['size'].to_numpy())
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 50, threshold=500)

threshold_trade = 2000
threshold = 4000

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
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = Add, mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = Cancel, mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = Trade, mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities KHC seconde limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()


# %%
def process_dataframe(df, depth):
    df = df.replace(614, 0)
    df = df[df['symbol'] == 'LCID']
    df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
    df[f'bid_sz_0{depth}_diff'] = df[f'bid_sz_0{depth}'].diff()
    df[f'ask_sz_0{depth}_diff'] = df[f'ask_sz_0{depth}'].diff()
    df = df[df['depth'] == depth]
    # df['ts_event'] = pd.to_datetime(df['ts_event'], errors='coerce')
    # try:
    #     df['ts_event'] = pd.to_datetime(df['ts_event'], format="%Y-%m-%d %H:%M:%S%z", errors='coerce')
    # except ValueError:
    #     df['ts_event'] = pd.to_datetime(df['ts_event'], format='mixed', errors='coerce')
    
    df['temps_ecoule'] = df['ts_event'].diff()
    df['temps_ecoule_secondes'] = df['temps_ecoule'].dt.total_seconds()
    condition_T = (
        (df['action'] == 'T') &
        (
            ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == -df['size'])) |
            ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == -df['size']))
        )
    )
    condition_A = (
        (df['action'] == 'A') &
        (
            ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == df['size'])) |
            ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == df['size']))
        )
    )
    condition_C = (
        (df['action'] == 'C') &
        (
            ((df['side'] == 'B') & (df[f'bid_sz_0{depth}_diff'] == -df['size'])) |
            ((df['side'] == 'A') & (df[f'ask_sz_0{depth}_diff'] == -df['size']))
        )
    )
    df['status'] = np.where(condition_T | condition_A | condition_C, 'OK', 'NOK')
    df = df[df['status'] == 'OK']
    return df

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

files_csv = glob.glob(os.path.join("/Volumes/T9/CSV_dezippe_nasdaq", "*.csv"))
#files_csv = glob.glob(os.path.join("/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq", "*.csv"))

# %%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def process_file(f, level, dic):
    MBO_ = pd.read_csv(f)
    MBO_filtered_depth_0_ = process_dataframe(MBO_, level)
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[(MBO_filtered_depth_0_['ts_event'].dt.hour >= 14) & (MBO_filtered_depth_0_['ts_event'].dt.hour < 20)]
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0_['bid_sz_00'].to_numpy())).tolist() + (np.unique(MBO_filtered_depth_0_['ask_sz_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_.dropna()
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    for row in MBO_filtered_depth_0_.itertuples():
        if row.side == 'A':
            taille = row.ask_sz_00
        elif row.side == 'B':
            taille = row.bid_sz_00
        else:
            continue
        
        if row.action == 'A':
            dic[taille][0].append(row.temps_ecoule_secondes)
        elif row.action == 'C':
            dic[taille][1].append(row.temps_ecoule_secondes)
        elif row.action == 'T':
            dic[taille][2].append(row.temps_ecoule_secondes)
    
    return dic
level = 0

dic = {}
results = Parallel(n_jobs=4)(delayed(process_file)(f, level, dic) for f in tqdm(files_csv))

for result in results:
    for size, actions in result.items():
        if size not in dic:
            dic[size] = actions
        else:
            for i in range(3):  # Add, Cancel, Trade
                dic[size][i].extend(actions[i])


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
# df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq')
# df = df[df['symbol'] == 'LCID']
# df = df[df['depth'] == 0]
# df['ts_event'] = pd.to_datetime(df['ts_event'])
# df = df[(df['ts_event'].dt.hour >= 14) & (df['ts_event'].dt.hour < 20)]
# average_sizes = np.mean(df['size'].to_numpy())
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 50, threshold=300)

threshold_trade = 2000
threshold = 4000

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
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = Add, mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = Cancel, mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = Trade, mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities LUCID premiere limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def process_file(f, level, dic):
    MBO_ = pd.read_csv(f)
    MBO_filtered_depth_0_ = process_dataframe(MBO_, level)
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[(MBO_filtered_depth_0_['ts_event'].dt.hour >= 14) & (MBO_filtered_depth_0_['ts_event'].dt.hour < 20)]
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0_['bid_sz_00'].to_numpy())).tolist() + (np.unique(MBO_filtered_depth_0_['ask_sz_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_.dropna()
    sizes.sort()
    dic = dico_queue_size(sizes, dic)  # Add, Cancel, Trade
    for row in MBO_filtered_depth_0_.itertuples():
        if row.side == 'A':
            taille = row.ask_sz_00
        elif row.side == 'B':
            taille = row.bid_sz_00
        else:
            continue
        
        if row.action == 'A':
            dic[taille][0].append(row.temps_ecoule_secondes)
        elif row.action == 'C':
            dic[taille][1].append(row.temps_ecoule_secondes)
        elif row.action == 'T':
            dic[taille][2].append(row.temps_ecoule_secondes)
    
    return dic
level = 1

dic = {}
results = Parallel(n_jobs=4)(delayed(process_file)(f, level, dic) for f in tqdm(files_csv))

for result in results:
    for size, actions in result.items():
        if size not in dic:
            dic[size] = actions
        else:
            for i in range(3):  # Add, Cancel, Trade
                dic[size][i].extend(actions[i])


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
# df = pd.read_csv('/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse_2/CSV_dezippe_nasdaq')
# df = df[df['symbol'] == 'LCID']
# df = df[df['depth'] == 1]
# df['ts_event'] = pd.to_datetime(df['ts_event'])
# df = df[(df['ts_event'].dt.hour >= 14) & (df['ts_event'].dt.hour < 20)]
# average_sizes = np.mean(df['size'].to_numpy())
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 50, threshold=300)

threshold_trade = 2000
threshold = 100

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
fig.add_trace(go.Scatter(x = sizes_add/average_sizes, y = Add, mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes, y = Cancel, mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes, y = Trade, mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities LUCID seconde limite', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

# %%
