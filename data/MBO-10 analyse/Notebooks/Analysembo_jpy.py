# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: EA
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


# %%
def dico_queue_size(sizes, dic):
    for i in range (len(sizes)):
        if sizes[i] not in dic:
            dic[sizes[i]] = [[], [], []]
    return dic

def compute_means(dico):
    sums = [0,0,0]
    means = [0,0,0]
    keys = np.array(list(dico.keys()))
    for i in range (len(keys)):
        means = [means[0]+keys[i]*len(dico[keys[i]][0]),means[1]+keys[i]*len(dico[keys[i]][1]),means[2]+keys[i]*len(dico[keys[i]][2])]
        sums = [sums[0]+len(dico[keys[i]][0]),sums[1]+len(dico[keys[i]][1]),sums[2]+len(dico[keys[i]][2])]
    return np.array(means)/np.array(sums)

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

files_csv = glob.glob(os.path.join("/Users/edouard/Desktop/EA p1  HFT/HFT_QR_RL_save/Sans titre/HFT_QR_RL/data/MBO-10 analyse/CSV_dezippés_", "*.csv"))

# %%
dic = {}

for f in tqdm(files_csv):
    MBO_ = pd.read_csv(f)
    MBO = MBO_[MBO_["publisher_id"]==39]
    MBO_filtered = MBO[MBO['symbol'] == "ASAI"]
    MBO_filtered_depth_0_ = MBO_filtered[MBO_filtered['depth'] == 0]
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

Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 10

# visualisation
dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 10, threshold=10)

for i in intensities:
    if len(intensities[i][0])!=0:
        if (len(intensities[i][0])>10):
            Add.append(np.mean(np.array(intensities[i][0])))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>10):
            Cancel.append(np.mean(np.array(intensities[i][1])))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>10):
            Trade.append(np.mean(np.array(intensities[i][2])))
            sizes_trade.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes[0], y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes[1], y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes[2], y = 1/np.array(Trade), mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities ASAI', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()


# %%
dic = {}

for f in tqdm(files_csv):
    MBO_ = pd.read_csv(f)
    MBO = MBO_[MBO_["publisher_id"] == 39]
    MBO_filtered = MBO[MBO['symbol'] == "RIOT"]
    MBO_filtered_depth_0_ = MBO_filtered[MBO_filtered['depth'] == 0]
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


Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 10

# visualisation
dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 10, threshold=10)

for i in intensities:
    if len(intensities[i][0])!=0:
        if (len(intensities[i][0])>10):
            Add.append(np.mean(np.array(intensities[i][0])))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>10):
            Cancel.append(np.mean(np.array(intensities[i][1])))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>10):
            Trade.append(np.mean(np.array(intensities[i][2])))
            sizes_trade.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes[0], y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes[1], y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes[2], y = 1/np.array(Trade), mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities RIOT', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()


# %%
dic = {}

for f in tqdm(files_csv):
    MBO_ = pd.read_csv(f)
    MBO = MBO_[MBO_["publisher_id"]==39]
    MBO_filtered = MBO[MBO['symbol'] == "HL"]
    MBO_filtered_depth_0_ = MBO_filtered[MBO_filtered['depth'] == 0]
    MBO_filtered_depth_0_['bid_sz_00_diff'] = MBO_filtered_depth_0_['bid_sz_00'].diff()
    MBO_filtered_depth_0_['ask_sz_00_diff'] = MBO_filtered_depth_0_['ask_sz_00'].diff()
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[
        ~(
            (MBO_filtered_depth_0_['action'] == 'C') &
            (
                ((MBO_filtered_depth_0_['side'] == 'B') & (MBO_filtered_depth_0_['bid_sz_00_diff'] == 0)) |
                ((MBO_filtered_depth_0_['side'] == 'A') & (MBO_filtered_depth_0_['ask_sz_00_diff'] == 0))
            )
        )
    ]
    MBO_filtered_depth_0 = MBO_filtered_depth_0_.iloc[1:] # on enleve le premier NA
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0['bid_sz_00'].to_numpy())).tolist()+(np.unique(MBO_filtered_depth_0['ask_sz_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0['ts_event'] = pd.to_datetime(MBO_filtered_depth_0['ts_event'])
    MBO_filtered_depth_0['temps_ecoule'] = MBO_filtered_depth_0['ts_event'].diff()
    MBO_filtered_depth_0['temps_ecoule_secondes'] = MBO_filtered_depth_0['temps_ecoule'].dt.total_seconds()
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


Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 10

# visualisation
dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 10, threshold=10)

for i in intensities:
    if len(intensities[i][0])!=0:
        if (len(intensities[i][0])>10):
            Add.append(np.mean(np.array(intensities[i][0])))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>10):
            Cancel.append(np.mean(np.array(intensities[i][1])))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>10):
            Trade.append(np.mean(np.array(intensities[i][2])))
            sizes_trade.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes[0], y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes[1], y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes[2], y = 1/np.array(Trade), mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities HL', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()


# %%
dic = {}

for f in tqdm(files_csv):
    MBO_ = pd.read_csv(f)
    MBO = MBO_[MBO_["publisher_id"]==39]
    MBO_filtered = MBO[MBO['symbol'] == "CGAU"]
    MBO_filtered_depth_0_ = MBO_filtered[MBO_filtered['depth'] == 0]
    MBO_filtered_depth_0_['bid_sz_00_diff'] = MBO_filtered_depth_0_['bid_sz_00'].diff()
    MBO_filtered_depth_0_['ask_sz_00_diff'] = MBO_filtered_depth_0_['ask_sz_00'].diff()
    MBO_filtered_depth_0_ = MBO_filtered_depth_0_[
        ~(
            (MBO_filtered_depth_0_['action'] == 'C') &
            (
                ((MBO_filtered_depth_0_['side'] == 'B') & (MBO_filtered_depth_0_['bid_sz_00_diff'] == 0)) |
                ((MBO_filtered_depth_0_['side'] == 'A') & (MBO_filtered_depth_0_['ask_sz_00_diff'] == 0))
            )
        )
    ]
    MBO_filtered_depth_0 = MBO_filtered_depth_0_.iloc[1:] # on enleve le premier NA
    sizes = np.unique(np.array((np.unique(MBO_filtered_depth_0['bid_sz_00'].to_numpy())).tolist()+(np.unique(MBO_filtered_depth_0['ask_sz_00'].to_numpy())).tolist()))
    MBO_filtered_depth_0['ts_event'] = pd.to_datetime(MBO_filtered_depth_0['ts_event'])
    MBO_filtered_depth_0['temps_ecoule'] = MBO_filtered_depth_0['ts_event'].diff()
    MBO_filtered_depth_0['temps_ecoule_secondes'] = MBO_filtered_depth_0['temps_ecoule'].dt.total_seconds()
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


Add = []
Cancel = []
Trade = []
sizes_add = []
sizes_cancel = []
sizes_trade = []
threshold = 10

# visualisation
dic = remove_nan_from_dico(dic)
average_sizes = compute_means(dic)
intensities = dict(sorted(dic.items()))
intensities = filtrage(intensities, 10, threshold=10)

for i in intensities:
    if len(intensities[i][0])!=0:
        if (len(intensities[i][0])>10):
            Add.append(np.mean(np.array(intensities[i][0])))
            sizes_add.append(i)
    if len(intensities[i][1])!=0:
        if (len(intensities[i][1])>10):
            Cancel.append(np.mean(np.array(intensities[i][1])))
            sizes_cancel.append(i)
    if len(intensities[i][2])!=0:
        if (len(intensities[i][2])>10):
            Trade.append(np.mean(np.array(intensities[i][2])))
            sizes_trade.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x = sizes_add/average_sizes[0], y = 1/np.array(Add), mode ='lines', name ='Add', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_cancel/average_sizes[1], y = 1/np.array(Cancel), mode ='lines', name = 'Cancel', showlegend = True))
fig.add_trace(go.Scatter(x = sizes_trade/average_sizes[2], y = 1/np.array(Trade), mode ='lines', name = 'Trade', showlegend = True))
fig.update_layout(title='Intensities CGAU', xaxis_title='size', yaxis_title='intensity', showlegend=True)
fig.show()

